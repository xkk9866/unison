"""GPTCache and vCache as services in processes of their own.

Both published caches are Python libraries: a GPTCache lookup runs an ORM over sqlite and a FAISS search, a
vCache lookup a nearest-neighbour query and a Bayesian decision. Hosted inside the serving process that code
shares the interpreter lock with the engine's scheduler and with token streaming, which stretches a lookup of
two milliseconds to tens of milliseconds and queues the lookups of a round behind one another. Here the
unmodified libraries run in their own processes (the layout of a cache deployed next to an inference server)
and the executors talk to them over Unix sockets from the event loop, without any thread in the serving process.

  GPTCacheService   SHARDS single-threaded processes (GPTCache's in-memory sqlite store belongs to one thread);
                    a passage's cache lives in the process its id hashes to. Empty caches are created ahead of
                    use, so the first result of a passage is stored without paying for the creation of its cache.
  VCacheService     SHARDS processes with one thread per request in flight (vCache calls the model from inside
                    its decision); the model call is sent back to the serving process, which runs it on the engine.

Measured on the server while the campaign ran, with stub model calls. GPTCache: a lookup on a passage with entries
returns in 8 ms (the library call takes as long when made directly) and the twelve lookups of a round, issued at
once, in 10 to 14 ms on average; in one thread of the serving process each further lookup of a burst returned 6
to 13 ms after the one before it (the v17 main runs of 2026-11-10). vCache: a decision on an entry with
observations fits a logistic model to them and takes 14 to 25 ms; the three decisions a round makes on one
passage share an interpreter, so the twelve lookups of a round return in 40 to 50 ms, with four processes as
with eight.

A service starts empty and is terminated with the executor that owns it.
"""
import asyncio
import atexit
import itertools
import json
import os
import pickle
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from collections import deque

SHARDS = 4     # processes of a service
SPARE = 96     # empty caches each GPTCache process keeps ready


def _pack(obj):
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return struct.pack("<I", len(b)) + b


# ------------------------------------------------------------------------------------------- service processes
def _recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def _recv(conn):
    hdr = _recv_exact(conn, 4)
    if hdr is None:
        return None
    body = _recv_exact(conn, struct.unpack("<I", hdr)[0])
    return None if body is None else pickle.loads(body)


def _listen(path):
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    return srv


def _own_stdout():
    """The pipe to the executor carries one line, "ready"; whatever a library prints goes to stderr (the job log)."""
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    return out


def _accept(srv, out):
    """Announce readiness, then wait for the executor."""
    out.write("ready\n")
    out.close()
    srv.settimeout(600)
    try:
        conn, _ = srv.accept()
    except socket.timeout:
        os._exit(0)
    conn.settimeout(None)
    return conn


def serve_gptcache(path, opts):
    """GPTCache (official library): its data manager (sqlite + FAISS), its SearchDistanceEvaluation and its
    default similarity threshold. One cache per passage, keyed by the sub-question."""
    import select
    out = _own_stdout()
    from gptcache import Cache, Config
    from gptcache.adapter.api import get, put
    from gptcache.manager import CacheBase, VectorBase, get_data_manager
    from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
    srv = _listen(path)
    emb, made = {}, itertools.count()

    def make():
        # GPTCache persists a FAISS index to a file and loads that file when a cache is created; every cache gets
        # its own file in a directory of this run, so a cache starts empty
        c = Cache()
        index_path = os.path.join(opts["index_dir"], f"{os.getpid()}_{next(made)}.index")
        c.init(pre_embedding_func=lambda data, **kw: data["prompt"], embedding_func=lambda key, **kw: emb[key],
               data_manager=get_data_manager(CacheBase("sqlite", sql_url="sqlite:///:memory:"),
                                             VectorBase("faiss", dimension=opts["dim"], index_path=index_path)),
               similarity_evaluation=SearchDistanceEvaluation(), config=Config())
        return c

    # A cache is a few thousand ORM objects, and a full pass of the cyclic collector over the caches of a run
    # takes a large fraction of a second. The collector therefore never runs inside a request: it is off, and the
    # young generations are collected when the process is idle, after which the survivors are set aside.
    import gc
    gc.disable()
    spare = [make() for _ in range(opts["spare"])]
    gc.collect()
    gc.freeze()
    caches = {}
    conn = _accept(srv, out)
    while True:
        if len(spare) < opts["spare"] and not select.select([conn], [], [], 0)[0]:
            spare.append(make())  # nothing to serve: restock
            gc.collect(1)
            gc.freeze()
            continue
        msg = _recv(conn)
        if msg is None:
            os._exit(0)
        op, pid, key, vec, value = msg
        t = time.perf_counter()
        out = None
        c = caches.get(pid)
        if c is None and op == "put":
            c = caches[pid] = spare.pop() if spare else make()
        if c is not None:  # a passage without a cache has no entry: a miss
            emb[key] = vec
            out = get(key, cache_obj=c) if op == "get" else put(key, value, cache_obj=c)
            del emb[key]
        conn.sendall(_pack((out if op == "get" else None, time.perf_counter() - t)))


def serve_vcache(path, opts):
    """vCache (official implementation): VerifiedDecisionPolicy with its default delta, HNSW vector store,
    in-memory metadata. One cache per passage, keyed by the sub-question. The model call of an exploration or a
    miss is a message to the serving process; the label of an exploration compares the decisions of two notes,
    which the serving process sends along with each note."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    out = _own_stdout()
    from unison.ext_vcache import load
    V = load()
    srv = _listen(path)
    emb, dec, caches, waiting = {}, {}, {}, {}
    tl = threading.local()
    wlock, clock = threading.Lock(), threading.Lock()
    pool = ThreadPoolExecutor(max_workers=opts["threads"])
    conn = None

    def send(obj):
        b = _pack(obj)
        with wlock:
            conn.sendall(b)

    class Engine(V.InferenceEngine):
        def create(self, prompt, system_prompt=None):
            slot = waiting[tl.rid] = [threading.Event(), None]
            send(("need", tl.rid))
            slot[0].wait()
            del waiting[tl.rid]
            tl.engine = True
            return slot[1]

    class Emb(V.EmbeddingEngine):
        def get_embedding(self, text):
            return emb[text].tolist()

    class Judge(V.SimilarityEvaluator):
        def answers_similar(self, a, b, id_set_a=None, id_set_b=None):
            return dec[a] == dec[b]

    def cache(pid):
        c = caches.get(pid)
        if c is None:
            with clock:
                c = caches.get(pid)
                if c is None:
                    cfg = V.VCacheConfig(inference_engine=Engine(), embedding_engine=Emb(),
                                         vector_db=V.HNSWLibVectorDB(),
                                         embedding_metadata_storage=V.InMemoryEmbeddingMetadataStorage(),
                                         similarity_evaluator=Judge())
                    c = caches[pid] = V.VCache(cfg, V.VerifiedDecisionPolicy(delta=opts["delta"]))
        return c

    def infer(rid, pid, key):
        tl.rid, tl.engine = rid, False
        try:
            t = time.perf_counter()
            hit, resp, _, _ = cache(pid).infer_with_cache_info(key)
            send(("done", rid, bool(hit), resp, tl.engine, time.perf_counter() - t))
        except Exception as e:  # reported to the caller, which fails the run
            send(("error", rid, repr(e)))

    conn = _accept(srv, out)
    while True:
        msg = _recv(conn)
        if msg is None:
            os._exit(0)
        if msg[0] == "infer":
            _, rid, pid, key, vec = msg
            emb[key] = vec
            pool.submit(infer, rid, pid, key)
        else:  # ("resp", rid, note, decision): the model's answer to a "need"
            _, rid, resp, d = msg
            dec[resp] = d
            slot = waiting[rid]
            slot[1] = resp
            slot[0].set()


# ---------------------------------------------------------------------------------------------- executor side
_LIVE = []


def close_all():
    """Terminate the services of earlier executors (the warm-up pass, the previous method)."""
    for s in list(_LIVE):
        s.close()


atexit.register(close_all)


class _Proc:
    """One service process and the stream to it."""

    def __init__(self, kind, opts):
        self.dir = tempfile.mkdtemp(prefix=f"unison_{kind}_")
        self.path = os.path.join(self.dir, "s.sock")
        code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
        env["PYTHONPATH"] = code + os.pathsep + env.get("PYTHONPATH", "")
        self.p = subprocess.Popen([sys.executable, "-m", "unison.cache_service", kind, self.path,
                                   json.dumps(dict(opts, index_dir=self.dir))],
                                  cwd=code, env=env, stdout=subprocess.PIPE, text=True)
        self.sock = self.reader = self.writer = self.task = None
        self.lock = asyncio.Lock()

    def wait_ready(self):
        line = self.p.stdout.readline()
        self.p.stdout.close()
        if line.strip() != "ready":
            raise RuntimeError(f"cache service did not start (exit code {self.p.poll()})")
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.path)

    async def open(self, on_message):
        if self.writer is None:
            async with self.lock:
                if self.writer is None:
                    reader, writer = await asyncio.open_unix_connection(sock=self.sock)
                    self.task = asyncio.create_task(self._read(reader, on_message))
                    self.reader, self.writer = reader, writer

    async def _read(self, reader, on_message):
        try:
            while True:
                n = struct.unpack("<I", await reader.readexactly(4))[0]
                on_message(pickle.loads(await reader.readexactly(n)))
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.CancelledError):
            on_message(None)

    def send(self, obj):
        self.writer.write(_pack(obj))

    def close(self):
        for f in (getattr(self.task, "cancel", None), getattr(self.writer, "close", None),
                  getattr(self.sock, "close", None)):
            try:  # at interpreter exit the event loop is already closed
                if f is not None:
                    f()
            except Exception:
                pass
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.p.kill()
        shutil.rmtree(self.dir, ignore_errors=True)


class GPTCacheService:
    def __init__(self, shards=SHARDS, spare=SPARE, dim=768):
        close_all()
        self.procs = [_Proc("gptcache", dict(spare=spare, dim=dim)) for _ in range(shards)]
        self.queues = [deque() for _ in self.procs]  # a process answers in the order it is asked
        for p in self.procs:
            p.wait_ready()
        _LIVE.append(self)

    def _on(self, i, msg):
        if msg is None:  # the process is gone
            while self.queues[i]:
                f = self.queues[i].popleft()
                if not f.done():
                    f.set_exception(RuntimeError("GPTCache service ended"))
            return
        f = self.queues[i].popleft()
        if not f.done():
            f.set_result(msg)

    async def _call(self, op, pid, key, vec, value=None):
        i = zlib.crc32(repr(pid).encode()) % len(self.procs)
        await self.procs[i].open(lambda m, i=i: self._on(i, m))
        f = asyncio.get_running_loop().create_future()
        self.queues[i].append(f)
        self.procs[i].send((op, pid, key, vec, value))
        return await f

    async def get(self, pid, key, vec):
        """(cached value or None, seconds the lookup took inside the service)."""
        return await self._call("get", pid, key, vec)

    async def put(self, pid, key, vec, value):
        return (await self._call("put", pid, key, vec, value))[1]

    def close(self):
        for p in self.procs:
            p.close()
        if self in _LIVE:
            _LIVE.remove(self)


class VCacheService:
    def __init__(self, delta=0.02, shards=SHARDS, threads=256):
        close_all()
        self.procs = [_Proc("vcache", dict(delta=delta, threads=threads)) for _ in range(shards)]
        for p in self.procs:
            p.wait_ready()
        self.pending, self.ids, self.tasks = {}, itertools.count(), set()
        _LIVE.append(self)

    def _on(self, i, msg):
        if msg is None:
            for f, _, j in list(self.pending.values()):
                if j == i and not f.done():
                    f.set_exception(RuntimeError("vCache service ended"))
            return
        if msg[1] not in self.pending:
            return
        f, run_engine, _ = self.pending[msg[1]]
        if msg[0] == "need":
            t = asyncio.create_task(self._need(i, msg[1], f, run_engine))
            self.tasks.add(t)
            t.add_done_callback(self.tasks.discard)
        elif msg[0] == "done" and not f.done():
            f.set_result(tuple(msg[2:]))
        elif msg[0] == "error" and not f.done():
            f.set_exception(RuntimeError(f"vCache: {msg[2]}"))

    async def _need(self, i, rid, f, run_engine):
        try:
            resp, decision = await run_engine()
        except BaseException as e:
            if not f.done():
                f.set_exception(e if isinstance(e, Exception) else RuntimeError(repr(e)))
            raise
        self.procs[i].send(("resp", rid, resp, decision))

    async def infer(self, pid, key, vec, run_engine):
        """(hit, response, the model was called, seconds inside the service including the model call).
        `run_engine` is awaited when vCache calls the model; it returns the note as vCache stores it and the
        decision the note states."""
        i = zlib.crc32(repr(pid).encode()) % len(self.procs)
        await self.procs[i].open(lambda m, i=i: self._on(i, m))
        rid = next(self.ids)
        f = asyncio.get_running_loop().create_future()
        self.pending[rid] = (f, run_engine, i)
        self.procs[i].send(("infer", rid, pid, key, vec))
        try:
            return await f
        finally:
            del self.pending[rid]

    def close(self):
        for p in self.procs:
            p.close()
        if self in _LIVE:
            _LIVE.remove(self)


if __name__ == "__main__":
    {"gptcache": serve_gptcache, "vcache": serve_vcache}[sys.argv[1]](sys.argv[2], json.loads(sys.argv[3]))
