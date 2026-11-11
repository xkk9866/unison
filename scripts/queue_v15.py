"""Run the v15 manuscript suite on the authorized GPU pool (one job per idle GPU; resumable).

Every card of the pool is polled; a timing-sensitive job starts on a card whose co-tenants have been idle
(utilization < 10%) for IDLE_SECONDS and that has NEED[tier] + HEADROOM MiB free; timing-free jobs (smoke,
edit stress test, audits) need only the memory. A job never shares a GPU with another job of ours. vLLM's
memory fraction is derived from co-tenant memory plus our allowance (vLLM charges co-tenant memory against it).
Smoke must pass before manuscript jobs start. Jobs whose summary exists are skipped (restartable).
  python queue_v15.py            run everything
  python queue_v15.py list       print the job list
"""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(os.environ.get('UNISON_ROOT', '/home/yaokang/evishare_fast'))  # code, data, results, logs, .venv
CODE, LOG, RES = ROOT / 'code', ROOT / 'logs', ROOT / 'results'
LOG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)
PY = str(ROOT / '.venv/bin/python')
LOCK = threading.RLock()  # re-entrant: log() is called while STATE is being updated
STOP = threading.Event()
STATE = {}
ALLOWED_GPUS = (0, 1, 2, 3, 4)  # the cards this project may use; 5, 6 and 7 are never touched
GPUS = tuple(g for g in (int(x) for x in os.environ.get('UNISON_GPUS', '0,1,2,3,4').split(',')) if g in ALLOWED_GPUS)
IDLE_SECONDS = 10
HEADROOM = 3000     # MiB a job takes beyond its allowance, outside the engine's allocator pool: CUDA graphs and
                    # library workspaces of the engine process (measured 2.1 GB) and the driver's encoder (0.8 GB)
V = os.environ.get('UNISON_CAMPAIGN', 'v15')
U = os.environ.get('UNISON_U', 'u17' if V >= 'v17' else 'u15')
# KV-cache capacity is *fixed* per tier (blocks of 16 tokens) so that every run of a tier has the same capacity
# whatever the memory situation of its card: 3000 blocks = 48k tokens for every timing-sensitive run (above the
# largest concurrent footprint of any system in the main streams), 2000 blocks for timing-free runs.
KV_BLOCKS = {'load': 3000, 'timed': 3000, 'free': 2000}
KV_MIB_PER_BLOCK = 2.25   # Qwen3-8B: 36 layers x 2 x 8 heads x 128 x 2 bytes x 16 tokens
PEAK_MIB = 9300           # measured non-KV peak of the engine (weights, drafter, profile run)
# Our own memory allowance per tier. vLLM 0.8.5 charges co-tenant memory against gpu_memory_utilization (non-torch
# allocations count towards the profiled peak), so the fraction passed to the engine is
# (co-tenant memory + our allowance) / card memory.
NEED = {t: int(PEAK_MIB + b * KV_MIB_PER_BLOCK) for t, b in KV_BLOCKS.items()}
# Jobs whose reported quantities do not depend on wall-clock time (smoke invariants, the version-safety stress
# test's error/cross-version counts, the rationale audits). They may run next to a busy co-tenant as long as the
# engine fits; co-tenant memory is recorded in every summary.
TIMING_FREE = {f'{V}_smoke', f'{V}_csmoke', f'{V}_edit', f'{V}_collect', f'{V}_collect2', 'train', 'audit', 'auditw'}


def tier(job):
    if job[0] in TIMING_FREE:
        return 'free'
    return 'load' if '_load_' in job[0] or '_wload_' in job[0] else 'timed'


def log(message):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + message
    with LOCK:
        print(line, flush=True)
        with (LOG / f'queue_{V}.log').open('a') as f:
            f.write(line + '\n')


def gpu_state(gpu):
    out = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.free,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    free, total, util = map(int, out.strip().split(','))
    return free, total, util


def budget(gpu, job):
    """gpu_memory_utilization for this job on this card: memory held by others (co-tenants or our own holder) plus
    our allowance, as a card fraction."""
    free, total, _ = gpu_state(gpu)
    return round(max(0.3, min(0.97, (total - free + NEED[tier(job)]) / total)), 2)


def min_free_for(job):
    return NEED[tier(job)] + HEADROOM


def baseline_jobs():
    """Head-to-head runs against published systems only. The main streams already contain this set."""
    base = ['--n', '200', '--seed', '0', '--qsplit', 'test']
    methods = 'noshare,pld,suffix,gptcache,vcacheoff,u15'
    out = []
    for ds in ['hotpotqa', 'musique']:
        rate = '1.2' if ds == 'hotpotqa' else '0.6'
        out.append((f'{V}_wide', ds, ['--methods', methods, '--rate', rate, '--mu', '0.1',
                                       '--note', 'dfpw'] + base, None))
    for rate in ['0.6', '1.8', '2.4', '3.0']:
        out.append((f'{V}_load_r' + rate, 'hotpotqa', ['--methods', methods, '--rate', rate, '--mu', '0.1',
                                                         '--note', 'dfp'] + base, None))
    out.append((f'{V}_edit', 'hotpotqa', ['--methods', methods, '--rate', '1.2', '--mu', '1.0',
                                           '--note', 'dfp', '--versions', 'versions_edit.jsonl'] + base, None))
    for z in ['0', '1.2']:
        out.append((f'{V}_zipf_' + z, 'hotpotqa', ['--methods', methods, '--rate', '1.2', '--mu', '0.1',
                                                     '--note', 'dfp', '--zipf', z] + base, None))
    for rate in ['1.8', '2.4']:
        out.append((f'{V}_wload_r' + rate, 'hotpotqa', ['--methods', methods, '--rate', rate, '--mu', '0.1',
                                                          '--note', 'dfpw'] + base, None))
    return out


def v17_jobs():
    """Consensus-drafting campaign: head-to-head against published systems in every stream; one ablation job per
    dataset for the method's own variants."""
    base = ['--n', '200', '--seed', '0', '--qsplit', 'test']
    # the comparison set: published systems run from their own code or from the engine's implementation
    pub = f'noshare,pld,suffix,gptcache,vcacheoff,{U}'
    # {U}-suf is the layer with SuffixDecoding as the proposer (it differs from Unison in the drafts only);
    # {U}-sbd additionally writes one rationale per call (the reference of the rationale audit)
    abl = f'{U}-suf,{U}-sbd,{U}-rule,{U}-nopre,{U}-nopeer,{U}-noctx,{U}-noh,{U}-noplan,{U}-noid,{U}-nodf,{U}'
    out = []
    # training data for the consensus drafter (train pools, no sharing), its training, then the smoke invariants
    for i, ds in enumerate(['hotpotqa', 'musique']):
        rate = '1.2' if ds == 'hotpotqa' else '0.6'
        out.append((f'{V}_collect', ds, ['--methods', 'collect', '--split', 'train', '--n', '400', '--rate', rate,
                                          '--mu', '0.1', '--note', 'dfp', '--seed', '1', '--replay', '0', '--warmup', '0',
                                          '--collect', str(RES / f'{V}_collect' / ds / 'shards')],
                    None if i == 0 else (f'{V}_collect', 'hotpotqa')))
    out.append(('train', 'all', [f'{V}_collect/hotpotqa', f'{V}_collect/musique', f'{V}_collect2/hotpotqa',
                                 f'{V}_collect2/musique'], (f'{V}_collect2', 'musique')))
    out.append((f'{V}_smoke', 'hotpotqa', ['--methods', f'suffix,{U}-sbd,{U}-nopre,{U}-nopeer,{U}-noid,{U}', '--n', '20',
                                            '--rate', '1.2', '--mu', '0.1', '--seed', '0', '--qsplit', 'test',
                                            '--note', 'dfp'],
                ('train', 'all')))
    # a second pass over the train pools with uniform question popularity: it reaches the questions the first pass
    # (the test stream's popularity skew) left out, so the drafter is trained on about twice as many questions
    for ds in ['hotpotqa', 'musique']:
        rate = '1.2' if ds == 'hotpotqa' else '0.6'
        out.append((f'{V}_collect2', ds, ['--methods', 'collect', '--split', 'train', '--n', '800', '--rate', rate,
                                           '--mu', '0.1', '--note', 'dfp', '--seed', '2', '--zipf', '0', '--replay', '0',
                                           '--warmup', '0', '--collect', str(RES / f'{V}_collect2' / ds / 'shards')],
                    (f'{V}_collect', 'musique')))
    gate = (f'{V}_smoke', 'hotpotqa')
    # the two cache services on a short stream (hits, lookup times, clean shutdown) before the streams that use them
    out.append((f'{V}_csmoke', 'hotpotqa', ['--methods', 'gptcache,vcacheoff', '--n', '40', '--rate', '1.2',
                                             '--mu', '0.1', '--seed', '0', '--qsplit', 'test', '--note', 'dfp'], gate))
    for ds in ['hotpotqa', 'musique']:
        rate = '1.2' if ds == 'hotpotqa' else '0.6'
        out.append((f'{V}_main', ds, ['--methods', pub, '--rate', rate, '--mu', '0.1', '--note', 'dfp',
                                       '--replay2', '1'] + base, gate))
    for ds in ['hotpotqa', 'musique']:
        rate = '1.2' if ds == 'hotpotqa' else '0.6'
        out.append((f'{V}_abl', ds, ['--methods', abl, '--rate', rate, '--mu', '0.1', '--note', 'dfp'] + base, gate))
    for ds in ['hotpotqa', 'musique']:
        rate = '1.2' if ds == 'hotpotqa' else '0.6'
        out.append((f'{V}_wide', ds, ['--methods', pub, '--rate', rate, '--mu', '0.1', '--note', 'dfpw'] + base, gate))
    for rate in ['1.8', '2.4', '3.0', '0.6']:
        out.append((f'{V}_load_r' + rate, 'hotpotqa', ['--methods', pub, '--rate', rate, '--mu', '0.1',
                                                         '--note', 'dfp'] + base, gate))
    # the verification price at work: drafting variants under a filling and a saturated engine
    for rate in ['2.4', '3.0']:
        out.append((f'{V}_abload_r' + rate, 'hotpotqa', ['--methods', f'{U}-suf,{U}-rule,{U}-nopre,{U}', '--rate', rate,
                                                           '--mu', '0.1', '--note', 'dfp'] + base, gate))
    # the same attribution with the wide rationale (the regime in which a longer payload competes for the engine)
    for ds in ['hotpotqa', 'musique']:
        rate = '1.2' if ds == 'hotpotqa' else '0.6'
        out.append((f'{V}_wabl', ds, ['--methods', f'{U}-suf,{U}-nopre,{U}', '--rate', rate, '--mu', '0.1',
                                       '--note', 'dfpw'] + base, gate))
    for rate in ['0.9', '1.2']:
        out.append((f'{V}_load_r' + rate, 'musique', ['--methods', pub, '--rate', rate, '--mu', '0.1',
                                                        '--note', 'dfp'] + base, gate))
    out.append((f'{V}_edit', 'hotpotqa', ['--methods', f'gptcache,vcacheoff,{U}', '--rate', '1.2',
                                           '--mu', '1.0', '--note', 'dfp', '--versions', 'versions_edit.jsonl'] + base, gate))
    for z in ['0', '1.2']:
        out.append((f'{V}_zipf_' + z, 'hotpotqa', ['--methods', f'suffix,gptcache,{U}', '--rate', '1.2',
                                                     '--mu', '0.1', '--note', 'dfp', '--zipf', z] + base, gate))
    for rate in ['1.8', '2.4']:
        out.append((f'{V}_wload_r' + rate, 'hotpotqa', ['--methods', f'suffix,gptcache,{U}', '--rate', rate,
                                                          '--mu', '0.1', '--note', 'dfpw'] + base, gate))
    for ds in ['hotpotqa', 'musique']:
        out.append(('audit', ds, [f'{V}_abl'], (f'{V}_abl', ds)))
    # the main comparison on two more streams (other arrival times and another draw of the questions)
    for seed in ['1', '2']:
        for ds in ['hotpotqa', 'musique']:
            rate = '1.2' if ds == 'hotpotqa' else '0.6'
            out.append((f'{V}_main_s{seed}', ds, ['--methods', pub, '--rate', rate, '--mu', '0.1', '--note', 'dfp',
                                                  '--n', '200', '--seed', seed, '--qsplit', 'test'], gate))
    # MuSiQue at half the main rate: no system queues, so the comparison is one of service time alone
    out.append((f'{V}_load_r0.3', 'musique', ['--methods', pub, '--rate', '0.3', '--mu', '0.1', '--note', 'dfp'] + base, gate))
    return out


def jobs():
    if os.environ.get('UNISON_SUITE') == 'baselines':
        return baseline_jobs()
    if V >= 'v17':
        return v17_jobs()
    base = ['--n', '200', '--seed', '0', '--qsplit', 'test']
    main = f'noshare,pld,suffix,gptcache,vcacheoff,lacache,{U}-sbd,{U},noshare@cot'
    abl = f'{U}-lazy,{U}-sbd,{U}-coop,{U}-keyed,{U},{U}-noid,{U}-nodf,{U}-nolive,{U}-noprio'
    wide = f'noshare,suffix,lacache,{U}-sbd,{U}'
    load = f'suffix,gptcache,{U}-sbd,{U}'
    out = []
    for ds in ['hotpotqa', 'musique']:
        rate = '1.2' if ds == 'hotpotqa' else '0.6'
        out.append((f'{V}_main', ds, ['--methods', main, '--rate', rate, '--mu', '0.1', '--note', 'dfp',
                                       '--replay2', '1'] + base, None))
    for ds in ['hotpotqa', 'musique']:
        rate = '1.2' if ds == 'hotpotqa' else '0.6'
        out.append((f'{V}_abl', ds, ['--methods', abl, '--rate', rate, '--mu', '0.1', '--note', 'dfp'] + base, None))
    for ds in ['hotpotqa', 'musique']:
        rate = '1.2' if ds == 'hotpotqa' else '0.6'
        out.append((f'{V}_wide', ds, ['--methods', wide, '--rate', rate, '--mu', '0.1', '--note', 'dfpw'] + base, None))
    for rate in ['1.8', '2.4', '3.0', '0.6']:
        out.append((f'{V}_load_r' + rate, 'hotpotqa', ['--methods', load, '--rate', rate, '--mu', '0.1',
                                                         '--note', 'dfp'] + base, None))
    out.append((f'{V}_edit', 'hotpotqa', ['--methods', f'gptcache,vcacheoff,lacache,{U}-sbd,{U}', '--rate', '1.2',
                                           '--mu', '1.0', '--note', 'dfp', '--versions', 'versions_edit.jsonl'] + base, None))
    for rate in ['1.8', '2.4', '3.0']:
        out.append((f'{V}_wload_r' + rate, 'hotpotqa', ['--methods', f'{U}-sbd,{U}', '--rate', rate, '--mu', '0.1',
                                                          '--note', 'dfpw'] + base, None))
    for z in ['0', '1.2']:
        out.append((f'{V}_zipf_' + z, 'hotpotqa', ['--methods', f'suffix,lacache,{U}-sbd,{U}', '--rate', '1.2',
                                                     '--mu', '0.1', '--note', 'dfp', '--zipf', z] + base, None))
    for ds in ['hotpotqa', 'musique']:
        out.append(('audit', ds, [f'{V}_abl'], (f'{V}_abl', ds)))
    for ds in ['hotpotqa', 'musique']:
        out.append(('auditw', ds, [f'{V}_wide'], (f'{V}_wide', ds)))
    return out


def output(job):
    tag, ds, args, _ = job
    if tag in ('audit', 'auditw'):
        return RES / args[0] / ds / f'audit_{U}_vs_{U}-sbd.json'
    if tag == 'train':
        return RES / 'drafter' / 'eval.json'
    return RES / tag / ds / 'summary.json'


EMPTY_MIB = 1500          # a card with at most this much memory in use counts as empty (reservable)
CALM_UTIL = 60            # a timed job may start next to co-tenants whose SM utilization stays below this (%);
                          # the run records their activity and is discarded only if they compute heavily (below)
COTENANT_SM = 40          # co-tenant SM share (%) that counts as "busy" in the run-time monitor ...
COTENANT_BUSY = 0.6       # ... and the fraction of busy samples that spoils (aborts / invalidates) a timed run
NOISY_SM = 5              # a finished timed run whose mean co-tenant SM share exceeds this is kept but repeated on
                          # an empty card when one becomes available (polish phase); the cleaner run is kept


def contaminated(job):
    """A finished timing-sensitive job whose summary shows co-tenant compute (SM activity in more than
    COTENANT_BUSY of the samples of any method) or a co-tenant memory change of more than 1 GB between methods."""
    if tier(job) == 'free' or job[0] in ('audit', 'auditw'):
        return None
    try:
        S = json.loads(output(job).read_text())
    except Exception as e:
        return f'unreadable summary ({e})'
    busy = max((r.get('gpu_others_sm_busy', 0) for r in S), default=0)
    if busy > COTENANT_BUSY:
        return f'co-tenant SM busy fraction {busy:.2f}'
    for r in S:
        st = stalled(r)
        if st:
            return f'host stall during {r.get("method")} ({st})'
    return None  # co-tenant *memory* alone (idle processes) does not perturb timing; SM activity is recorded


STALL_S = 3.0             # a timed run whose event loop was stopped for longer than this by the host (memory
                          # pressure on the machine) is aborted and repeated: its arrivals were not delivered on time
HOST_FREE_GIB = 16        # no job of ours starts while the host has less available memory than this


def stalled(r):
    """Evidence in a method's summary that the host stopped the process: the largest event-loop lag where it was
    recorded, otherwise a mean lag far above what its 99th percentile allows (one long stop among short lags)."""
    if 'loop_lag_max_ms' in r:
        return f'{r["loop_lag_max_ms"] / 1e3:.1f}s' if r['loop_lag_max_ms'] > 1e3 * STALL_S else None
    mean, p99 = r.get('loop_lag_ms', 0.0), r.get('loop_lag_p99_ms', 0.0)
    return f'mean lag {mean:.1f}ms at p99 {p99:.0f}ms' if mean > 5 and mean > 0.25 * p99 else None


JOB_HOST_GIB = 12         # host memory one job of ours reaches at its peak (driver, engine process, the cache
                          # services of one method); it gets there minutes after its start
RECENT_STARTS = []        # monotonic start times of our jobs that may not have reached that peak yet


def host_free_gib():
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return 1e9


def host_need_gib():
    """Available host memory a new job of ours needs to see: the floor plus what our recent starts will still take."""
    now = time.monotonic()
    with LOCK:
        RECENT_STARTS[:] = [t for t in RECENT_STARTS if now - t < 900]
        return HOST_FREE_GIB + JOB_HOST_GIB * len(RECENT_STARTS)


def host_admit():
    """Register a start if the host has the memory for it (atomic over the workers); None otherwise."""
    now = time.monotonic()
    with LOCK:
        RECENT_STARTS[:] = [t for t in RECENT_STARTS if now - t < 900]
        if host_free_gib() < HOST_FREE_GIB + JOB_HOST_GIB * len(RECENT_STARTS):
            return None
        RECENT_STARTS.append(now)
        return now


def host_cancel(tok):
    with LOCK:
        if tok in RECENT_STARTS:
            RECENT_STARTS.remove(tok)


VISIBLE = 1400           # MiB a card of ours may show free: the schedulers of other projects on this machine place
                         # a job where they see 2.1 GB, and a job that arrives next to a measurement spoils it.
                         # They compute total - used, which is 470 MiB above the free memory a process sees
                         # (1900 was not low enough: a job was placed at 1715 MiB free)
HOLD_LEAVE = 700         # MiB a holder leaves free while no job of ours runs on the card
COTENANT_GROWTH = VISIBLE - HOLD_LEAVE  # extra MiB left free on a card that already hosts someone else's process
RUN_HEADROOM = VISIBLE   # MiB left free next to a running engine of ours; the engine's own slack (prefill
                         # activations) is a free block inside its allocator pool (plugin, memory hand-over)
GRAB_MARGIN = 3000       # extra MiB the engine process takes over at start-up when the card has them


def flag_path(gpu):
    return LOG / f'hold_gpu{gpu}.flag'


def proc_mem(gpu, pid):
    """MiB of GPU memory held by `pid` on this card (0 if none)."""
    out = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid,used_memory',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    for line in out.splitlines():
        p = line.split(',')
        if len(p) == 2 and p[0].strip().isdigit() and int(p[0]) == pid:
            return int(p[1])
    return 0


def start_holder(gpu, leave=HOLD_LEAVE):
    """Hold the memory of a card (see hold_gpu.py): everything but `leave` MiB while no job of ours runs; the
    hand-over protocol frees exactly what a job's engine needs for the seconds of its allocation."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    fp = flag_path(gpu)
    if fp.exists():
        fp.unlink()
    h = subprocess.Popen([PY, 'scripts/hold_gpu.py', str(leave), str(fp)], cwd=CODE, env=env,
                         stdout=(LOG / f'hold_gpu{gpu}.out').open('a'), stderr=subprocess.STDOUT)
    h.leave = leave
    log(f'HOLD gpu={gpu} pid={h.pid} (leaving {leave} MiB)')
    return h


class Adopted:
    """A holder process started by a previous queue process (survives queue restarts)."""

    def __init__(self, pid):
        self.pid = pid

    def terminate(self):
        import signal
        os.kill(self.pid, signal.SIGTERM)

    def kill(self):
        import signal
        os.kill(self.pid, signal.SIGKILL)

    def wait(self, timeout=20):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if not Path(f'/proc/{self.pid}').exists():
                return 0
            time.sleep(0.5)
        raise subprocess.TimeoutExpired('hold_gpu', timeout)


def _procs_on(gpu, needle):
    """pids of processes whose command line contains `needle` and whose CUDA_VISIBLE_DEVICES is this card."""
    out = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        if needle in line and 'grep' not in line:
            pid = int(line.split()[0])
            try:
                envb = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
            except OSError:
                continue
            if f'CUDA_VISIBLE_DEVICES={gpu}'.encode() in envb:
                found.append(pid)
    return found


def adopt_holder(gpu):
    """An orphaned hold_gpu.py on this card, if any."""
    for pid in _procs_on(gpu, 'scripts/hold_gpu.py'):
        log(f'ADOPT gpu={gpu} holder pid={pid}')
        return Adopted(pid)
    return None


def our_job_on(gpu):
    return bool(_procs_on(gpu, 'scripts/run_exp.py') or _procs_on(gpu, 'scripts/audit_eval.py'))


def idle_hold(gpu, holder):
    """Between two jobs the holder keeps the card: all of it but HOLD_LEAVE when nobody else is on it, all but
    VISIBLE next to someone else's process (told through the hand-over flag, see hold_gpu.py)."""
    fp = flag_path(gpu)
    try:
        free, total, _ = gpu_state(gpu)
        others = total - free - proc_mem(gpu, holder.pid)
        want = f'resume {HOLD_LEAVE if others <= EMPTY_MIB else VISIBLE}'
        if not fp.exists() or fp.read_text().strip() != want:
            fp.write_text(want)
    except (OSError, ValueError, subprocess.CalledProcessError):
        if fp.exists():
            fp.unlink()


def stop_holder(gpu, h):
    if h is None:
        return
    h.terminate()
    try:
        h.wait(timeout=20)
    except subprocess.TimeoutExpired:
        h.kill()
    log(f'RELEASE gpu={gpu} pid={h.pid}')


def execute(gpu, job, holder=None):
    tag, ds, args, _ = job
    t = tier(job)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), VLLM_USE_V1='1',
        UNISON_PRIO='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
        UNISON_KV_BLOCKS=str(KV_BLOCKS[t]))
    if holder is not None:
        # the holder's memory is ours (not a co-tenant). The engine process takes it over itself (plugin,
        # UNISON_HOLD_FLAG protocol) and keeps it as a free block of its own allocator pool, so the card never
        # shows more than VISIBLE MiB free. audit_eval runs vLLM too, so the same protocol applies.
        free0, total0, _ = gpu_state(gpu)
        avail = free0 + proc_mem(gpu, holder.pid)
        want = min(avail - 400, NEED[t] + HEADROOM + GRAB_MARGIN)
        env.update(UNISON_HOLDER_PID=str(holder.pid), UNISON_HOLD_FLAG=str(flag_path(gpu)),
                   UNISON_RUN_HEADROOM_MIB=str(RUN_HEADROOM), UNISON_VISIBLE_MIB=str(VISIBLE),
                   UNISON_ENGINE_WANT_MIB=str(int(want)), UNISON_POOL_MIB=str(NEED[t]))
    env.update(UNISON_COTENANT_SM=str(COTENANT_SM))
    if t != 'free':
        env.update(UNISON_ABORT_COTENANT_BUSY=str(COTENANT_BUSY), UNISON_ABORT_STALL_S=str(STALL_S))
    free, total, _ = gpu_state(gpu)
    used = total - free
    util = f'{budget(gpu, job):.2f}'
    if tag in ('audit', 'auditw'):
        cmd = [PY, 'scripts/audit_eval.py', '--tag', args[0], '--ds', ds,
               '--a', U, '--b', f'{U}-sbd', '--n', '400', '--gpu_util', util]
    else:
        cmd = [PY, 'scripts/run_exp.py', '--tag', tag, '--ds', ds, *args, '--gpu_util', util]
    path = LOG / f'{tag}_{ds}_gpu{gpu}_{int(time.time())}.log'
    polish = (tag, ds) in REQUIRE_EMPTY and output(job).exists()
    prev = None
    if polish:  # rerun of a noisy result: keep the old summary aside, restore it if the new one is not cleaner
        prev = output(job).with_name(f'summary.noisy.{int(time.time())}.json')
        output(job).rename(prev)
    log(f'START gpu={gpu} util={util} kv_blocks={KV_BLOCKS[t]} held={int(holder is not None)} used={used} '
        f'{"polish " if polish else ""}{tag}/{ds} log={path}')
    try:
        with path.open('w') as f:
            rc = subprocess.run(cmd, cwd=CODE, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    finally:
        if holder is not None:  # the holder returns to its idle hold level
            idle_hold(gpu, holder)
    if polish:
        clean = rc == 0 and output(job).exists() and not contaminated(job)
        new_noise, old_noise = (noise(job) if clean else None), noise_file(prev)
        if new_noise is not None and new_noise <= old_noise:
            log(f'POLISHED {tag}/{ds}: co-tenant SM {new_noise:.0f}% (was {old_noise:.0f}%)')
            return True, False
        if output(job).exists():
            output(job).rename(output(job).with_name(f'summary.discarded.{int(time.time())}.json'))
        prev.rename(output(job))
        log(f'POLISH_FAILED {tag}/{ds}: previous result restored')
        return True, False
    if rc == 0 and tag == f'{V}_smoke':
        checker = 'scripts/check_smoke17.py' if U == 'u17' else 'scripts/check_smoke15.py'
        check = subprocess.run([PY, checker, str(output(job)), str(path)],
                               cwd=CODE, capture_output=True, text=True)
        log(check.stdout.strip() + check.stderr.strip())
        rc = check.returncode
    if rc != 0 or not output(job).exists():
        txt = path.read_text(errors='replace')
        oom = 'No available memory for the cache blocks' in txt or 'CUDA out of memory' in txt \
            or 'Engine core initialization failed' in txt or 'COTENANT_ABORT' in txt or 'HOST_STALL_ABORT' in txt
        why = '(co-tenant compute: aborted, will retry)' if 'COTENANT_ABORT' in txt else \
              '(host stalled the run: aborted, will retry)' if 'HOST_STALL_ABORT' in txt else \
              '(engine init / memory: will retry)' if oom else ''
        log(f'FAILED gpu={gpu} {tag}/{ds} exit={rc} {why}; inspect {path}')
        if tag == f'{V}_smoke' and not oom:  # an invariant failed: stop everything and report
            STOP.set()
        return False, oom
    why = contaminated(job)
    if why:
        bad = output(job).with_name(f'summary.contaminated.{int(time.time())}.json')
        output(job).rename(bad)
        log(f'CONTAMINATED gpu={gpu} {tag}/{ds}: {why}; summary moved to {bad.name}; will rerun')
        return False, True
    log(f'DONE gpu={gpu} {tag}/{ds}')
    return True, False


TRAIN_MIB = 8500   # GPU memory of the drafter's training, measured 7.8 GB (the hidden states stay in host memory)
TRAIN_WAIT = 1800  # seconds to wait for a card with that much room before training on the CPUs instead


def transferring():
    """A collection directory that is still being copied into the results tree (marker file COPYING)."""
    return bool(list(RES.glob(f'{V}_collect*/*/COPYING')))


def train_ready(todo):
    """The training job, if it is waiting and its collections are complete."""
    with LOCK:
        for j in todo:
            if j[0] == 'train' and STATE[(j[0], j[1])] in ('pending', 'retry') \
                    and (j[3] is None or STATE.get(j[3]) == 'done'):
                return j
    return None


def claim(job):
    with LOCK:
        key = (job[0], job[1])
        if STATE[key] in ('pending', 'retry'):
            STATE[key] = 'running'
            return True
    return False


def run_train(job, gpu, holder=None):
    """Train the drafter on `gpu` (None: the server's CPUs). On a card we hold, the holder frees the trainer's
    share and keeps the rest, so the smoke run and the campaign follow on the same card."""
    key = (job[0], job[1])
    where = 'cpu' if gpu is None else f'gpu{gpu}'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu), PYTHONUNBUFFERED='1')
    env.pop('OMP_NUM_THREADS', None)
    path = LOG / f'train_{job[1]}_{where}_{int(time.time())}.log'
    if holder is not None:
        flag_path(gpu).write_text(f'release {TRAIN_MIB + 1500}')
        time.sleep(6)
    log(f'START {where} held={int(holder is not None)} train/{job[1]} log={path}')
    try:
        with path.open('w') as f:
            rc = subprocess.run([PY, 'scripts/train_drafter.py', '--runs', *job[2], '--out', str(RES / 'drafter')],
                                cwd=CODE, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    finally:
        if holder is not None:  # the holder returns to its idle hold level
            idle_hold(gpu, holder)
    ok = rc == 0 and output(job).exists()
    log(f'DONE {where} train/{job[1]}' if ok else f'FAILED {where} train/{job[1]} exit={rc}; inspect {path}')
    with LOCK:
        TRIES[key] = TRIES.get(key, 0) + 1
        STATE[key] = 'done' if ok else ('retry' if TRIES[key] < 3 else 'failed')
        failed = STATE[key] == 'failed'
    if failed:  # nothing downstream can run without the drafter
        log(f'QUEUE_{V.upper()}_FAILED train')
        STOP.set()
    return ok


def train_worker(todo):
    """The drafter's training needs no engine. A card we already hold runs it (see worker); when we hold none,
    it runs on the pool card with the most free memory, and on the server's CPUs when no card has room."""
    while not STOP.is_set():
        with LOCK:
            open_ = [j for j in todo if j[0] == 'train' and STATE[(j[0], j[1])] not in ('done', 'failed')]
        if not open_:
            return
        job = train_ready(todo)
        if job is None or transferring() or any(_procs_on(g, 'scripts/hold_gpu.py') for g in GPUS):
            STOP.wait(15)
            continue
        if not claim(job):
            continue
        gpu, t0 = None, time.monotonic()
        while gpu is None and time.monotonic() - t0 < TRAIN_WAIT and not STOP.is_set():
            try:  # a card with room for the trainer and a margin (other jobs start in the same seconds)
                best = max(GPUS, key=lambda g: gpu_state(g)[0])
                gpu = best if gpu_state(best)[0] >= TRAIN_MIB + 3000 else None
            except (OSError, ValueError, subprocess.CalledProcessError):
                gpu = None
            if gpu is None:
                STOP.wait(20)
        run_train(job, gpu)  # no card had room for TRAIN_WAIT seconds: the server's CPUs


def running_jobs():
    """(tag, ds) of v15 jobs currently running on this machine, whoever launched them (a previous queue process
    whose jobs survived a restart, or a manual run). They are waited for, never duplicated."""
    import re
    out = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout
    found = set()
    for line in out.splitlines():
        m = re.search(r'scripts/(run_exp|audit_eval)\.py .*--tag (\S+) .*--ds (\S+)', line)
        if m:
            tag, ds = m.group(2), m.group(3)
            if m.group(1) == 'audit_eval':
                tag = 'auditw' if tag == f'{V}_wide' else 'audit'
            if tag.startswith(V) or tag in ('audit', 'auditw'):
                found.add((tag, ds))
    return found


def external_watch(todo):
    """Jobs marked 'external' become 'done' when their output appears, or 'pending' when their process is gone."""
    while not STOP.is_set():
        with LOCK:
            ext = [j for j in todo if STATE[(j[0], j[1])] == 'external']
            open_ = [j for j in todo if STATE[(j[0], j[1])] not in ('done', 'failed')]
        if not open_:
            return
        if not ext:
            STOP.wait(30)
            continue
        live = running_jobs()
        msgs = []
        with LOCK:
            for j in ext:
                key = (j[0], j[1])
                if output(j).exists():
                    why = contaminated(j)
                    if why:
                        output(j).rename(output(j).with_name(f'summary.contaminated.{int(time.time())}.json'))
                        STATE[key] = 'retry'
                        msgs.append(f'EXTERNAL_CONTAMINATED {j[0]}/{j[1]}: {why}; will rerun')
                    else:
                        STATE[key] = 'done'
                        msgs.append(f'EXTERNAL_DONE {j[0]}/{j[1]}')
                elif key not in live:
                    STATE[key] = 'pending'
                    msgs.append(f'EXTERNAL_GONE {j[0]}/{j[1]} (no output: queued again)')
        for m in msgs:
            log(m)
        STOP.wait(30)


def noise_file(path):
    """Co-tenant noise of a summary file: the largest per-method mean co-tenant SM share (%), raised to at least
    NOISY_SM + 1 when another process held memory on the card during any method (GPU time-slicing between contexts
    inflates our iteration time even when the other process is mostly idle; the engine's profiled base iteration
    time is the tell-tale). -1 if unknown."""
    try:
        S = json.loads(Path(path).read_text())
        v = [r.get('gpu_others_sm_mean', -1) for r in S if r.get('gpu_others_sm_mean', -1) >= 0]
        n = max(v) if v else -1
        if any(r.get('gpu_others_mib', 0) > 500 for r in S):
            n = max(n, NOISY_SM + 1)
        return n
    except Exception:
        return -1


def noise(job):
    return noise_file(output(job))


REQUIRE_EMPTY = set()   # (tag, ds) of polish-phase reruns: only on an empty (held) card


def pick(todo, free_mib, quiet, empty=False):
    """The first runnable job for a card with `free_mib` free whose co-tenants are (quiet) calm."""
    now = time.monotonic()
    live = None
    with LOCK:
        for j in todo:
            key = (j[0], j[1])
            if j[0] == 'train':  # no engine: see train_worker
                continue
            if STATE[key] in ('pending', 'retry') and (j[3] is None or STATE.get(j[3]) == 'done') \
                    and free_mib >= min_free_for(j) and (quiet or j[0] in TIMING_FREE) \
                    and now >= NOT_BEFORE.get(key, 0) and (empty or key not in REQUIRE_EMPTY):
                live = running_jobs() if live is None else live
                if key in live:  # started by someone else meanwhile: wait for it instead
                    STATE[key] = 'external'
                    continue
                STATE[key] = 'running'
                return j
    return None


def finish(job, ok, oom):
    with LOCK:
        key = (job[0], job[1])
        if ok:
            STATE[key] = 'done'
        else:  # keep the job's priority; retry after a backoff (memory failures up to 6 times, others twice)
            TRIES[key] = TRIES.get(key, 0) + (1 if oom else 3)
            STATE[key] = 'retry' if TRIES[key] < 6 else 'failed'
            NOT_BEFORE[key] = time.monotonic() + 120


def worker(gpu, todo):
    ready_since = quiet_since = None
    low_mem = waiting = False
    holder = adopt_holder(gpu)
    try:
        while not STOP.is_set():
            with LOCK:
                pending = [j for j in todo if STATE[(j[0], j[1])] in ('pending', 'retry', 'external')]
            if not pending:
                return
            try:
                free_mib, total, util = gpu_state(gpu)
                usable = free_mib >= NEED['free'] + HEADROOM
            except (OSError, ValueError, subprocess.CalledProcessError) as e:
                log(f'GPU_QUERY_ERROR gpu={gpu} {e}')
                usable = False
            if our_job_on(gpu):  # one of our jobs (e.g. adopted from a previous queue) is on this card: wait
                STOP.wait(3)
                continue
            if holder is not None:
                # no job of ours on the card: the holder keeps it (also after the job of a previous queue process
                # ended and left its hand-over flag, and when a co-tenant has left meanwhile)
                idle_hold(gpu, holder)
            gib, need = host_free_gib(), host_need_gib()
            if gib < need:  # the machine is short of memory: a new job of ours would push it into swap
                if not low_mem:
                    log(f'HOST_MEMORY gpu={gpu}: {gib:.0f} GiB available on the host (< {need}); waiting')
                low_mem = True
                ready_since = quiet_since = None
                STOP.wait(30)
                continue
            low_mem = False
            if holder is not None:  # we hold this card: run the next job at once, no new idle window needed
                avail = free_mib + proc_mem(gpu, holder.pid)  # what the holder can hand over, plus what is free
                tj = train_ready(todo)
                if tj is not None:  # the drafter's training runs on the card we hold; what follows needs it
                    if transferring():
                        STOP.wait(10)
                        continue
                    if claim(tj):
                        run_train(tj, gpu, holder)
                        continue
                tok = host_admit()
                if tok is None:  # another worker of ours took the host memory in the same seconds
                    STOP.wait(10)
                    continue
                job = pick(todo, avail, quiet=util < CALM_UTIL, empty=total - avail <= EMPTY_MIB)
                if job is None:
                    host_cancel(tok)
                    # nothing of ours fits next to the card's co-tenants right now, or they compute: the card is
                    # kept until they leave (a released card is filled by other schedulers within seconds and
                    # never comes back empty)
                    if not waiting:
                        log(f'HOLD_WAIT gpu={gpu}: {avail} MiB for us, co-tenant utilization {util}%; '
                            f'keeping the card for {len(pending)} pending jobs')
                    waiting = True
                    STOP.wait(15)
                    continue
                waiting = False
                ok, oom = execute(gpu, job, holder)
                host_cancel(tok)  # the job is over: its host memory is free again
                finish(job, ok, oom)
                continue
            if not usable:
                ready_since = quiet_since = None
                STOP.wait(10)
                continue
            ready_since = ready_since or time.monotonic()
            quiet_since = (quiet_since or time.monotonic()) if util < CALM_UTIL else None
            if time.monotonic() - ready_since < IDLE_SECONDS:
                STOP.wait(5)
                continue
            # timing-sensitive jobs need a card whose co-tenants stayed calm (utilization < CALM_UTIL) throughout
            # the window (their activity is recorded in every summary); timing-free jobs need only the memory
            quiet = quiet_since is not None and time.monotonic() - quiet_since >= IDLE_SECONDS
            empty = total - free_mib <= EMPTY_MIB
            tok = host_admit()
            if tok is None:
                STOP.wait(10)
                continue
            job = pick(todo, free_mib, quiet, empty=empty)
            if job is None:
                host_cancel(tok)
                STOP.wait(5)
                continue
            # hold the card's spare memory so that no other project's scheduler places work next to our run (timed
            # jobs) or takes the memory the engine is about to allocate (every job: an engine start that loses its
            # memory half-way costs the whole start): all of it on an empty card, all but a growth buffer next to
            # a co-tenant; the hold is kept across our consecutive jobs on this card
            others_before = total - free_mib
            holder = start_holder(gpu, HOLD_LEAVE if empty else HOLD_LEAVE + COTENANT_GROWTH)
            t_h = time.monotonic()
            while time.monotonic() - t_h < 90 and proc_mem(gpu, holder.pid) < 1000 and holder.poll() is None:
                time.sleep(2)  # the holder's allocation precedes the engine start
            time.sleep(4)
            free2, total2, _ = gpu_state(gpu)
            held = proc_mem(gpu, holder.pid)
            others_now = total2 - free2 - held
            if others_now - others_before > 1500 or held + free2 < min_free_for(job):
                # someone else allocated in the same seconds: give the card up
                log(f'HOLD_LOST gpu={gpu}: others grew {others_before}->{others_now} MiB while holding '
                    f'(held {held} MiB); releasing and re-queueing {job[0]}/{job[1]}')
                stop_holder(gpu, holder)
                holder = None
                host_cancel(tok)
                with LOCK:
                    STATE[(job[0], job[1])] = 'pending'
                ready_since = quiet_since = None
                STOP.wait(10)
                continue
            ok, oom = execute(gpu, job, holder)
            host_cancel(tok)
            finish(job, ok, oom)
            ready_since = quiet_since = None
    finally:
        stop_holder(gpu, holder)


TRIES = {}
NOT_BEFORE = {}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'list':
        for j in jobs():
            print(j[0], j[1], ' '.join(j[2]))
        return 0
    import fcntl
    lockfile = (LOG / f'queue_{V}.lock').open('w')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    log(f'QUEUE_START gpus={GPUS} need_mib={NEED} headroom={HEADROOM} idle_seconds={IDLE_SECONDS}')
    smoke_methods = f'{U},{U}-nopeer,{U}-sbd,{U}-noid' if U == 'u17' else f'{U},{U}-sbd,{U}-noid,lacache'
    smoke = (f'{V}_smoke', 'hotpotqa', ['--methods', smoke_methods,
        '--n', '20', '--rate', '1.2', '--mu', '0.1', '--seed', '0', '--qsplit', 'test', '--note', 'dfp'], None)
    upfront_smoke = V < 'v17'  # v17: the smoke run is a queued job gated on the trained drafter
    if upfront_smoke and os.environ.get('UNISON_SKIP_SMOKE') != '1' and (not output(smoke).exists() or os.environ.get('V15_FORCE_SMOKE') == '1'):
        if output(smoke).exists():
            output(smoke).unlink()
        todo = [smoke]
        STATE[(smoke[0], smoke[1])] = 'pending'
        with concurrent.futures.ThreadPoolExecutor(len(GPUS)) as pool:
            list(pool.map(lambda g: worker(g, todo), GPUS))
        if STOP.is_set() or STATE[(smoke[0], smoke[1])] != 'done':
            log(f'QUEUE_{V.upper()}_FAILED smoke')
            return 1
    for ds in ['hotpotqa', 'musique']:
        if not (ROOT / 'data' / ds / 'eval/versions_edit.jsonl').exists():
            subprocess.run([PY, 'scripts/gen_edits.py', ds], cwd=CODE, check=True)
    todo = jobs()
    live = running_jobs()
    for j in todo:
        key = (j[0], j[1])
        if output(j).exists():
            why = contaminated(j)
            if why:
                output(j).rename(output(j).with_name(f'summary.contaminated.{int(time.time())}.json'))
                log(f'CONTAMINATED (existing) {j[0]}/{j[1]}: {why}; will rerun')
                STATE[key] = 'pending'
            else:
                STATE[key] = 'done'
        else:
            STATE[key] = 'external' if key in live else 'pending'
        if STATE[key] == 'external':
            log(f'EXTERNAL {j[0]}/{j[1]} is already running: waiting for it')
    only = os.environ.get('UNISON_ONLY')  # comma-separated tag fragments: run only these jobs (gates still apply)
    if only:
        todo = [j for j in todo if any(s in j[0] for s in only.split(','))]
        for k in [k for k in STATE if k not in {(j[0], j[1]) for j in todo} and STATE[k] != 'done']:
            STATE[k] = 'skipped'
        log(f'ONLY {only}: ' + ' '.join(f'{j[0]}/{j[1]}({STATE[(j[0], j[1])]})' for j in todo))
    watcher = threading.Thread(target=external_watch, args=(todo,), daemon=True)
    watcher.start()
    threading.Thread(target=train_worker, args=(todo,), daemon=True).start()
    with concurrent.futures.ThreadPoolExecutor(len(GPUS)) as pool:
        list(pool.map(lambda g: worker(g, todo), GPUS))
    failed = [k for k, v in STATE.items() if v not in ('done', 'skipped')]
    log(f'QUEUE_{V.upper()}_FAILED {failed}' if failed else f'QUEUE_{V.upper()}_DONE')
    # polish phase: timed results obtained next to a noisy co-tenant are repeated on empty cards as they become
    # available; the cleaner of the two runs is kept. Ctrl-C / kill ends this phase at any time.
    noisy = [j for j in todo if tier(j) != 'free' and j[0] not in ('audit', 'auditw')
             and STATE[(j[0], j[1])] == 'done' and noise(j) > NOISY_SM]
    if noisy and os.environ.get('V15_POLISH', '1') == '1':
        log(f'POLISH_START {len(noisy)} noisy results: ' + ' '.join(f'{j[0]}/{j[1]}({noise(j):.0f}%)' for j in noisy))
        for j in noisy:
            REQUIRE_EMPTY.add((j[0], j[1]))
            STATE[(j[0], j[1])] = 'pending'
            TRIES.pop((j[0], j[1]), None)
        with concurrent.futures.ThreadPoolExecutor(len(GPUS)) as pool:
            list(pool.map(lambda g: worker(g, noisy), GPUS))
        log(f'QUEUE_{V.upper()}_POLISH_DONE')
    return int(bool(failed))


if __name__ == '__main__':
    sys.exit(main())
