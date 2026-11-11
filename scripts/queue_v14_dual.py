"""Run the v14 manuscript suite on the authorized GPU pool, one job per GPU.

Idle GPUs take the next ready job. An occupied GPU waits without reserving work.
Each comparison stays on one GPU; smoke must pass before manuscript jobs start.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path('/data/yaokang/evishare')
CODE, LOG, RES = ROOT / 'code', ROOT / 'logs', ROOT / 'results'
PY = str(ROOT / '.venv/bin/python')
LOCK = threading.Lock()
STOP = threading.Event()
STATE = {}
GPUS = tuple(g for g in (int(x) for x in os.environ.get('UNISON_GPUS', '0,1,4,3').split(',')) if g in (0, 1, 2, 3, 4))
IDLE_SECONDS = 10


def log(message):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + message
    with LOCK:
        print(line, flush=True)
        with (LOG / 'queue_v14_dual.log').open('a') as f:
            f.write(line + '\n')


def idle(gpu):
    out = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    free, util = map(int, out.strip().split(','))
    return free >= 15000 and util < 10


def jobs():
    base = ['--n', '200', '--seed', '0', '--qsplit', 'test', '--note', 'dfp']
    main = 'noshare,pld,suffix,gptcache,vcacheoff,lacache,u14-sbd,u14,noshare@cot'
    abl = 'u14-lazy,u14-sbd,u14-coop,u14-keyed,u14,u14-nolive,u14-noprio'
    load = 'suffix,gptcache,u14-sbd,u14'
    out = []
    for tag, methods in [('v14_main', main), ('v14_abl', abl)]:
        for ds in ['hotpotqa', 'musique']:
            args = ['--methods', methods, '--rate', '1.2' if ds == 'hotpotqa' else '0.6', '--mu', '0.1']
            if tag == 'v14_main':
                args += ['--replay2', '1']
            out.append((tag, ds, args + base, None))
    for rate in ['0.6', '1.8', '2.4', '3.0']:
        out.append(('v14_load_r' + rate, 'hotpotqa', ['--methods', load, '--rate', rate, '--mu', '0.1'] + base, None))
    out.append(('v14_edit', 'hotpotqa', ['--methods', 'gptcache,vcacheoff,lacache,u14-sbd,u14',
        '--rate', '1.2', '--mu', '1.0', '--versions', 'versions_edit.jsonl'] + base, None))
    for ds in ['hotpotqa', 'musique']:
        out.append(('audit', ds, [], ('v14_abl', ds)))
    return out


def output(job):
    tag, ds, _, _ = job
    return RES / ('v14_abl' if tag == 'audit' else tag) / ds / (
        'audit_u14_vs_u14-sbd.json' if tag == 'audit' else 'summary.json')


def execute(gpu, job):
    tag, ds, args, _ = job
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), VLLM_USE_V1='1',
        UNISON_PRIO='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    if tag == 'audit':
        cmd = [PY, 'scripts/audit_eval.py', '--tag', 'v14_abl', '--ds', ds,
               '--a', 'u14', '--b', 'u14-sbd', '--n', '400', '--gpu_util', '0.85']
    else:
        cmd = [PY, 'scripts/run_exp.py', '--tag', tag, '--ds', ds, *args, '--gpu_util', '0.85']
    path = LOG / f'{tag}_{ds}_gpu{gpu}_{int(time.time())}.log'
    log(f'START gpu={gpu} {tag}/{ds} log={path}')
    with path.open('w') as f:
        rc = subprocess.run(cmd, cwd=CODE, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc == 0 and tag == 'v14_smoke':
        check = subprocess.run([PY, 'scripts/check_smoke.py', str(output(job)), str(path)],
                               cwd=CODE, capture_output=True, text=True)
        log(check.stdout.strip() + check.stderr.strip())
        rc = check.returncode
    if rc != 0 or not output(job).exists():
        log(f'FAILED gpu={gpu} {tag}/{ds} exit={rc}; inspect {path}')
        STOP.set()
        return False
    log(f'DONE gpu={gpu} {tag}/{ds}')
    return True


def worker(gpu, todo):
    ready_since = None
    while not STOP.is_set():
        with LOCK:
            pending = [j for j in todo if STATE[(j[0], j[1])] == 'pending']
        if not pending:
            return
        try:
            free = idle(gpu)
        except (OSError, ValueError, subprocess.CalledProcessError) as e:
            log(f'GPU_QUERY_ERROR gpu={gpu} {e}')
            free = False
        if not free:
            ready_since = None
            STOP.wait(10)
            continue
        ready_since = ready_since or time.monotonic()
        if time.monotonic() - ready_since < IDLE_SECONDS:
            STOP.wait(5)
            continue
        chosen = None
        with LOCK:
            for j in todo:
                key = (j[0], j[1])
                if STATE[key] == 'pending' and (j[3] is None or STATE.get(j[3]) == 'done'):
                    STATE[key] = 'running'
                    chosen = j
                    break
        if chosen is None:
            STOP.wait(5)
            continue
        ok = execute(gpu, chosen)
        with LOCK:
            STATE[(chosen[0], chosen[1])] = 'done' if ok else 'failed'
        ready_since = None


def main():
    import fcntl
    lockfile = (LOG / 'queue_v14_dual.lock').open('w')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    log(f'QUEUE_START gpus={GPUS} min_free_mib=15000 idle_seconds={IDLE_SECONDS}')
    smoke = ('v14_smoke', 'hotpotqa', ['--methods', 'u14,u14-sbd,u14-noprio,lacache',
        '--n', '20', '--rate', '1.2', '--mu', '0.1', '--seed', '0', '--qsplit', 'test', '--note', 'dfp'], None)
    todo = [smoke]
    STATE[(smoke[0], smoke[1])] = 'pending'
    if os.environ.get('V14_RESUME_AUDIT') != '1':
        with concurrent.futures.ThreadPoolExecutor(len(GPUS)) as pool:
            list(pool.map(lambda g: worker(g, todo), GPUS))
    elif not all(output(j).is_file() for j in jobs() if j[0] != 'audit'):
        raise RuntimeError('Audit-only resume requires completed manuscript benchmarks')
    if STOP.is_set():
        return 1
    for ds in ['hotpotqa', 'musique']:
        if not (ROOT / 'data' / ds / 'eval/versions_edit.jsonl').exists():
            subprocess.run([PY, 'scripts/gen_edits.py', ds], cwd=CODE, check=True)
    todo = jobs()
    for j in todo:
        STATE[(j[0], j[1])] = 'done' if output(j).exists() else 'pending'
    with concurrent.futures.ThreadPoolExecutor(len(GPUS)) as pool:
        list(pool.map(lambda g: worker(g, todo), GPUS))
    log('QUEUE_V14_DUAL_FAILED' if STOP.is_set() else 'QUEUE_V14_DUAL_DONE')
    return int(STOP.is_set())


if __name__ == '__main__':
    sys.exit(main())
