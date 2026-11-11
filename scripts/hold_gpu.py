"""Hold GPU memory until killed: python hold_gpu.py <MiB to leave free> [flag file].

Started by the queue the moment a card is chosen, so that memory-driven job schedulers of other projects on the
machine do not place work on the card. It keeps the card's free memory at <leave> MiB and absorbs any surplus
within a second (e.g. when our engine exits between two consecutive jobs); nothing is released until the queue
terminates the process, except through the flag file. The memory is held in chunks of 256 MiB so that any amount
can be given back exactly.

Flag file commands (the last one written applies):
  resume <MiB>    hold: absorb whatever is free beyond <MiB>; never release (between jobs, written by the queue)
  release <MiB>   free chunks until <MiB> are free on the card, then stand still: the engine process of a job
                  takes that memory in one allocation (its allocator pool) within the next tenth of a second
  feed <MiB>      regulate: keep the card's free memory at <MiB> from both sides. Surplus is absorbed; when the
                  free memory falls 400 MiB below <MiB>, chunks are released until it is back. A running job gets
                  what its engine allocates outside the pool (CUDA graphs, library workspaces, the encoder of the
                  driver process) from this reserve, in small steps, and the card never shows more than <MiB> free
Without a flag file the <leave> of the command line applies as in "resume"."""
import os
import sys
import time

import torch

MiB = 1 << 20
CHUNK = 256 * MiB
leave_default = int(sys.argv[1]) * MiB
flag = sys.argv[2] if len(sys.argv) > 2 else None
held = []
mode, level = "hold", leave_default
last_flag = None
t_log = t_act = 0.0


def absorb(n):
    """Take `n` bytes of the card's free memory in chunks; False if another allocation raced with ours."""
    while n >= 64 * MiB:
        k = min(CHUNK, n)
        try:
            held.append(torch.empty(k, dtype=torch.uint8, device="cuda"))
        except RuntimeError:
            return False
        n -= k
    return True


def give(target):
    """Release chunks until at least `target` bytes are free on the card (or nothing is left to release): the
    chunks the gap asks for go back in one call, so the memory appears at once and not as a ramp."""
    n = 0
    while held:
        gap = target - torch.cuda.mem_get_info()[0]
        if gap <= 0:
            break
        k = min(len(held), -(-gap // CHUNK))
        del held[-k:]
        torch.cuda.empty_cache()
        n += k
    return n


while True:
    cmd = None
    if flag and os.path.exists(flag):
        try:
            cmd = open(flag).read().strip()
        except OSError:
            cmd = last_flag
    if cmd != last_flag:
        last_flag = cmd
        parts = (cmd or "").split()
        if len(parts) == 2 and parts[0] in ("release", "feed", "resume") and parts[1].isdigit():
            mode, level = {"resume": "hold"}.get(parts[0], parts[0]), int(parts[1]) * MiB
            if mode == "release":
                give(level)
                mode = "still"
                print(f"release: {torch.cuda.mem_get_info()[0] >> 20} MiB free for the engine, "
                      f"{sum(t.numel() for t in held) >> 20} MiB kept", flush=True)
            else:
                print(f"{parts[0]}: keeping {level >> 20} MiB free"
                      + (" (from both sides)" if mode == "feed" else ""), flush=True)
        elif cmd is None:
            mode, level = "hold", leave_default
        t_act = time.time()
    if mode != "still":
        free, total = torch.cuda.mem_get_info()
        if free - level >= 64 * MiB:
            absorb(free - level)
            t_act = time.time()
        elif mode == "feed" and level - free > 400 * MiB and held:
            give(level)
            t_act = time.time()
        if time.time() - t_log > 60:
            t_log = time.time()
            print(f"{mode}: holding {sum(t.numel() for t in held) >> 20} MiB in {len(held)} chunks, "
                  f"{torch.cuda.mem_get_info()[0] >> 20} MiB free of {total >> 20}", flush=True)
    # close to an event (a command, memory taken or given) the card is watched closely; otherwise ten times a
    # second (one query of the device's free memory; the engine of a job that ends frees its memory at once)
    time.sleep(0.02 if time.time() - t_act < 15 else 0.1)
