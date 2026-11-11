"""Micro-batching of small-model inference off the asyncio event loop.

Concurrent requests that arrive while a batch is running are coalesced into the next batch,
which runs in a dedicated worker thread so token streaming from the engine is never blocked.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor


class AsyncBatcher:
    def __init__(self, fn, max_items=256, pool=None):
        self.fn = fn                    # sync: list[item] -> list[result]
        self.max_items = max_items
        self.queue = []
        self.busy = False
        self.pool = pool or ThreadPoolExecutor(max_workers=1)

    async def __call__(self, item):
        fut = asyncio.get_running_loop().create_future()
        self.queue.append((item, fut))
        if not self.busy:
            self.busy = True
            asyncio.create_task(self._drain())
        return await fut

    async def _drain(self):
        loop = asyncio.get_running_loop()
        try:
            while self.queue:
                batch, self.queue = self.queue[:self.max_items], self.queue[self.max_items:]
                try:
                    res = await loop.run_in_executor(self.pool, self.fn, [b[0] for b in batch])
                    for (_, f), r in zip(batch, res):
                        f.set_result(r)
                except Exception as e:  # propagate to every waiter
                    for _, f in batch:
                        if not f.done():
                            f.set_exception(e)
        finally:
            self.busy = False
