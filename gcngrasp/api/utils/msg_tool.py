from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable


class LatestMsg:

    def __init__(self):
        self.data = None
        self.lock = threading.RLock()
        self.updated: set[threading.Event] = set()

    def put(self, *args):
        with self.lock:
            self.data = args if len(args) > 1 else args[0]
            for event in self.updated: event.set()

    def get(self):
        return self.data

    def create_getter(self, timeout=None):
        return self.Getter(self, timeout)

    class Getter:

        def __init__(self,
                     lmsg: "LatestMsg",
                     timeout: float = None):
            self.lmsg = lmsg
            self.timeout = timeout
            self.event = threading.Event()
            lmsg.updated.add(self.event)

        def __call__(self):
            if self.event.wait(self.timeout):
                self.event.clear()
                return self.lmsg.get()

        def __del__(self):
            self.lmsg.updated.remove(self.event)


@dataclass
class ApproxTimeSynchronizer:
    callback: Callable
    n_inputs: int
    queue_size: int
    slop: float

    def __post_init__(self):
        self.queues = [deque(maxlen=self.queue_size) for _ in range(self.n_inputs)]
        self.locks = [threading.Lock() for _ in range(self.n_inputs)]
        self.lock_sync = threading.RLock()

    def __len__(self):
        return self.n_inputs

    def append(self, i, msg, t=None):
        if t is None: t = time.time()
        with self.locks[i]:
            if self.queues[i] and t == self.queues[i][-1][0]: return

        self.queues[i].append((t, msg))
        self._try_sync()

    def describe(self) -> list[dict]:
        infos = []
        for q, lock in zip(self.queues, self.locks):
            with lock:
                tmp = {"size": len(q)}
                if len(q): tmp["t-range"] = q[0][0], q[-1][0]
            infos.append(tmp)
        return infos

    def _try_sync(self):
        if self.lock_sync.acquire(blocking=False):
            while True:
                try:
                    heads = [q[0] for q in self.queues]
                except IndexError:
                    break
                t1 = max(heads)[0]
                t0 = t1 - self.slop

                # discard messages that are too early
                changed = False
                for q, lock in zip(self.queues, self.locks):
                    with lock:
                        while q and q[0][0] < t0:
                            changed = bool(q.popleft())
                        while len(q) >= 2 and abs(t1 - q[0][0]) > abs(t1 - q[1][0]):
                            changed = bool(q.popleft())

                if not changed:
                    if min(heads)[0] < t0: break

                    # success
                    args = []
                    for q, lock in zip(self.queues, self.locks):
                        with lock:
                            args.append(q.popleft()[1])
                    self.callback(*args)

            self.lock_sync.release()


if __name__ == '__main__':
    import random

    ats = ApproxTimeSynchronizer(callback=print, n_inputs=3, queue_size=10, slop=2.)

    for i in range(100):
        for j in range(len(ats)):
            if random.random() < 0.6:
                ats.append(j, i, t=i)
        print(ats.describe())
