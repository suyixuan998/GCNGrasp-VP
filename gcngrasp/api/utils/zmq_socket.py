import atexit
import itertools
import pickle
import time
from dataclasses import dataclass, field
from typing import Callable

import zmq

CONFIG = {"io_threads": 1}


def set_io_threads(n: int):
    if CONFIG.get("ctx"):
        assert CONFIG["io_threads"] == n, "Cannot change io_threads"
    CONFIG["io_threads"] = n


def get_context() -> zmq.Context:
    if not CONFIG.get("ctx"):
        CONFIG["ctx"] = zmq.Context(CONFIG["io_threads"])
        atexit.register(CONFIG["ctx"].destroy)
    return CONFIG["ctx"]


@dataclass
class Rate:
    hz: int

    def __post_init__(self):
        self.interval = 1 / self.hz

    def __iter__(self):
        for i in itertools.count():
            t1 = time.time() + self.interval
            yield i
            time.sleep(max(0, t1 - time.time()))


@dataclass
class ZmqSocket:
    type: zmq.SocketType
    addr: str
    queue_size: int

    func_serial: Callable = pickle.dumps
    func_deserial: Callable = pickle.loads

    sock: zmq.Socket = field(init=False)

    def __post_init__(self):
        assert self.is_client or self.is_server, "Invalid socket type"

        # create socket
        self.sock = get_context().socket(self.type)
        self.sock.set_hwm(self.queue_size)
        self.sock.setsockopt(zmq.LINGER, 0)

        # for sub
        if self.type == zmq.SUB:
            self.sock.setsockopt(zmq.SUBSCRIBE, b'')

        # resolve address
        if "://" not in self.addr:
            protocol = "tcp" if ":" in self.addr else "ipc"
            self.addr = f"{protocol}://{self.addr}"

        # bind or connect
        func_conn = self.sock.bind if self.is_server else self.sock.connect
        func_conn(self.addr)

    @property
    def is_client(self) -> bool:
        return self.type in (zmq.SUB, zmq.REQ, zmq.PULL)

    @property
    def is_server(self) -> bool:
        return self.type in (zmq.PUB, zmq.REP, zmq.PUSH)

    def send(self,
             x,
             response: bool = False):
        assert x is not None, "x cannot be None"
        self.sock.send(self.func_serial(x))
        if response: return self.recv(blocking=True)

    def recv(self, blocking: bool = True):
        try:
            flags = 0 if blocking else zmq.NOBLOCK
            return self.func_deserial(self.sock.recv(flags=flags))
        except zmq.Again as e:
            pass

    def recv_auto(self,
                  hz: int,
                  callback: Callable,
                  response: bool,
                  thread: bool):
        if not thread:
            for _ in Rate(hz):
                data = self.recv(blocking=False)
                if data is not None:
                    ret = callback(data)
                    if response: self.send(ret)
        else:
            import threading
            threading.Thread(
                target=self.recv_auto, args=(hz, callback),
                kwargs=dict(response=response, thread=False), daemon=True
            ).start()


if __name__ == '__main__':
    addr = "0.0.0.0:5555"

    server = ZmqSocket(zmq.REP, addr, 1)
    print(server.type)
    client = ZmqSocket(zmq.REQ, addr, 1)

    server.recv_auto(10, (2).__add__, response=True, thread=True)
    time.sleep(1)

    print("sent")
    print(client.send(1, response=True))
    time.sleep(1)
