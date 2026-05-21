from __future__ import annotations

import asyncio
import concurrent.futures
import pickle
import time
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable

import requests

try:
    import fastapi
except ImportError:
    pass

try:
    import zmq

    try:
        from .zmq_socket import ZmqSocket
    except:
        from zmq_socket import ZmqSocket
except ImportError:
    pass

PROCESS_POOL = concurrent.futures.ProcessPoolExecutor()
THREAD_POOL = concurrent.futures.ThreadPoolExecutor()


@dataclass
class BaseInvoker:

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return partial(self, item)

    def try_call(self, func, *args, **kwargs):
        try:
            return self(func, *args, **kwargs)
        except Exception as e:
            return e

    async def invoke_async(self, *args, **kwargs):
        try:
            return await asyncio.to_thread(self.try_call, *args, **kwargs)
        except AttributeError:
            return self.try_call(*args, **kwargs)

    def invoke_process(self, *args, **kwargs) -> concurrent.futures.Future:
        return PROCESS_POOL.submit(self, *args, **kwargs)

    def invoke_thread(self, *args, **kwargs) -> concurrent.futures.Future:
        return THREAD_POOL.submit(self, *args, **kwargs)


@dataclass
class LocalInvoker(BaseInvoker):
    functions: dict

    def __call__(self, func, *args, **kwargs):
        return self.functions[func](*args, **kwargs)

    def check(self,
              require: list[str] = [],
              optional: list[str] = []):
        funcs = set(self.functions)
        require_miss = set(require) - funcs
        optional_miss = set(optional) - funcs

        if require_miss:
            raise ValueError(f"Required functions missing: {require_miss}")
        if optional_miss:
            warnings.warn(f"Optional functions missing: {optional_miss}")

    def func_docs(self,
                  fmt: str = "md") -> str:
        if fmt == "md":
            return "\n\n".join(f"## {k}\n\n```\n{v.__doc__ or 'unknown'}\n```" for k, v in self.functions.items())
        else:
            raise ValueError(f"Unknown format: {fmt}")


@dataclass
class BaseClient(BaseInvoker):

    def __call__(self, func, *args, **kwargs):
        data = {"t": time.time(), "func": func, "args": args, "kwargs": kwargs}
        ret = self.get_response(data)
        if isinstance(ret, Exception):
            raise ret
        return ret

    def get_response(self, data: dict):
        raise NotImplementedError


@dataclass
class BaseServer(LocalInvoker):
    loginfo: Callable

    async def get_response(self, data: dict):
        info = {"t_send": round(time.time() - data["t"], 3)}
        func = data["func"]

        t0 = time.time()
        ret = await self.invoke_async(func, *data["args"], **data["kwargs"])
        info["t_invoke"] = round(time.time() - t0, 3)
        self.loginfo(f"{func}: {info}")
        return ret


# FastAPI
@dataclass
class FastapiClient(BaseClient):
    url: str
    path_invoke: str = "invoke"

    def __post_init__(self):
        assert requests.get(f"{self.url}/docs").status_code == 200
        print(f"See {self.url}/docs for API documentation.")

    def get_response(self, data: dict):
        response = requests.post(
            f"{self.url}/{self.path_invoke}", data=pickle.dumps(data))
        assert response.status_code == 200, f"{response.status_code}, {response.text}"
        return pickle.loads(response.content)


@dataclass
class FastapiServer(BaseServer):

    async def get_response(self, data: "fastapi.Request"):
        # handle serialization errors
        try:
            data: dict = pickle.loads(await data.body())
        except:
            raise fastapi.HTTPException(status_code=400, detail="Bad input data.")

        # handle runtime errors
        try:
            result = await super().get_response(data)
        except ValueError as e:
            raise fastapi.HTTPException(status_code=400, detail=f"{e}")
        except Exception as e:
            raise fastapi.HTTPException(status_code=500, detail=f"{e}")

        return fastapi.Response(content=pickle.dumps(result), media_type="application/octet-stream")


# ZeroMQ
@dataclass
class ZmqSocketCLient(BaseClient):
    sock: "ZmqSocket"

    def __post_init__(self):
        assert self.sock.type == zmq.REQ

    def get_response(self, data: dict):
        return self.sock.send(data, response=True)


@dataclass
class ZmqSocketServer(BaseServer):
    sock: "ZmqSocket"

    def __post_init__(self):
        assert self.sock.type == zmq.REP

    def main(self,
             hz: int = 500,
             thread: bool = False):
        self.sock.recv_auto(
            hz, callback=lambda data: asyncio.run(self.get_response(data)), response=True, thread=thread
        )


if __name__ == '__main__':
    import numpy as np

    # invoke = FastapiClient("http://127.0.0.1:8000")
    invoke = ZmqSocketCLient(sock=ZmqSocket(zmq.REQ, addr="0.0.0.0:5555", queue_size=10))

    results = [
        invoke("add", 1, np.array([1, 3, 2])),
        invoke("sleep", 2),
        invoke("sleep", 2),
    ]
    print(results)
    print([r.result() for r in results])
