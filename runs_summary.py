from __future__ import annotations

from gcngrasp import Path
from gcngrasp.utils.tb_tool import *

RUNS = Path("runs")


def nbv_summary():
    folder = RUNS / "nbv_logs"
    if not folder.exists(): return

    ret = {}
    for log in folder.iterdir():
        log_name = log.name.split("_", maxsplit=2)[-1]

        for scene in log.iterdir():
            f = Path(scene / "results.yaml")
            if not f.exists(): continue
            metric = f.yaml()["avg"]["AP"]
            ret[f"{log_name}-{scene.name}"] = metric

    return ret


def training_summary(name: str, k: int = 4) -> pd.DataFrame:
    results = {}
    folder = RUNS / "train" / name

    for log in folder.iterdir():
        reader = tbparse.SummaryReader(log)
        hparams = dict(reader.hparams.values)
        x = filter_by_tag(reduce(reader.scalars), r"^test")

        # fold
        split, fold = hparams["fold"]
        results.setdefault(split, [None] * k)[int(fold)] = x

    # merge
    ret = pd.DataFrame()
    for s, tests in results.items():
        try:
            ret[s] = sum(tests) / k
        except:
            miss = [f"{s}{i}" for i, x in enumerate(tests) if x is None]
            print(f"{name}: missing {miss}")

    if not len(ret): return None
    ret.index.name = name
    return ret


if __name__ == '__main__':
    for n in ["v1-org", "v2", "v2-aff"]:
        try:
            ret = training_summary(n)
            if ret is None: continue
            print(ret)
        except Exception as e:
            print(e)

    for k, v in nbv_summary().items():
        print(f"{k}: {v}")
