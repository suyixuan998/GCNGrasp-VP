import subprocess

import optuna

from gcngrasp import Path
from gcngrasp.api.utils.bayes_opt import BayesOpt

CFG_SRC = Path("config/system/params.yaml")
CFG_DST = Path("runs/tmp.yaml")


def objective(trial: optuna.Trial):
    cfg = CFG_SRC.yaml()
    for name in ["occ", "elev"]:
        cfg["planner"]["loss_weight"][name] = trial.suggest_float(name, 0, 2, step=.1)
    if trial.should_prune(): raise optuna.TrialPruned()
    CFG_DST.yaml(cfg)

    result = subprocess.run(
        ["python", "main_nbv.py", "--config", CFG_DST],  # , "--n-seed", "1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if result.returncode: return

    folder = sorted(Path("runs/nbv_logs").iterdir())[-1]
    ret = sum([f.yaml()["avg"]["AP"][1:3] for f in folder.rglob("results.yaml")], [])
    return sum(ret) / len(ret)


if __name__ == '__main__':
    opt = BayesOpt(Path("runs/bayes.bin"), next_param={'occ': 0.6, 'elev': 0.2})
    print(opt.dataframe)
    print(opt.best)
    opt(objective, 100)
    opt.dataframe.to_csv("runs/bayes.csv")
