import pickle
from pathlib import Path
from typing import Callable

import optuna
import pandas as pd


class NoRepeatPruner(optuna.pruners.BasePruner):

    def __init__(self):
        self.seen_params = set()

    def prune(self,
              study: optuna.study.Study,
              trial: optuna.trial.FrozenTrial) -> bool:
        params = tuple(sorted(trial.params.items()))
        if params in self.seen_params: return True
        self.seen_params.add(params)
        return False


class BayesOpt:
    best = property(lambda self: self.study.best_trial)

    def __init__(self,
                 file: Path,
                 direction: str = "maximize",
                 next_param: dict = None):
        """
        A simple wrapper for Optuna to perform Bayesian optimization.
        :param file: file to save the trials
        :param direction: optimization direction, either "minimize" or "maximize"
        :param next_param: next parameters to enqueue for the optimization
        """
        self.study = optuna.create_study(direction=direction, pruner=NoRepeatPruner())
        self.file = file
        if file.is_file():
            self.study.add_trials(pickle.loads(file.read_bytes()))
        if next_param: self.enqueue(next_param)

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.study.trials_dataframe(
            attrs=("datetime_start", "duration", "params", "value")
        )

    def enqueue(self, params):
        self.study.enqueue_trial(params)
        self.save()

    def save(self):
        self.file.write_bytes(pickle.dumps(self.study.trials))

    def __len__(self):
        return len(tuple(filter(lambda t: t.value, self.study.trials)))

    def __call__(self,
                 objective: Callable,
                 n_trials: int):
        while len(self) < n_trials:
            self.study.optimize(objective, 1)
            self.save()


if __name__ == "__main__":
    file = Path("../../runs/bayes.bin")
    file.unlink(missing_ok=True)


    def obj(trial: optuna.Trial):
        x = trial.suggest_float("x", 0, 1, step=.2)
        y = trial.suggest_float("y", 0, 1, step=.2)

        if trial.should_prune(): raise optuna.TrialPruned()
        return x ** 2 + y


    opt = BayesOpt(file)
    opt(obj, 10)
