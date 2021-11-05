import multiprocessing
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import click

from bci.eeg import EEG
from cca.model import load_model
from bci.util import RealtimeModel

import optuna
from util.optimize import optimize_parallel


def objective(trial: optuna.Trial, *, eegs, window_size, n_preds, preds_per_sec):
    model = load_model(trial, window_size=window_size)
    model = RealtimeModel(model, window_size=window_size, n_preds=n_preds, preds_per_sec=preds_per_sec)

    import cca.model
    cca_model = model.model  # type: cca.model.CCA
    baseline = np.mean([
        cca_model.predict(eeg, return_weights=True).mean(axis=0)
        for eeg in eegs
    ], axis=0)
    cca_model.baseline = baseline

    return np.mean([model.test(eeg) for eeg in eegs])


@click.command()
@click.argument('db')
@click.option('--dataset', type=click.Path(exists=True), default='C:/datasets/wearable-sensing', show_default=True)
@click.option('--eeg', multiple=True, type=str, show_default=True)
@click.option('--window-size', default=3.5, show_default=True)
@click.option('--n-preds', default=4, show_default=True)
@click.option('--preds-per-sec', default=4, show_default=True)
@click.option('--subject', default='', show_default=True)
@click.option('--study', default='{timestamp}-{subject}-window_size={window_size}', show_default=True)
@click.option('--timestamp', default=datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), show_default=True)
@click.option('--n-trials', default=1500, show_default=True)
@click.option('--n-jobs', default=multiprocessing.cpu_count()-1, show_default=True)
@click.option('--baseline', default=False, type=bool)
def main(*,
         db: str,
         dataset: str,
         eeg: Tuple[str],
         window_size: float,
         n_preds: int,
         preds_per_sec: int,
         subject: str,
         study: str,
         timestamp: str,
         n_trials: int,
         n_jobs: int,
         baseline: bool,
         ):
    eegs = [EEG.load(Path(dataset)/e) for e in eeg]
    study = optuna.create_study(
        study_name=study.format(window_size=window_size, n_preds=n_preds, preds_per_sec=preds_per_sec, subject=subject, timestamp=timestamp),
        storage=db,
        directions=['maximize'],
        sampler=optuna.samplers.NSGAIISampler(),
        load_if_exists=True,
    )
    # study.optimize(
    #     partial(objective, window_size=window_size, eegs=eegs, n_preds=n_preds, preds_per_sec=preds_per_sec),
    #     n_trials=n_trials,
    #     catch=(ValueError,),
    # )
    optimize_parallel(
        study,
        partial(objective, window_size=window_size, eegs=eegs, n_preds=n_preds, preds_per_sec=preds_per_sec),
        n_trials=n_trials,
        n_jobs=n_jobs,
        catch=(ValueError,),
    )
    print(study.best_value)
    print(study.best_params)


if __name__ == '__main__':
    main()
