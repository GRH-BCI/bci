import multiprocessing
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import click

from bci.eeg import EEG
from param_search import load_model
from bci.util import RealtimeModel

import optuna
from util.optimize import optimize_parallel


def test(trial: optuna.trial.FrozenTrial, *, eegs, window_size, n_preds, preds_per_sec):
    model = load_model(trial, window_size=window_size)
    model = RealtimeModel(model, window_size=window_size, n_preds=n_preds, preds_per_sec=preds_per_sec)

    for eeg in eegs:
        y_pred, y_true = model.test(eeg, return_='y_pred'), eeg.y
        uncertain_mask = np.isnan(y_pred)
        correct = (y_pred == y_true).astype(float)
        correct[uncertain_mask] = np.nan
        acc = np.nanmean(correct)
        uncertain = uncertain_mask.astype(float).sum() / y_pred.shape[0]
        print(f'{acc * 100}%, with {uncertain * 100}% uncertain')
        if uncertain == 0:
            from sklearn.metrics import confusion_matrix
            print(confusion_matrix(y_true, y_pred, normalize='true'))


@click.command()
@click.argument('db')
@click.argument('study-name')
@click.option('--dataset', type=click.Path(exists=True), default='C:/datasets/wearable-sensing', show_default=True)
@click.option('--eeg', multiple=True, type=str, show_default=True)
@click.option('--window-size', default=3.5, show_default=True)
@click.option('--n-preds', default=4, show_default=True)
@click.option('--preds-per-sec', default=4, show_default=True)
def main(*,
         db: str,
         study_name: str,
         dataset: str,
         eeg: Tuple[str],
         window_size: float,
         n_preds: int,
         preds_per_sec: int,
         ):
    eegs = [EEG.load(Path(dataset)/e) for e in eeg]
    study = optuna.load_study(
        study_name=study_name,
        storage=db,
    )
    test(study.best_trial, eegs=eegs, window_size=window_size, n_preds=n_preds, preds_per_sec=preds_per_sec)


if __name__ == '__main__':
    main()
