import multiprocessing
from copy import copy
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import click

from bci.eeg import EEG
from bci.util import RealtimeModel

import optuna

from fbcca import FBCCA
from util.optimize import optimize_parallel


def load_model(trial: optuna.trial.BaseTrial, *, window_size):
    channels = [c for c in [
        trial.suggest_categorical('Pz', ['Pz', '']),
        trial.suggest_categorical('P3', ['P3', '']),
        trial.suggest_categorical('P4', ['P4', '']),
        trial.suggest_categorical('T5', ['T5', '']),
        trial.suggest_categorical('T6', ['T6', '']),
        trial.suggest_categorical('O1', ['O1', '']),
        trial.suggest_categorical('O2', ['O2', '']),
    ] if c != '']
    if not channels:
        raise optuna.exceptions.TrialPruned()

    ref = trial.suggest_categorical('ref', ['A2', 'Cz', ''])

    filter_band = trial.suggest_uniform('fb_lo', 0.01, 10), trial.suggest_uniform('fb_hi', 40, 149)
    fb_mode = trial.suggest_categorical('fb_mode', ['a', 'b', 'c'])
    fb_start = trial.suggest_uniform('fb_start', 5, 15)
    fb_step = trial.suggest_uniform('fb_step', 1, 50)
    n_fbs = trial.suggest_int('n_fbs', 1, 5)
    if fb_mode == 'a':
        filter_banks = [(fb_start, fb_start + (i + 1) * fb_step)
                        for i in range(n_fbs)]
    elif fb_mode == 'b':
        filter_banks = [(fb_start + i * fb_step, fb_start + (i + 1) * fb_step)
                        for i in range(n_fbs)]
    elif fb_mode == 'c':
        filter_banks = [(fb_start + i * fb_step, fb_start + (n_fbs) * fb_step)
                        for i in range(n_fbs)]
    else:
        raise ValueError(f'Invalid fb_mode {fb_mode}')

    filter_banks = [
        (min(start, 120), min(stop, 120))
        for start, stop in filter_banks
        if start < 120 or stop < 120
    ]

    n_harmonics = trial.suggest_int('n_harmonics', 1, 5)

    a = trial.suggest_uniform('a', 0.01, 2)
    b = trial.suggest_uniform('b', 0.01, 2)
    window = 0, window_size

    def preprocess(eeg: EEG):
        eeg = copy(eeg)
        if ref:
            eeg.X = eeg.X - eeg.X[:, :, eeg.montage == ref]
        eeg.X = eeg.X[:, :, np.isin(eeg.montage, channels)]
        eeg.X = eeg.X[:, int(eeg.fs * window[0]):int(eeg.fs * window[1]), :]
        eeg = eeg.notch(60).bandpass(filter_band)
        return eeg

    model = FBCCA(filter_banks=filter_banks,
                  n_harmonics=n_harmonics, a=a, b=b,
                  preprocess=preprocess)
    return model



def objective(trial: optuna.Trial, *, eegs, window_size, n_preds, preds_per_sec):
    model = load_model(trial, window_size=window_size)
    model = RealtimeModel(model, window_size=window_size, n_preds=n_preds, preds_per_sec=preds_per_sec)

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
         ):
    eegs = [EEG.load(Path(dataset)/e) for e in eeg]
    study = optuna.create_study(
        study_name=study.format(window_size=window_size, n_preds=n_preds, preds_per_sec=preds_per_sec, subject=subject, timestamp=timestamp),
        storage=db,
        directions=['maximize'],
        sampler=optuna.samplers.NSGAIISampler(),
        load_if_exists=True,
    )

    if n_jobs > 0:
        optimize = lambda **kwargs: optimize_parallel(study, n_jobs=n_jobs, **kwargs)
    else:
        optimize = lambda **kwargs: study.optimize(**kwargs)

    optimize(
        func=partial(objective, window_size=window_size, eegs=eegs, n_preds=n_preds, preds_per_sec=preds_per_sec),
        n_trials=n_trials,
        catch=(ValueError, FloatingPointError),
    )

    print(study.best_value)
    print(study.best_params)


if __name__ == '__main__':
    main()
