import time
from copy import copy
from pathlib import Path
from typing import Tuple

import numpy as np
import click

from bci.eeg import EEG
from fbcca.cca import CACC
from bci.util import RealtimeModel, chunkify, timer


def test(eegs_train, eegs_test, *, window_size: float, n_preds: int, preds_per_sec: int):
    channels = ['Pz', 'P3', 'P4', 'T5', 'T6', 'O1', 'O2']
    filter_band = 1, 50

    def preprocess(eeg: EEG):
        eeg = copy(eeg)
        eeg.X = eeg.X[:, :, np.isin(eeg.montage, channels)]
        eeg = eeg.notch(60).bandpass(filter_band)
        return eeg

    model = CACC(preprocess=preprocess)

    with timer('Training'):
        for eeg in eegs_train:
            model.train_add(eeg)
        model.train_finish()

    model = RealtimeModel(model, window_size=window_size, n_preds=n_preds, preds_per_sec=preds_per_sec)

    with timer('Testing'):
        for eeg in eegs_test:
            # model.train(eeg)
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
@click.option('--dataset', type=click.Path(exists=True), default='C:/datasets/wearable-sensing', show_default=True)
@click.option('--eeg', multiple=True, type=str, show_default=True)
@click.option('--window-size', default=3.5, show_default=True)
@click.option('--n-preds', default=4, show_default=True)
@click.option('--preds-per-sec', default=4, show_default=True)
def main(*,
         dataset: str,
         eeg: Tuple[str],
         window_size: float,
         n_preds: int,
         preds_per_sec: int,
         ):
    with timer('Loading'):
        eegs_test = [EEG.load(Path(dataset)/e) for e in eeg]
        eegs_train = [chunkify(eeg, window_size=window_size) for eeg in eegs_test]
    test(eegs_train=eegs_train, window_size=window_size, eegs_test=eegs_test, n_preds=n_preds, preds_per_sec=preds_per_sec)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Took {end - start} seconds.')
