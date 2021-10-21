import json
import time
from functools import partial
from threading import Thread

import optuna as optuna

from bci.app import App, wait_for_threads
from bci.model import load_model
from bci.util import RealtimeModel, collect_input


def main(app: App, *, model: RealtimeModel):
    app.connect_to_headset()
    app.connect_to_leds()
    app.calibrate_leds()

    threads = [
        Thread(target=lambda: collect_input(app, listeners=[model])),
        Thread(target=lambda: experiment(app, model=model)),
    ]
    for t in threads:
        t.start()
    wait_for_threads(threads)


def experiment(app: App, *, model: RealtimeModel):
    app.leds.start([0, 1, 2, 3])

    while not app.dsi_input.is_attached():
        time.sleep(0.1)

    while True:
        model.clear_buffers()
        y_pred = model.predict()
        dir = ['left', 'right', 'top', 'bottom'][y_pred]
        app.gui.set_arrow(dir)


if __name__ == '__main__':
    config = json.load(open('config.json'))

    study = optuna.load_study(config['optuna_study_name'],
                              storage=config['optuna_storage'])
    trial = study.best_trial

    n_preds = 4
    preds_per_sec = 4

    window_size = trial.params['window_size']
    model = load_model(trial, window_size=window_size)
    model = RealtimeModel(model, window_size=window_size, n_preds=n_preds, preds_per_sec=preds_per_sec)
    App(
        fullscreen=True,
        experiment_func=partial(main, model=model),
        headset_port=config['headset_port'],
        leds_port=config['leds_port'],
    ).loop()
