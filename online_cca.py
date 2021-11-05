import json
from functools import partial
from threading import Thread

import optuna as optuna

from bci.app import App
from cca.model import load_model
from bci.util import RealtimeModel, InputDistributor


def main(app: App, *, model: RealtimeModel):
    app.connect_to_headset()
    app.connect_to_leds()
    app.calibrate_leds()

    input_distributor = InputDistributor(app, listeners=[model])
    input_distributor.wait_for_connection()

    Thread(target=lambda: input_distributor.loop()).start()

    experiment(app, model=model)


def experiment(app: App, *, model: RealtimeModel):
    app.leds.start([0, 1, 2, 3])

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
