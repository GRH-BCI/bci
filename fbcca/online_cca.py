import json
from functools import partial
from threading import Thread

import optuna as optuna

from bci.app import App
from fbcca.param_search import load_model
from bci.input_distributor import RealtimeModel, InputDistributor


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
    window_size = 3.5
    n_preds = 4
    preds_per_sec = 4
    study_name = '2021-11-05-08-05-49-hosein-window_size=3.5'
    db = 'postgresql://postgres:i5gMr!Pfcdm$dn8YqhTf#$hL?jkb@localhost:5432/postgres'

    study = optuna.load_study(study_name, storage=db)
    trial = study.best_trial

    model = load_model(trial, window_size=window_size)
    model = RealtimeModel(model, window_size=window_size, n_preds=n_preds, preds_per_sec=preds_per_sec)
    App(
        fullscreen=True,
        experiment_func=partial(main, model=model),
    ).loop()
