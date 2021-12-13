import json
import pickle
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from threading import Thread

import numpy as np
import optuna as optuna

from bci.app import App
from fbcca.param_search import load_model
from bci.util import FileRecorder, InputDistributor, RealtimeModel


def main(app: App, *, model: RealtimeModel, path: Path):
    app.connect_to_headset()
    app.connect_to_leds()
    app.calibrate_leds()

    file_logger = FileRecorder(path, app.dsi_input.get_channel_names())
    input_distributor = InputDistributor(app, listeners=[file_logger, model])

    Thread(target=lambda: input_distributor.loop()).start()
    Thread(target=lambda: file_logger.loop()).start()

    experiment(app, model=model, path=path)


def experiment(app: App, *, model: RealtimeModel, path: Path):
    app.leds.start([0, 1, 2, 3])

    while not app.dsi_input.is_attached():
        time.sleep(0.1)

    ys_true = np.repeat([0, 1, 2, 3], 4).astype(int)
    np.random.shuffle(ys_true)
    ys_pred = []

    for y_true in ys_true:
        dir_true = ['left', 'right', 'top', 'bottom'][y_true]
        app.gui.set_arrow(dir_true)
        time.sleep(0.5)  # Allow some reaction time
        model.clear_buffers()
        timestamp = app.dsi_input.get_latest_timestamp()
        app.dsi_input.push_marker(timestamp, dir_true)
        try:
            y_pred = model.predict(timeout=10)
            dir_pred = ['left', 'right', 'top', 'bottom'][y_pred]
            app.gui.set_arrow(dir_pred)
        except TimeoutError:
            y_pred = np.nan
        ys_pred.append(y_pred)
        app.gui.set_text('Correct' if y_true == y_pred else f'Incorrect (y_true={y_true}, y_pred={y_pred})')
        time.sleep(2)  # Show result
        app.gui.set_arrow(None)
        app.gui.set_text('')
        time.sleep(1)  # Downtime

    print(np.mean(ys_true == ys_pred))

    np.savetxt(path/'y_true.txt', ys_true)
    np.savetxt(path/'y_pred.txt', ys_pred)


if __name__ == '__main__':
    window_size = 3.5
    n_preds = 4
    preds_per_sec = 4
    study_name = '2021-11-05-08-05-49-hosein-window_size=3.5'
    db = 'postgresql://postgres:i5gMr!Pfcdm$dn8YqhTf#$hL?jkb@localhost:5432/postgres'

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    path = Path('C:/datasets/wearable-sensing') / timestamp
    path.mkdir(parents=True, exist_ok=False)

    study = optuna.load_study(study_name, storage=db)
    trial = study.best_trial

    pickle.dump(trial, open(path/'trial.pickle', 'wb'))
    json.dump({'n_preds': n_preds, 'preds_per_sec': preds_per_sec}, open(path/'metadata.json', 'w'))

    model = load_model(trial, window_size=window_size)
    model = RealtimeModel(model, window_size=window_size, n_preds=n_preds, preds_per_sec=preds_per_sec)
    App(
        fullscreen=True,
        experiment_func=partial(main, model=model, path=path),
    ).loop()
