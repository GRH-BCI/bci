import json
import pickle
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from threading import Thread

import numpy as np
import optuna as optuna

from bci.app import App, wait_for_threads
from bci.model import load_model
from bci.util import FileRecorder, collect_input, RealtimeModel


def main(app: App, *, model: RealtimeModel, path: Path):
    app.connect_to_headset()
    app.connect_to_leds()
    app.calibrate_leds()

    file_logger = FileRecorder(path, app.dsi_input.channel_names)

    threads = [
        Thread(target=lambda: collect_input(app, listeners=[file_logger, model])),
        Thread(target=lambda: experiment(app, model=model, path=path)),
    ]
    for t in threads:
        t.start()
    wait_for_threads(threads)


def experiment(app: App, *, model: RealtimeModel, path: Path):
    app.leds.start([0, 1, 2, 3])

    while not app.dsi_input.connected:
        time.sleep(0.1)

    ys_true = np.repeat([0, 1, 2, 3], 4).astype(int)
    np.random.shuffle(ys_true)
    ys_pred = []

    for y_true in ys_true:
        dir_true = ['left', 'right', 'top', 'bottom'][y_true]
        app.gui.set_arrow(dir_true)
        time.sleep(0.5)  # Allow some reaction time
        model.clear_buffers()
        timestamp = app.dsi_input.latest_timestamp
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

    app.leds.stop()
    app.gui.kill()


if __name__ == '__main__':
    config = json.load(open('config.json'))

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    path = Path(config['dataset_path']) / timestamp
    path.mkdir(parents=True, exist_ok=False)

    study = optuna.load_study(config['optuna_study_name'],
                              storage=config['optuna_storage'])
    trial = study.best_trial

    n_preds = 4
    preds_per_sec = 4

    pickle.dump(trial, open(path/'trial.pickle', 'wb'))
    json.dump({'n_preds': n_preds, 'preds_per_sec': preds_per_sec}, open(path/'metadata.json', 'w'))

    # window_size = 7
    window_size = trial.params['window_size']
    model = load_model(trial, window_size=window_size)
    model = RealtimeModel(model, window_size=window_size, n_preds=n_preds, preds_per_sec=preds_per_sec)
    App(
        experiment_func=partial(main, model=model, path=path),
        headset_port=config['headset_port'],
        leds_port=config['leds_port'],
    ).loop()
