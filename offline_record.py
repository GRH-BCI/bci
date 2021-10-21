import json
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from threading import Thread

import numpy as np

from bci.app import App, wait_for_threads
from bci.util import FileRecorder, collect_input


def main(app: App, *, path: Path, window_size: float):
    app.connect_to_headset()
    app.connect_to_leds()
    app.calibrate_leds()

    file_logger = FileRecorder(path, app.dsi_input.get_channel_names())

    threads = [
        Thread(target=lambda: collect_input(app, listeners=[file_logger])),
        Thread(target=lambda: experiment(app, window_size=window_size)),
        Thread(target=lambda: file_logger.loop()),
    ]
    for t in threads:
        t.start()
    wait_for_threads(threads)


def experiment(app: App, window_size: float):
    app.leds.start([0, 1, 2, 3])

    while not app.dsi_input.is_attached():
        time.sleep(0.1)

    ys_true = np.repeat([0, 1, 2, 3], 4).astype(int)
    np.random.shuffle(ys_true)

    for y_true in ys_true:
        dir_true = ['left', 'right', 'top', 'bottom'][y_true]
        app.gui.set_arrow(dir_true)
        time.sleep(0.5)  # Allow some reaction time
        timestamp = app.dsi_input.get_latest_timestamp()
        app.dsi_input.push_marker(timestamp, dir_true)
        time.sleep(window_size)
        app.gui.set_arrow(None)
        app.gui.set_text('')
        time.sleep(2)  # Downtime

    app.gui.kill()
    app.dsi_input.stop()


if __name__ == '__main__':
    config = json.load(open('config.json'))

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    path = Path(config['dataset_path']) / timestamp
    path.mkdir(parents=True, exist_ok=False)

    window_size = 5
    App(
        experiment_func=partial(main, path=path, window_size=window_size),
        headset_port=config['headset_port'],
        leds_port=config['leds_port'],
    ).loop()
