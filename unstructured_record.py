import json
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from threading import Thread

import numpy as np

from bci.app import App, wait_for_threads
from bci.util import FileRecorder, collect_input


def main(app: App, *, path: Path):
    app.connect_to_headset()
    file_logger = FileRecorder(path, app.dsi_input.get_channel_names())

    threads = [
        Thread(target=lambda: collect_input(app, listeners=[file_logger])),
        Thread(target=lambda: experiment(app)),
    ]
    for t in threads:
        t.start()
    wait_for_threads(threads)


# def experiment(app: App):
#     HZ = 20
#
#     app.leds.frequencies = np.array([HZ, HZ, HZ, HZ])
#
#     app.connect_to_leds()
#     app.calibrate_leds()
#
#     app.gui.set_text('Click to start')
#     app.gui.wait_for_click()
#
#     app.dsi_input.push_marker(app.dsi_input.latest_timestamp, 'baseline')
#     app.gui.set_text('Baseline (30 seconds)')
#     time.sleep(30)
#
#     app.gui.set_text('Click to continue')
#     app.gui.wait_for_click()
#
#     app.dsi_input.push_marker(app.dsi_input.latest_timestamp, f'{HZ}Hz')
#     app.gui.set_text('Frequency (30 seconds)')
#     app.leds.start([0, 1, 2, 3])
#     time.sleep(30)
#     app.leds.stop()
#     app.gui.kill()


def experiment(app: App):
    time.sleep(60)
    app.gui.kill()
    app.dsi_input.stop()


if __name__ == '__main__':
    config = json.load(open('config.json'))

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    path = Path(config['dataset_path']) / timestamp
    path.mkdir(parents=True, exist_ok=False)

    App(
        experiment_func=partial(main, path=path),
        headset_port=config['headset_port'],
        leds_port=config['leds_port'],
        fullscreen=False,
    ).loop()
