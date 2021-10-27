import time
from threading import Thread
from typing import List

import numpy as np
import pygame
from pylsl import pylsl

from .dsi_input import DSIInput
from .gui import BCIGUI
from .leds import LEDs


def wait_for_threads(threads: List[Thread], *, delay=0.1):
    running = [True for t in threads]
    while any(running):
        for i, t in enumerate(threads):
            try:
                t.join(timeout=delay)
                running[i] = False
            except TimeoutError:
                continue


class App:
    def __init__(self, *, freq=(12, 13, 14, 15), experiment_func,
                 headset_port='COM19', leds_port='COM21',
                 fullscreen=False):
        self.freq = np.array(freq)
        self.gui = BCIGUI(fullscreen=fullscreen)
        self.leds = LEDs(self.freq)
        self.dsi_input = DSIInput()
        self.experiment_func = experiment_func
        self.headset_port = headset_port
        self.leds_port = leds_port

    def kill(self):
        self.gui.kill()
        self.leds.stop()
        self.dsi_input.stop()

    def loop(self):
        threads = [
            Thread(target=self.headset_thread, daemon=True),
            Thread(target=self.experiment_thread, daemon=True),
        ]
        for t in threads:
            t.start()

        while self.gui.tick():
            for t in threads:
                try:
                    t.join(timeout=0)
                except TimeoutError:
                    pass

        self.kill()

    def headset_thread(self):
        try:
            while not self.dsi_input.is_attached():
                time.sleep(0.1)
            self.dsi_input.loop()
        except Exception as ex:
            print('Exception in headset_thread:', ex)
            raise

    def experiment_thread(self):
        try:
            self.experiment_func(self)
            print('Experiment finished!')
            self.kill()
        except Exception as ex:
            print('Exception in experiment_thread:', ex)
            raise

    def connect_to_headset(self):
        self.gui.set_text('Connecting to headset')
        while True:
            try:
                self.dsi_input.attach(self.headset_port)
                break
            except Exception as ex:
                print(ex)
                time.sleep(1)
        self.gui.set_text('')

    def calibrate_leds(self):
        self.gui.set_text('Calibrating LEDs. Tap when ready.')
        calibrating = True

        def calibrate_leds_thread():
            while True:
                for i, dir in enumerate(['left', 'right', 'top', 'bottom']):
                    if not calibrating:
                        return

                    self.gui.set_arrow(dir)
                    self.leds.start(i)
                    time.sleep(1)
                    self.gui.set_arrow(None)
                    self.leds.stop()
                    time.sleep(0.5)

        t = Thread(target=calibrate_leds_thread)
        t.start()
        self.gui.wait_for_click()
        self.gui.set_text('')
        calibrating = False
        t.join()

    def connect_to_leds(self):
        self.gui.set_text('Connecting to LEDs')
        while True:
            try:
                self.leds.connect(self.leds_port)
                break
            except Exception as ex:
                print(ex)
                time.sleep(1)
        self.gui.set_text('')
