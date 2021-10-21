import time
from threading import Thread
from typing import List

import numpy as np
import pygame
from pylsl import pylsl

from .DSIInputCpp import DSIInput
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

    def loop(self):
        threads = [
            Thread(target=self.headset_thread),
            Thread(target=self.experiment_thread),
        ]
        for t in threads:
            t.start()

        while self.gui.tick():
            for t in threads:
                try:
                    t.join(timeout=0)
                except TimeoutError:
                    pass
        pygame.quit()  # XXX: shouldn't have to call pygame directly here

    def headset_thread(self):
        while not self.dsi_input.is_attached():
            time.sleep(0.1)
        self.dsi_input.loop()

    def experiment_thread(self):
        try:
            self.experiment_func(self)
        except Exception as ex:
            print('Exception in experiment_thread:', ex)
            raise

    def lsl_out_thread(self):
        try:
            while not self.dsi_input.is_attached():
                time.sleep(0.1)
            info = pylsl.StreamInfo('WearableSensing', 'EEG', self.dsi_input.n_channels,
                                    self.dsi_input.fs, 'float32', __file__)
            channels = info.desc().append_child('channels')
            for name in self.dsi_input.channel_names:
                channel = channels.append_child('channel')
                channel.append_child_value('label', name)
                channel.append_child_value('type', 'EEG')
            outlet = pylsl.StreamOutlet(info, 9)
            while True:
                try:
                    timestamp, data = self.dsi_input.pull()
                    # print(timestamp)
                    outlet.push_sample(data, timestamp)
                    time.sleep(0.001)
                except IndexError:
                    continue

        except Exception as ex:
            print('Exception in lsl_out_thread:', ex)

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
