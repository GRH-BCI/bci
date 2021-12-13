import time
from threading import Thread
import numpy as np

from .dsi_input import DSIInput
from .gui import BCIGUI
from .leds import LEDs


class App:
    def __init__(self, *, freq=(12, 13, 14, 15), experiment_func,
                 headset=None, leds=None,
                 headset_port='COM19', leds_port='COM21',
                 fullscreen=False, gui=True):
        self.freq = np.array(freq)
        self.gui = BCIGUI(fullscreen=fullscreen) if gui else None
        self.done = False
        self.leds = leds if leds is not None else LEDs(self.freq)
        self.dsi_input = headset if headset is not None else DSIInput()
        self.experiment_func = experiment_func
        self.headset_port = headset_port
        self.leds_port = leds_port

    def kill(self):
        self.done = True
        if self.gui is not None:
            self.gui.kill()
        self.dsi_input.stop()

    def loop(self):
        threads = [
            Thread(target=self.headset_thread, daemon=False, name='HeadsetThread'),
            Thread(target=self.experiment_thread, daemon=False, name='ExperimentThread'),
        ]
        try:
            for t in threads:
                t.start()
            while not self.done:
                if self.gui is not None:
                    self.done = self.done or not self.gui.tick()
                else:
                    time.sleep(0.1)
        finally:
            self.kill()
            for t in threads:
                t.join()
            self.leds.stop()

    def headset_thread(self):
        try:
            while not self.dsi_input.is_attached() and not self.done:
                time.sleep(0.1)
            if self.dsi_input.is_attached():
                self.dsi_input.loop()
                time.sleep(0.1)
        except Exception as ex:
            print('Exception in headset_thread:', ex)
        finally:
            self.kill()
            print('headset_thread finished!')

    def experiment_thread(self):
        try:
            self.experiment_func(self)
        except Exception as ex:
            print('Exception in experiment_thread:', ex)
        finally:
            self.kill()
            print('experiment_thread finished!')

    def set_text(self, str):
        if self.gui is not None:
            self.gui.set_text(str)
        if str:
            print(f'Text: "{str}"')

    def set_arrow(self, dir):
        if self.gui is not None:
            self.gui.set_arrow(dir)
        print('Direction:', dir)

    def wait_for_response(self):
        if self.gui is not None:
            self.gui.wait_for_click()
        else:
            input('[GUI unavailable. Press enter to continue]')

    def connect_to_headset(self, retry=True):
        if self.dsi_input.is_attached():
            return

        self.set_text('Connecting to headset')
        while not self.done:
            try:
                self.dsi_input.attach(self.headset_port)
                break
            except Exception as ex:
                print('Exception in headset attach:', ex)
                if not retry:
                    break
                time.sleep(1)
        self.set_text('')

    def calibrate_leds(self):
        if not self.leds.connected and not self.done:
            raise RuntimeError('Cannot calibrate unconnected leds')

        self.set_text('Position LEDs according to arrows. Tap when ready.')
        calibrating = True

        def calibrate_leds_thread():
            while not self.done:
                for i, dir in enumerate(['left', 'right', 'top', 'bottom']):
                    if not calibrating:
                        return

                    self.set_arrow(dir)
                    self.leds.start(i)
                    time.sleep(1)
                    self.set_arrow(None)
                    self.leds.stop()
                    time.sleep(0.5)

        t = Thread(target=calibrate_leds_thread, name='CalibrateLEDsThread')
        t.start()
        self.wait_for_response()
        self.set_text('')
        calibrating = False
        t.join()

    def connect_to_leds(self, retry=True):
        if self.leds.connected:
            return

        self.set_text('Connecting to LEDs')
        while not self.done:
            try:
                self.leds.connect(self.leds_port)
                break
            except Exception as ex:
                print(ex)
                if not retry:
                    break
                time.sleep(1)
        self.set_text('')
