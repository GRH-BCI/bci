from typing import Optional

import serial
from more_itertools import always_iterable


class LEDs:
    """
    Interfaces with external Arduino to turn on/off strobe LEDs.

    Note that this **doesn't work** if the Arduino has additional peripherals plugged in. I don't know why. I think it's
    a bug in `BCI_and_Strobe.ino`, but maybe I'm just interfacing with it incorrectly.
    """
    def __init__(self, frequencies):
        self.connected = False
        self.frequencies = frequencies
        self.port = None  # type: Optional[serial.Serial]

    def connect(self, port_name):
        # Baud-rate matches line 151 of `BCI_and_Strobe.ino`
        self.port = serial.Serial(port_name, baudrate=250000)
        self.connected = True

    def start(self, which):
        """
        Starts some/all strobe LEDs.
        :param which: integer or list of integers indicating which strobe LEDs to turn on.
        """
        for led in always_iterable(which):
            self.port.write('1{}{}{}{}050{:0>4}'.format(
                1 if led == 0 else 0,
                1 if led == 1 else 0,
                1 if led == 2 else 0,
                1 if led == 3 else 0,
                self.frequencies[led],
            ).encode())

    def stop(self):
        """ Stop all strobe LEDs. """
        if self.connected:
            self.port.write(b'111110000000')
