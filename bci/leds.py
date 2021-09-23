from typing import Optional

import serial
from more_itertools import always_iterable


class LEDs:
    def __init__(self, frequencies):
        self.connected = False
        self.frequencies = frequencies
        self.port = None  # type: Optional[serial.Serial]

    def connect(self, port_name):
        self.port = serial.Serial(port_name, baudrate=250000)
        self.connected = True

    def start(self, which):
        for led in always_iterable(which):
            self.port.write('1{}{}{}{}050{:0>4}'.format(
                1 if led == 0 else 0,
                1 if led == 1 else 0,
                1 if led == 2 else 0,
                1 if led == 3 else 0,
                self.frequencies[led],
            ).encode())

    def stop(self):
        self.port.write(b'111110000000')
