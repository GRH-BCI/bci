import sys

import numpy as np

from . import DSI
from collections import deque


class DSIInput:
    def __init__(self):
        self.headset = DSI.Headset()
        self.connected = False
        self.n_channels, self.fs = None, None
        self.channel_names = []
        self.data = deque(maxlen=None)
        self.markers = deque(maxlen=None)
        self.latest_timestamp = 0

    def connect(self, port):
        @DSI.MessageCallback
        def message_callback(*args):
            return self.message_callback(*args)

        self.headset.SetMessageCallback(message_callback)
        self.headset.Connect(port)
        self.headset.StartBackgroundAcquisition()
        self.n_channels = self.headset.GetNumberOfChannels()
        self.fs = self.headset.GetSamplingRate()
        self.channel_names = []
        for i in range(self.n_channels):
            channel = self.headset.GetChannelByIndex(i)
            name = channel.GetName()
            self.channel_names.append(name)
        self.connected = True

    def loop(self):
        @DSI.SampleCallback
        def data_callback(*args):
            return self.data_callback(*args)

        self.headset.ReallocateBuffers(1, 1)
        self.headset.ConfigureBatch(100, 0.5)
        ts = []
        while True:
            t = self.headset.WaitForBatch()
            ts.append(t)
            print(ts)
            print(self.headset.GetNumberOfBufferedSamples())

    def message_callback(self, message, level=0):
        print(f'DSI Message (level {level}): {message.decode()}')
        return 1

    def data_callback(self, _headset, timestamp, _userdata):
        for _ in range(self.headset.GetNumberOfBufferedSamples()):
            self.push(timestamp, [
                self.headset.GetChannelByIndex(i).ReadBuffered()
                for i in range(self.n_channels)
            ])

    def push(self, timestamp, data):
        # print(data)
        if not(0.1 < timestamp < 1e100):
            return

        if not np.isclose(timestamp - self.latest_timestamp, 1 / self.fs):
            print(f'Unexpected time delta: {timestamp - self.latest_timestamp}', file=sys.stderr)

        self.data.append((timestamp, data))
        self.latest_timestamp = max(self.latest_timestamp, timestamp)

    def push_marker(self, timestamp, marker):
        if not (0.1 < timestamp < 1e100):
            return
        self.markers.append((timestamp, marker))
        self.latest_timestamp = max(self.latest_timestamp, timestamp)

    def pull(self):
        return self.data.popleft()

    def pull_marker(self):
        return self.markers.popleft()
