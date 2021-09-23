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

        self.headset.SetSampleCallback(data_callback, None)
        while True:
            self.headset.Idle(1.0)

    def message_callback(self, message, level=0):
        print(f'DSI Message (level {level}): {message.decode()}')
        return 1

    def data_callback(self, _headset, timestamp, _userdata):
        self.push(timestamp, [
            self.headset.GetChannelByIndex(i).GetSignal()
            for i in range(self.n_channels)
        ])

    def push(self, timestamp, data):
        # print(data)
        self.data.append((timestamp, data))
        self.latest_timestamp = max(self.latest_timestamp, timestamp)

    def push_marker(self, timestamp, marker):
        self.markers.append((timestamp, marker))
        self.latest_timestamp = max(self.latest_timestamp, timestamp)

    def pull(self):
        return self.data.popleft()

    def pull_marker(self):
        return self.markers.popleft()
