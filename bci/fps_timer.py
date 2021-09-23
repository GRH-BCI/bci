import time
import warnings


class FPSTimer:
    def __init__(self, fps):
        self.frame_budget = 1/fps
        self.start = time.time()

    def wait(self):
        # Based off of http://docs.ros.org/en/diamondback/api/rostime/html/rate_8cpp_source.html#l00040
        expected_end = self.start + self.frame_budget
        actual_end = time.time()

        if actual_end < self.start:
            expected_end = actual_end + self.frame_budget

        sleep_time = expected_end - actual_end
        actual_frame_time = actual_end - self.start
        self.start = expected_end

        if sleep_time < 0:
            warnings.warn(f'FPSTimer timedelta is: {actual_frame_time}')
            if actual_end > expected_end + self.frame_budget:
                self.start = actual_end
        else:
            time.sleep(sleep_time)
