import contextlib
from datetime import datetime


@contextlib.contextmanager
def timer(section=None):
    """ Context manager that prints the duration of the context. """
    start_time = datetime.now()
    try:
        yield
    finally:
        end_time = datetime.now()
        print(f'{section + " took" if section is not None else "Took"} {end_time-start_time}')
