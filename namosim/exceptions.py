import signal
import typing as t
from contextlib import contextmanager


class TimeoutError(Exception):
    def __init__(self):
        pass


@contextmanager
def timeout(seconds: int):
    """
    A context manager that raises a `TimeoutError` if its context block fails
    to complete execution within the provided number of seconds.
    """

    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(seconds)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum: int, frame: t.Any):
    raise TimeoutError
