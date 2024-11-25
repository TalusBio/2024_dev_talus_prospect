from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta
from time import time

from rich.pretty import pprint


@contextmanager
def simple_timer(name: str) -> Generator[None, None, None]:
    """Implements a simple timer context manager.

    Args:
        name (str): The name of the timer.

    Yields
    ------
        None

    """
    pprint(f"Starting {name}")
    start = time()
    yield
    end = time()
    dt = timedelta(seconds=end - start)
    pprint(f"{name}: {str(dt)}")
