from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta
from time import time
from itertools import islice

from rich.pretty import pprint


def batched(iterable, n):
    """Batch data into lists of length *n*. The last batch may be shorter.

    >>> list(batched('ABCDEFG', 3))
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


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
