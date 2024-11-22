from time import time
from contextlib import contextmanager
from datetime import timedelta
from rich.pretty import pprint


@contextmanager
def simple_timer(name):
    pprint(f"Starting {name}")
    start = time()
    yield
    end = time()
    dt = timedelta(seconds=end - start)
    pprint(f"{name}: {str(dt)}")
