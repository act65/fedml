from functools import partial
from deai.logging import FileLogger, error_handling_decorator
import threading

@partial(error_handling_decorator, logger=FileLogger(f"logs/test"))
def test():
    assert False

if __name__ == "__main__":
    threading.Thread(target=test).start()