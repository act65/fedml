import logging
import traceback
import functools

class FlexibleLogger(logging.Logger):
    def __init__(self, name, level=logging.INFO):
        super().__init__(name, level)

    def metric(self, name, value, step=None):
        log_message = f"Metric: {name} = {value}"
        if step is not None:
            log_message += f", Step: {step}"
        self.info(log_message)

class PrintLogger(FlexibleLogger):
    def __init__(self, name, level=logging.INFO):
        super().__init__(name, level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.addHandler(ch)

class FileLogger(FlexibleLogger):
    def __init__(self, name, level=logging.INFO, log_file_path=None):
        super().__init__(name, level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if log_file_path is None:
            log_file_path = f'{name}.log'  # Default log file name based on logger name
        fh = logging.FileHandler(log_file_path)
        fh.setFormatter(formatter)
        self.addHandler(fh)

def error_handling_decorator(func, logger):
    """
    Decorator to run a function in a try-except block and log errors using a FlexibleLogger.

    Args:
        logger: An instance of a FlexibleLogger (or subclass) to use for logging errors.
    """
    @functools.wraps(func)
    def wrapped_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"Exception in function '{func.__name__}': {e}"
            logger.error(error_message) # Use the provided logger instance
            logger.error(traceback.format_exc())
            raise e
    return wrapped_function