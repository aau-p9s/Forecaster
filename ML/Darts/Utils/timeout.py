import signal
from Api.lib.variables import train_timeout

def signal_handler(arg1, arg2):
    raise RuntimeError("Timed out!")

def timeout(time=train_timeout):
    def outer_wrapper(f):
        def wrapper(*kwargs):
            if not train_timeout == -1:
                signal.signal(signal.SIGALRM, signal_handler)
                signal.alarm(time)
            return f(*kwargs)
        return wrapper
    return outer_wrapper
