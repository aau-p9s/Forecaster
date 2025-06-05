import signal
from Api.lib.variables import train_timeout
import traceback

def signal_handler(arg1, arg2):
    raise RuntimeError("Timed out!")

def timeout(time=train_timeout):
    def outer_wrapper(f):
        def wrapper(*args, **kwargs):
            if not time == -1:
                signal.signal(signal.SIGALRM, signal_handler)
                signal.alarm(time)
            try:
                return f(*args, **kwargs)
            except RuntimeError:
                pass
            except Exception as e:
                traceback.print_exc()
                print("fuck...")
            finally:
                if not time == -1:
                    signal.alarm(0)
        return wrapper
    return outer_wrapper
