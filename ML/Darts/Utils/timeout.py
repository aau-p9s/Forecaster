import signal

def signal_handler(arg1, arg2):
    raise RuntimeError("Timed out!")

def timeout(time=300):
    def outer_wrapper(f):
        def wrapper(*kwargs):
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(time)
            try:
                return f(*kwargs)
            except RuntimeError as e:
                return None
        return wrapper
    return outer_wrapper
