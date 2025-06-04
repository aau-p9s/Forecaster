import signal

def signal_handler(arg1, arg2):
    raise RuntimeError("Timed out!")

def timeout(f, time=300):
    def wrapper(*kwargs):
        signal.signal(signal.SIGKILL, signal_handler)
        signal.alarm(time)
        try:
            return f(*kwargs)
        except RuntimeError as e:
            return None
    return wrapper
