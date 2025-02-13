from threading import Thread
# This class is an extension on the built-in Thread class, which allows for returning data
class ReturnableThread(Thread):
    def __init__(self, target, name):
        Thread.__init__(self)
        self.target = target
        self.result = None
        self.name = name
    
    def run(self) -> None:
        self.result = self.target()