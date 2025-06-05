#!/usr/bin/env python3
import multiprocessing as mp
from Api.api import *

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    start_api()
