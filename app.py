#!/usr/bin/env python3
import multiprocessing as mp
from Api.api import *
from ML.Darts.Utils.models import PositiveGaussianLikelihood

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    start_api()
