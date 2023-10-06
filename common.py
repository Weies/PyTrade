import csv
# 以cell方式实现RNN
# %%
import os

import numpy as np
import pandas as pd

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers, losses, optimizers, Sequential
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def MA(array=[], day=5):
    num_days = len(array)
    a = np.array([0.0 for i in range(num_days)])
    for i in range(num_days):
        a[i] = array[i] + (a[i - 1] if i > 0 else 0)

    for i in range(num_days - 1, 0, -1):
        true_day = day if i > day else i
        a[i] = (a[i] - a[i - true_day]) / true_day

    return a


def BOLL(array=[], days=20):
    UB, M, BB = array.copy(), MA(array, 20), array.copy()

    num_days = len(array)

    for i in range(num_days - 1, 0, -1):
        std = np.std(array[max(i - days, 0): i])
        UB[i] = M[i] + std * 2
        BB[i] = M[i] - std * 2

    return UB, M, BB


def MACD(array=[]):
    a = pd.DataFrame(array)
    ewm12 = a.ewm(span=12, adjust=False).mean()
    ewm26 = a.ewm(span=26, adjust=False).mean()
    dif = ewm12 - ewm26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = dif - dea
    return dif, dea, macd


# data.part(-160)
#
# plt.plot(data.close, label="close")
# UB, M, BB = BOLL(data.close)
# plt.plot(UB, label="UB")
# plt.plot(M, label="M")
# plt.plot(BB, label="BB")
# plt.legend()
# plt.show()
