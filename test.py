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


def percent(per=""):
    if len(per) > 1:
        return float(per[0:-1])
    else:
        return 0.0


class StockData:
    def __init__(self):
        self.time = []
        self.day = []
        self.open = []
        self.high = []
        self.low = []
        self.close = []
        self.rise = []
        self.volume = []

    def push(self, time, day, open, high, low, close, rise, volume):
        self.time.append(time)
        self.day.append(day)
        self.open.append(open)
        self.high.append(high)
        self.low.append(low)
        self.close.append(close)
        self.rise.append(percent(rise))
        self.volume.append(volume)

    def better(self):
        self.open = np.array(self.open, dtype=float)
        self.high = np.array(self.high, dtype=float)
        self.low = np.array(self.low, dtype=float)
        self.close = np.array(self.close, dtype=float)
        self.rise = np.array(self.rise, dtype=float)
        self.volume = np.array(self.volume, dtype=float)

    def part(self, n):
        self.time = self.time[0:n]
        self.day = self.day[0:n]
        self.open = self.open[0:n]
        self.high = self.high[0:n]
        self.low = self.low[0:n]
        self.close = self.close[0:n]
        self.rise = self.rise[0:n]
        self.volume = self.volume[0:n]


data = StockData()

# 打开CSV文件
with open('test.csv', mode='r', encoding='utf-8') as csvfile:
    # 创建一个CSV阅读器
    csv_reader = csv.reader(csvfile)
    next(csv_reader)
    next(csv_reader)
    for row in csv_reader:
        data.push(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[8])
    data.better()


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


data.part(100)

# plt.plot(data.close, label="close")
UB, M, BB = MACD(data.close)
plt.plot(UB, label="UB")
plt.plot(M, label="M")
plt.plot(BB, label="BB")
plt.legend()
plt.show()
