# 导入必要的库
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
import csv
import os
import tensorflow as tf

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # 生成一个合成的时间序列数据集
# np.random.seed(1)
# data = np.sin(np.arange(1000) * 0.01) + np.random.normal(0, 0.1, 1000)
data = np.load("data.npy")  # np.array([])

if not data.size:
    # 打开CSV文件
    with open('test.csv', mode='r', encoding='utf-8') as csvfile:
        # 创建一个CSV阅读器
        csv_reader = csv.reader(csvfile)
        data = []
        for row in csv_reader:
            data.append(row[2])
        data = data[1:]
        data = np.array(data).astype(dtype=float)
        np.save("data.npy", data)

data = data[0:1000]


# 定义一个函数来将时间序列转换为监督学习问题
def series_to_supervised(data, n_in=1, n_out=1):
    df = pd.DataFrame(data)
    cols = list()
    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # 将所有列拼接起来
    agg = pd.concat(cols, axis=1)
    # 删除含有缺失值的行
    agg.dropna(inplace=True)
    return agg.values


# 将时间序列转换为监督学习问题，使用前5个时间步作为输入，下一个时间步作为输出
X = series_to_supervised(data, 5, 1)[:, :-1]
y = series_to_supervised(data, 5, 1)[:, -1]

# 划分训练集和测试集，使用前800个样本作为训练集，后面的作为测试集
train_X = X[:800]
train_y = y[:800]
test_X = X[800:]
test_y = y[800:]

# 将输入数据调整为LSTM所需的三维格式 [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], 1))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))

train = 0

model = Sequential()
if train:
    # 定义LSTM回归模型，使用一个隐藏层，50个神经元，输出层是一个全连接层，损失函数是均方误差，优化器是Adam
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 训练模型，使用32个批次大小，100个迭代周期，不打印训练进度
    history = model.fit(train_X, train_y, batch_size=32, epochs=100, verbose=0)
    model.save("the.model")
else:
    model = keras.models.load_model("the.model")

# # 绘制训练损失曲线
# plt.plot(history.history['loss'], label='train')
# plt.legend()
# plt.show()

mytest = np.array([])
mytest.resize(test_X.shape[0] * test_X.shape[1])
mytest = mytest.reshape((test_X.shape[0], test_X.shape[1], 1))
mytest[-5:, :, 0] = [43., 47.6, 42.2, 40.77, 40.51]
print(test_X)
print(mytest)

# 在测试集上进行预测
yhat = model.predict(test_X)

# 绘制真实值和预测值的对比图
plt.plot(test_y, label='actual')
plt.plot(yhat, label='predicted')
plt.legend()
plt.show()
