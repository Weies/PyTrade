import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("data.csv") # 读取数据
df.index = pd.to_datetime(df['trade_date']) # 将日期设置为index
fig, ax = plt.subplots(figsize=(15,8))
mpf.candlestick2_ochl(ax, df['open'], df['close'], df['high'], df['low'], width=0.6, colorup='r', colordown='g')
plt.title("股票K线图")
plt.xlabel("日期")
plt.ylabel("股价")
plt.show()