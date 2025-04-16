import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('windows.csv')

sizes = [40, 80, 160]
plt.figure(figsize=(14, 6))

for i, size in enumerate(sizes):
    plot_df = df.loc[df['window_size'] == size]
    plot_df = plot_df.loc[plot_df['start_point'] < 1530]

    x = (plot_df['start_point'] + (size/2)).astype(int).astype(str)
    y = plot_df['accuracy']

    plt.subplot(3, 1, i + 1)
    plt.bar(x, y)
    plt.ylabel(f'Window Size {size}')

plt.show()