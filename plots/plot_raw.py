import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

bacteria_files = {
    'E. coli': 'bacteria/ECOL1.csv',
    'P. aeruginosa': 'bacteria/Paeruginosa2.csv',
    'S. agalactiae': 'bacteria/Salgalactiae1.csv',
    'S. aureus': 'bacteria/Saureus2.csv'
}

linestyles = ['-', '--', '-.', ':']
colors = ['blue', 'red', 'orange', 'green']

plt.figure(figsize=(14, 4))

#plt.axvspan(170, 270, color='grey', alpha=0.3)
#plt.axvspan(580, 640, color='grey', alpha=0.3)
#plt.axvspan(710, 790, color='grey', alpha=0.3)


def plot_confidence_interval(df, label, color, plot=False):
  mean_line = df.mean(axis=0)
  ci = 1.96 * df.std(axis=0) / np.sqrt(len(df))  # ci=95% 
  plt.plot(mean_line, label=label, color=color)
  plt.fill_between(df.columns, mean_line - ci, mean_line + ci, color=color, alpha=0.2) #,label="95% Confidence Interval")
  if plot:
    plt.show()

for idx, (key, value) in enumerate(bacteria_files.items()):
    df = pd.read_csv(value, quotechar='"', skiprows=2)
    df = df.transpose().reset_index(drop=True)
    #df.plot()
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    #df = df.transpose()
    #df.plot(figsize=(14, 4))
    df = df[:10]
    plot_confidence_interval(df, label=key, color=colors[idx])

plt.legend()
plt.xlabel('Sample Points')
plt.ylabel('Mean Intensity')
plt.title('Mean Intensity of Bacteria Samples with Confidence Intervals')
plt.show()
