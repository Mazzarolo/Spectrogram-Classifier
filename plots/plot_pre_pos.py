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

def astrospikes(x_column, y_column, spike_value, mean_rows):
    if len(x_column) != len(y_column):
        raise ValueError("x_column and y_column must have the same length.")
    data_spectra = pd.DataFrame({'x': x_column, 'Intensity': y_column})
    data_spectra['median'] = data_spectra['Intensity'].rolling(window=mean_rows).median()
    data_spectra['Intensity'] = np.where(
        (data_spectra['Intensity'] - data_spectra['median']).abs() >= spike_value,
        np.nan,
        data_spectra['Intensity']
    )
    data_spectra['Intensity'] = data_spectra['Intensity'].interpolate(method='index')
    return data_spectra['Intensity'].values

def rescale_min_max(df):
  mmax=max(df.max())
  mmin=min(df.min())
  for column in df.columns:
    df.loc[:, column] = (df[column] - mmin) / (mmax - mmin)  
  return df

def plot_confidence_interval(df, label, color, plot=False):
    mean_line = df.mean(axis=0)
    ci = 1.96 * df.std(axis=0) / np.sqrt(len(df))  # 95% confidence interval
    plt.plot(mean_line, label=label, color=color)
    plt.fill_between(df.columns, mean_line - ci, mean_line + ci, color=color, alpha=0.2)
    if plot:
        plt.show()

plt.figure(figsize=(14, 4))

for idx, (key, value) in enumerate(bacteria_files.items()):
    df = pd.read_csv(value, quotechar='"', skiprows=2)
    
    x_column = df.iloc[:, 0].values
    y_columns = df.iloc[:, 1:]

    for col in y_columns.columns:
        y_columns.loc[:, col] = astrospikes(x_column, y_columns[col].values, spike_value=100, mean_rows=5)

    y_columns = rescale_min_max(y_columns)
    y_columns = y_columns.transpose().reset_index(drop=True)
    y_columns.columns = x_column
    y_columns = y_columns.drop(y_columns.index[0])

    y_columns = y_columns.groupby(y_columns.index // 50).mean()
    y_columns = y_columns[:10]
    plot_confidence_interval(y_columns, label=key, color=colors[idx])

plt.legend()
plt.xlabel('Sample Points')
plt.ylabel('Mean Intensity')
plt.title('Mean Intensity of Bacteria Samples with Confidence Intervals') 
plt.show()
