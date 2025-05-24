import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import conf
from scipy.signal import find_peaks

class Processed_wine_data:
  instance = None
  def __new__(cls, *args, **kwargs):
        # Define apenas uma instância independente de quantas vezes o construtor ser chamado
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance
  
  def __init__(self, mean_rows):
    x_w1, y_w1 = process_file(r'wine/Clean_W1.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_w2, y_w2 = process_file(r'wine/Clean_W2.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_w3, y_w3 = process_file(r'wine/Clean_W3.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_w4, y_w4 = process_file(r'wine/Clean_W4.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    
    #here i create the true y which are the labels for training the model, and the old y will work as the x for inputing in the model
    true_y_w1 = ['w1']*y_w1.T.values.shape[0]
    true_y_w2 = ['w2']*y_w2.T.values.shape[0]
    true_y_w3 = ['w3']*y_w3.T.values.shape[0]
    true_y_w4 = ['w4']*y_w4.T.values.shape[0]
    true_y_w1 = np.array(true_y_w1)
    true_y_w2 = np.array(true_y_w2)
    true_y_w3 = np.array(true_y_w3)
    true_y_w4 = np.array(true_y_w4)

    self.x_full_dataset = np.concatenate([y_w1.T.values,y_w2.T.values,y_w3.T.values,y_w4.T.values])
    self.y_full_dataset = np.concatenate([true_y_w1,true_y_w2,true_y_w3,true_y_w4])

def generate_samples(df, num):
  dic={}
  for i in range(num):
    dic[i] = df.sample(frac=0.1)
  return dic

def astrospikes(x_column,y_column,spike_value,mean_rows):
    if len(x_column) != len(y_column):
        raise ValueError("x_column and y_column must have the same length.")
    #data_spectra['z_score']=stats.zscore(data_spectra['Intensity']) #finding astronomical spikes
    data_spectra = pd.DataFrame({'x': x_column,'Intensity': y_column})
    data_spectra['median'] = data_spectra['Intensity'].rolling(window=mean_rows).median()
    data_spectra['Intensity'] = np.where((data_spectra['Intensity'] - data_spectra['median']).abs()>=spike_value, np.nan, data_spectra['Intensity'])
    data_spectra['Intensity'] = data_spectra['Intensity'].interpolate(method='index')
    return data_spectra['Intensity'].values


def read_and_remove_glass(x, y, fileglass="Glass_DI.txt"):
    datag = pd.DataFrame(np.loadtxt(fileglass))
    # Extract 'x' values and 'y' values
    x_valuesg = datag.iloc[:, 0]  # Assuming 'x' values are in the first column
    y_valuesg = datag.iloc[:, 1]  # Assuming 'y' values are in columns after the first
    y_valuesg = y_valuesg/y_valuesg.max()*y.max()
    y_valuesi, _ = remove_fluorescence(x_valuesg, y_valuesg)
    return pd.DataFrame(y - y_valuesi)

def read_and_remove_baseline(x, y, filebaseline):
    data = np.loadtxt(filebaseline)
    # Extract 'x' values and 'y' values
    x_values = data[:, 0]  # Assuming 'x' values are in the first column
    y_values = data[:, 1]  # Assuming 'y' values are in columns after the first
    return y - y_values


def remove_fluorescence(x, y):
    # Find indices of local minima in the wave displacements
    local_minima_indices, _ = find_peaks(-y,   distance = 90)
    # Fit a polynomial curve that passes through the given points
    #degree = len(local_minima_indices) - 1  # Degree of the polynomial
    degree=8
    start_point_index = 0
    end_point_index = len(x) - 1
    fit_indices = np.concatenate(([start_point_index], local_minima_indices, [end_point_index]))
    coefficients = np.polyfit(x[fit_indices], y[fit_indices], degree)
    #plt.plot(x, y, label='Original ')
    # Generate the curve using the fitted polynomial coefficients
    fitted_curve = np.polyval(coefficients, x)
    #plt.plot(x, fitted_curve, label='Fluorescence ')
    area_under_curve = np.trapz(fitted_curve, x)
    #miny = np.min(y - fitted_curve)
    return y - fitted_curve, area_under_curve

def rescale_min_max(df):
  mmax=max(df.max())
  mmin=min(df.min())
  for column in df.columns:
    df[column] = (df[column] - mmin) / (mmax - mmin)  
  return df

def process_file(filename, baseline="nan", glass="nan", fluorescence="nan", rescaley=1, mean_rows=5):
  data = pd.read_csv(filename, sep='\t', header=None)
  cut_value = conf.START_POINT
  data = data.iloc[cut_value:cut_value+conf.N_FEATURES, :].reset_index(drop=True)
  print(cut_value)

  x_values = data.iloc[:, 0] 
  y_values = data.iloc[:, 1:]
  num_y_columns = y_values.shape[1]
  fluo_int = np.zeros([y_values.shape[1]])

  for i in range(num_y_columns):
    y_values.iloc[:, i] = astrospikes(x_values,y_values.iloc[:, i],spike_value=100, mean_rows=mean_rows)    
    if (fluorescence != "nan"):
      y_values.iloc[:, i], fluo_int[i] = remove_fluorescence(x_values,y_values.iloc[:, i])
    if (glass != "nan"):
      y_values.iloc[:, i] = read_and_remove_glass(x_values,y_values.iloc[:, i])
  if rescaley:
    y_values = rescale_min_max(y_values)

  return x_values, y_values

