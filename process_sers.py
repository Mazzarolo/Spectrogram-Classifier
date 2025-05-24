import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import conf
from scipy.signal import find_peaks
from process_abstract import AbstractProcessData

class ProcessSersData(AbstractProcessData):
  @staticmethod
  def astrospikes(x_column,y_column,spike_value,mean_rows):
    if len(x_column) != len(y_column):
        raise ValueError("x_column and y_column must have the same length.")
    data_spectra = pd.DataFrame({'x': x_column,'Intensity': y_column})
    data_spectra['median'] = data_spectra['Intensity'].rolling(window=mean_rows).median()
    data_spectra['Intensity'] = np.where((data_spectra['Intensity'] - data_spectra['median']).abs()>=spike_value, np.nan, data_spectra['Intensity'])
    data_spectra['Intensity'] = data_spectra['Intensity'].interpolate(method='index')
    return data_spectra['Intensity'].values

  @staticmethod
  def read_and_remove_glass(x, y, fileglass="Glass_DI.txt"):
    datag = pd.DataFrame(np.loadtxt(fileglass))
    x_valuesg = datag.iloc[:, 0]
    y_valuesg = datag.iloc[:, 1]
    y_valuesg = y_valuesg/y_valuesg.max()*y.max()
    y_valuesi, _ = ProcessSersData.remove_fluorescence(x_valuesg, y_valuesg)
    return pd.DataFrame(y - y_valuesi)

  @staticmethod
  def read_and_remove_baseline(x, y, filebaseline):
    data = np.loadtxt(filebaseline)
    x_values = data[:, 0]
    y_values = data[:, 1]
    return y - y_values

  @staticmethod
  def remove_fluorescence(x, y):
    local_minima_indices, _ = find_peaks(-y,   distance = 90)
    degree=8
    start_point_index = 0
    end_point_index = len(x) - 1
    fit_indices = np.concatenate(([start_point_index], local_minima_indices, [end_point_index]))
    coefficients = np.polyfit(x[fit_indices], y[fit_indices], degree)
    fitted_curve = np.polyval(coefficients, x)
    area_under_curve = np.trapz(fitted_curve, x)
    return y - fitted_curve, area_under_curve

  @staticmethod
  def rescale_min_max(df):
    mmax=max(df.max())
    mmin=min(df.min())
    for column in df.columns:
      df[column] = (df[column] - mmin) / (mmax - mmin)  
    return df
  
  @staticmethod
  def process_sers_file(data, glass="nan", fluorescence="nan", rescaley=1, mean_rows=5):
    if conf.N_FEATURES == 0:
      conf.N_FEATURES = data.shape[0]
    cut_value = conf.START_POINT
    data = data.iloc[cut_value:cut_value+conf.N_FEATURES, :].reset_index(drop=True)

    x_values = data.iloc[:, 0] 
    y_values = data.iloc[:, 1:]
    num_y_columns = y_values.shape[1]
    fluo_int = np.zeros([y_values.shape[1]])

    for i in range(num_y_columns):
      y_values.iloc[:, i] = ProcessSersData.astrospikes(x_values,y_values.iloc[:, i],spike_value=100, mean_rows=mean_rows)    
      if (fluorescence != "nan"):
        y_values.iloc[:, i], fluo_int[i] = ProcessSersData.remove_fluorescence(x_values,y_values.iloc[:, i])
      if (glass != "nan"):
        y_values.iloc[:, i] = ProcessSersData.read_and_remove_glass(x_values,y_values.iloc[:, i])
    if rescaley:
      y_values = ProcessSersData.rescale_min_max(y_values)
    
    return x_values, y_values