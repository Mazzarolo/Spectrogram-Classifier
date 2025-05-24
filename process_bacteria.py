import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import conf
from scipy.signal import find_peaks

class ProcessBacteriaData:
  instance = None
  def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance
  
  def __init__(self, mean_rows=5):
    x_ecol1, y_ecol1 = process_file(r'bacteria/ECOL1.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_ecol2, y_ecol2 = process_file(r'bacteria/ECOL2.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_ecol3, y_ecol3 = process_file(r'bacteria/ECOL3.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_paeruginosa2, y_paeruginosa2 = process_file(r'bacteria/Paeruginosa2.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_salgalactiae1, y_salgalactiae1 = process_file(r'bacteria/Salgalactiae1.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_salgalactiae2, y_salgalactiae2 = process_file(r'bacteria/Salgalactiae2.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_salgalactiae3, y_salgalactiae3 = process_file(r'bacteria/Salgalactiae3.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_saureus2, y_saureus2 = process_file(r'bacteria/Saureus2.csv',baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)

    true_y_ecol1 = ['ecol']*y_ecol1.T.values.shape[0]
    true_y_ecol2 = ['ecol']*y_ecol2.T.values.shape[0]
    true_y_ecol3 = ['ecol']*y_ecol3.T.values.shape[0]
    true_y_pae = ['pae']*y_paeruginosa2.T.values.shape[0]
    true_y_sal1 = ['sal']*y_salgalactiae1.T.values.shape[0]
    true_y_sal2 = ['sal']*y_salgalactiae2.T.values.shape[0]
    true_y_sal3 = ['sal']*y_salgalactiae3.T.values.shape[0]
    true_y_sau = ['sau']*y_saureus2.T.values.shape[0]
    true_y_ecol1 = np.array(true_y_ecol1)
    true_y_ecol2 = np.array(true_y_ecol2)
    true_y_ecol3 = np.array(true_y_ecol3)
    true_y_pae = np.array(true_y_pae)
    true_y_sal1 = np.array(true_y_sal1)
    true_y_sal2 = np.array(true_y_sal2)
    true_y_sal3 = np.array(true_y_sal3)
    true_y_sau = np.array(true_y_sau)
    
    self.x_extra_test = np.concatenate([y_ecol3.T.values,y_salgalactiae3.T.values])
    self.y_extra_test = np.concatenate([true_y_ecol3,true_y_sal3])

    self.x_full_dataset = np.concatenate([y_ecol1.T.values,y_ecol2.T.values,y_salgalactiae1.T.values,y_salgalactiae2.T.values,y_paeruginosa2.T.values,y_saureus2.T.values])
    self.y_full_dataset = np.concatenate([true_y_ecol1,true_y_ecol2,true_y_sal1,true_y_sal2,true_y_pae,true_y_sau])

def astrospikes(x_column,y_column,spike_value,mean_rows):
    if len(x_column) != len(y_column):
        raise ValueError("x_column and y_column must have the same length.")
    data_spectra = pd.DataFrame({'x': x_column,'Intensity': y_column})
    data_spectra['median'] = data_spectra['Intensity'].rolling(window=mean_rows).median()
    data_spectra['Intensity'] = np.where((data_spectra['Intensity'] - data_spectra['median']).abs()>=spike_value, np.nan, data_spectra['Intensity'])
    data_spectra['Intensity'] = data_spectra['Intensity'].interpolate(method='index')
    return data_spectra['Intensity'].values

def read_and_remove_glass(x, y, fileglass="Glass_DI.txt"):
    datag = pd.DataFrame(np.loadtxt(fileglass))
    x_valuesg = datag.iloc[:, 0]
    y_valuesg = datag.iloc[:, 1]
    y_valuesg = y_valuesg/y_valuesg.max()*y.max()
    y_valuesi, _ = remove_fluorescence(x_valuesg, y_valuesg)
    return pd.DataFrame(y - y_valuesi)

def read_and_remove_baseline(x, y, filebaseline):
    data = np.loadtxt(filebaseline)
    x_values = data[:, 0]
    y_values = data[:, 1]
    return y - y_values

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

def rescale_min_max(df):
  mmax=max(df.max())
  mmin=min(df.min())
  for column in df.columns:
    df[column] = (df[column] - mmin) / (mmax - mmin)  
  return df

def process_file(filename, baseline="nan", glass="nan", fluorescence="nan", rescaley=1, mean_rows=5):
  data = pd.read_csv(filename, quotechar='"',  skiprows=2)
  if conf.N_FEATURES == 0:
    conf.N_FEATURES = data.shape[0]
  cut_value = conf.START_POINT
  data = data.iloc[cut_value:cut_value+conf.N_FEATURES, :].reset_index(drop=True)

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