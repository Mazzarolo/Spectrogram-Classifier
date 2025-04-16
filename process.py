import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import conf
from scipy.signal import find_peaks

class Processed_data:
  instance = None
  def __new__(cls, *args, **kwargs):
        # Define apenas uma instância independente de quantas vezes o construtor ser chamado
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance
  
  def __init__(self, mean_rows):
    x_ecol1, y_ecol1 = process_file(r'bacteria/ECOL1.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_ecol2, y_ecol2 = process_file(r'bacteria/ECOL2.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_ecol3, y_ecol3 = process_file(r'bacteria/ECOL3.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_paeruginosa2, y_paeruginosa2 = process_file(r'bacteria/Paeruginosa2.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_salgalactiae1, y_salgalactiae1 = process_file(r'bacteria/Salgalactiae1.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_salgalactiae2, y_salgalactiae2 = process_file(r'bacteria/Salgalactiae2.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_salgalactiae3, y_salgalactiae3 = process_file(r'bacteria/Salgalactiae3.csv', baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    x_saureus2, y_saureus2 = process_file(r'bacteria/Saureus2.csv',baseline="nan", glass="nan", fluorescence="nan", mean_rows=mean_rows)
    #plt.figure(figsize=(14, 8))
    #plt.plot(y_ecol1.values[:,0], label='E. coli', linestyle='-')
    #plt.plot(y_paeruginosa2.values[:,0], label='P. aeruginosa', linestyle='-')
    #plt.plot(y_salgalactiae1.values[:,0], label='S. agalactiae', linestyle='-')
    #plt.plot(y_saureus2.values[:,0], label='S. aureus', linestyle='-')
    #plt.legend()
    #plt.xlabel('Sample Points')
    #plt.ylabel('Mean Intensity')
    #plt.title('Mean Intensity of Bacteria Samples')
    #plt.show()
    #exit(1)
    
    #here i create the true y which are the labels for training the model, and the old y will work as the x for inputing in the model
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
  data = pd.read_csv(filename, quotechar='"',  skiprows=2)
  # to get a window of dataset
  cut_value = conf.START_POINT
  data = data.iloc[cut_value:cut_value+conf.N_FEATURES, :].reset_index(drop=True)
  #print(conf.N_FEATURES)
  print(cut_value)
  #intervals_peak = [data.iloc[:330, :], data.iloc[560:800, :]]
  # peak size 570
  #intervals_non_peak = [data.iloc[330:560, :], data.iloc[800:, :]]
  # non peak size 445
  #data = pd.concat(intervals_peak, axis=0).reset_index(drop=True)
  #plt.figure(figsize=(10, 6))
  #plt.plot(data.iloc[:, 3])
  #plt.show()
  #print(data.shape[0])
  #exit(1)

  x_values = data.iloc[:, 0] 
  y_values = data.iloc[:, 1:]
  num_y_columns = y_values.shape[1]  # Number of 'y' columns
  fluo_int = np.zeros([y_values.shape[1]]) #fluorecence signal for each sample

  for i in range(num_y_columns):
    y_values.iloc[:, i] = astrospikes(x_values,y_values.iloc[:, i],spike_value=100, mean_rows=mean_rows)    
    if (fluorescence != "nan"):
      y_values.iloc[:, i], fluo_int[i] = remove_fluorescence(x_values,y_values.iloc[:, i])
    if (glass != "nan"):
      y_values.iloc[:, i] = read_and_remove_glass(x_values,y_values.iloc[:, i])
        #y_values[:, i] = remove_baseline_linear(x_values,y_values[:, i])
        #convert_to_image(x_values, y_values[:, i], fluo_int[i], f"{image_file_path}{'_'}{i}", N = 324)
  if rescaley:
    y_values = rescale_min_max(y_values)
    #plt.plot(x_values, y_values[:, i], label=f'Y{i+1}')

  #average_y = np.mean(y_values, axis=1)
  #TODO
  #if (baseline != "nan"):
  #  average_y = read_and_remove_baseline(x_values,average_y, baseline)

  return x_values, y_values
 
#TODO> y_ecol1.loc[:,'point 478':'point 1580']
  #C:\Users\Henrique\Desktop\Nira\nira-ml\data\bacteria\ECOL1.csv
# here it returns the wavenumber in x position and the intensity in the y, its a bit wrong
#np.array()

""" OLD
ecol1 = pd.read_csv('ECOL1.csv', quotechar='"',  skiprows=2)
ecol2 = pd.read_csv('ECOL2.csv', quotechar='"', skiprows=2)
ecol3 = pd.read_csv('ECOL3.csv', quotechar='"', skiprows=2)
paeruginosa2 = pd.read_csv('Paeruginosa2.csv', quotechar='"', skiprows=2)
salgalactiae1 = pd.read_csv('Salgalactiae1.csv', quotechar='"', skiprows=2)
salgalactiae2 = pd.read_csv('Salgalactiae2.csv', quotechar='"', skiprows=2)
salgalactiae3 = pd.read_csv('Salgalactiae3.csv', quotechar='"', skiprows=2)
saureus2 = pd.read_csv('Saureus2.csv', quotechar='"', skiprows=2)

dic_ecol1 = generate_samples(ecol1, 100)
for i in range(100):
  dic_ecol1[i].mean().plot()
plt.show()
"""

