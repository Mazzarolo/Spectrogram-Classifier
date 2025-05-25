import numpy as np
import pandas as pd
from SpectrogramClassifier.PreProcess.process_sers import ProcessSersData

class ProcessBacteriaData(ProcessSersData):
  def __init__(self, mean_rows=5):
    x_ecol1, y_ecol1 = ProcessBacteriaData.process_file(r'bacteria/ECOL1.csv', mean_rows=mean_rows)
    x_ecol2, y_ecol2 = ProcessBacteriaData.process_file(r'bacteria/ECOL2.csv', mean_rows=mean_rows)
    x_ecol3, y_ecol3 = ProcessBacteriaData.process_file(r'bacteria/ECOL3.csv', mean_rows=mean_rows)
    x_paeruginosa2, y_paeruginosa2 = ProcessBacteriaData.process_file(r'bacteria/Paeruginosa2.csv', mean_rows=mean_rows)
    x_salgalactiae1, y_salgalactiae1 = ProcessBacteriaData.process_file(r'bacteria/Salgalactiae1.csv', mean_rows=mean_rows)
    x_salgalactiae2, y_salgalactiae2 = ProcessBacteriaData.process_file(r'bacteria/Salgalactiae2.csv', mean_rows=mean_rows)
    x_salgalactiae3, y_salgalactiae3 = ProcessBacteriaData.process_file(r'bacteria/Salgalactiae3.csv', mean_rows=mean_rows)
    x_saureus2, y_saureus2 = ProcessBacteriaData.process_file(r'bacteria/Saureus2.csv', mean_rows=mean_rows)

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

  @staticmethod
  def process_file(filename, glass="nan", fluorescence="nan", rescaley=1, mean_rows=5):
    data = pd.read_csv(filename, quotechar='"', skiprows=2)
    return ProcessSersData.process_sers_file(data, glass=glass, fluorescence=fluorescence, rescaley=rescaley, mean_rows=mean_rows)