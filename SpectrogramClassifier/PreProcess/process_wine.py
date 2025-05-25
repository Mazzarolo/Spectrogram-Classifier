import numpy as np
import pandas as pd
from SpectrogramClassifier.PreProcess.process_sers import ProcessSersData

class ProcessWineData(ProcessSersData):  
  def __init__(self, mean_rows=5):
    x_w1, y_w1 = ProcessWineData.process_file(r'wine/Clean_W1.csv', mean_rows=mean_rows)
    x_w2, y_w2 = ProcessWineData.process_file(r'wine/Clean_W2.csv', mean_rows=mean_rows)
    x_w3, y_w3 = ProcessWineData.process_file(r'wine/Clean_W3.csv', mean_rows=mean_rows)
    x_w4, y_w4 = ProcessWineData.process_file(r'wine/Clean_W4.csv', mean_rows=mean_rows)
    x_w5, y_w5 = ProcessWineData.process_file(r'wine/Clean_W5.csv', mean_rows=mean_rows)
    x_w6, y_w6 = ProcessWineData.process_file(r'wine/Clean_W6.csv', mean_rows=mean_rows)
    
    true_y_w1 = ['w1']*y_w1.T.values.shape[0]
    true_y_w2 = ['w2']*y_w2.T.values.shape[0]
    true_y_w3 = ['w3']*y_w3.T.values.shape[0]
    true_y_w4 = ['w4']*y_w4.T.values.shape[0]
    true_y_w5 = ['w5']*y_w5.T.values.shape[0]
    true_y_w6 = ['w6']*y_w6.T.values.shape[0]
    true_y_w1 = np.array(true_y_w1)
    true_y_w2 = np.array(true_y_w2)
    true_y_w3 = np.array(true_y_w3)
    true_y_w4 = np.array(true_y_w4)
    true_y_w5 = np.array(true_y_w5)
    true_y_w6 = np.array(true_y_w6)

    self.x_full_dataset = np.concatenate([y_w1.T.values,y_w2.T.values,y_w3.T.values,y_w4.T.values,y_w5.T.values,y_w6.T.values])
    self.y_full_dataset = np.concatenate([true_y_w1,true_y_w2,true_y_w3,true_y_w4,true_y_w5,true_y_w6])

  @staticmethod
  def process_file(filename, glass="nan", fluorescence="nan", rescaley=1, mean_rows=5):
    data = pd.read_csv(filename, sep='\t', header=None)
    return ProcessSersData.process_sers_file(data, glass=glass, fluorescence=fluorescence, rescaley=rescaley, mean_rows=mean_rows)