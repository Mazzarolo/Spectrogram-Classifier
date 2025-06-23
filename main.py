from SpectrogramClassifier.data import Data
from SpectrogramClassifier.PreProcess.process_wine import ProcessWineData
from SpectrogramClassifier.PreProcess.process_bacteria import ProcessBacteriaData
from SpectrogramClassifier.model import ClassifierModel
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean", type=int, default=5)
    args = parser.parse_args()
    
    data = Data(ProcessWineData, mean_rows=args.mean, perc_val=0.1, perc_test=0.1, folder='WineAgSmall') 
    
    model = ClassifierModel(data)
    model.fit()
    model.train()

if __name__=='__main__':
    main()