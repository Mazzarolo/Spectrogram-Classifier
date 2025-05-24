import torch
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from data import Data
import pickle
import time
import sys
from logger import setup_logging, print_times
from process_wine import ProcessWineData
from process_bacteria import ProcessBacteriaData
from NN import NN
import conf
import argparse
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean", type=int, default=5)
    parser.add_argument("--n_features", type=int, default=0)
    parser.add_argument("--start_point", type=int, default=0)
    args = parser.parse_args()

    mean = args.mean
    conf.N_FEATURES = args.n_features
    conf.START_POINT = args.start_point

    data = Data(mean_rows=mean, processor_class=ProcessBacteriaData, perc_val=0.15, perc_test=0.05) 
    
    model = NN(data=data,lr=1e-6)
    
    seed_everything(42, workers=True)
    
    trainer = Trainer(max_epochs=10000, 
                      log_every_n_steps=10,
                      callbacks=[
                          EarlyStopping(monitor="val_loss",mode='max',patience=25,stopping_threshold=1)
                          ],
                      deterministic=True,enable_checkpointing=False)
    
    trainer.fit(model, datamodule=data)

    trainer.test(model, datamodule=data)

if __name__=='__main__':
    main()
