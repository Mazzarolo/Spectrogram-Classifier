import torch
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from data import Data
import pickle
import time
import sys
from logger import setup_logging
from NN import NN
import conf
import csv

def main():
    mean = 5

    if len(sys.argv) > 1:
        mean = int(sys.argv[1])
    if len(sys.argv) > 2:
        conf.N_FEATURES = int(sys.argv[2])
    if len(sys.argv) > 3:
        conf.START_POINT = int(sys.argv[3])

    data = Data(mean_rows=mean) 
    
    model = NN(data=data,lr=1e-6)
    
    seed_everything(42, workers=True)
    
    trainer = Trainer(max_epochs=10000, 
                      log_every_n_steps=10,
                      callbacks=[
                          EarlyStopping(monitor="val_loss",mode='max',patience=25,stopping_threshold=1)
                          ],
                      deterministic=True,enable_checkpointing=False)
    
    start_train_time = time.time()
    trainer.fit(model, datamodule=data)
    end_train_time = time.time()
    train_time = end_train_time - start_train_time

    start_test_time = time.time()
    trainer.test(model, datamodule=data)
    end_test_time = time.time()
    test_time = end_test_time - start_test_time

    end_point = conf.START_POINT + conf.N_FEATURES

    with open("results/windows_times.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([conf.N_FEATURES, 563 + conf.START_POINT, 563 + end_point, train_time, test_time])

if __name__=='__main__':
    main()
