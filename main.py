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
    
    logger = setup_logging()

    if len(sys.argv) < 2:
        logger.error("Por favor, forneça o valor de 'mean' como argumento.")
        sys.exit(1)
    
    mean = int(sys.argv[1])
    conf.N_FEATURES = int(sys.argv[2])
    conf.START_POINT = int(sys.argv[3])

    print(mean)

    data = Data(mean_rows=mean) 
    
    model = NN(data=data,lr=1e-6)
    
    seed_everything(42, workers=True)
    
    #model.load_state_dict(state_dict)
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
    logger.info(f"Tempo de treinamento para média de {mean} linhas: {train_time:.2f} segundos")

    start_test_time = time.time()
    trainer.test(model, datamodule=data)
    end_test_time = time.time()
    test_time = end_test_time - start_test_time
    logger.info(f"Tempo de teste para média de {mean} linhas: {test_time:.2f} segundos")
    
    logger.info("Características do modelo:")
    logger.info(str(model))
    logger.info(f"Accuracy: {model.accuracy_to_print}")

    end_point = conf.START_POINT + conf.N_FEATURES

    with open("windows_times.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([conf.N_FEATURES, 563 + conf.START_POINT, 563 + end_point, train_time, test_time])

if __name__=='__main__':
    main()
