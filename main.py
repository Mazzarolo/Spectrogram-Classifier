from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from SpectrogramClassifier.data import Data
from SpectrogramClassifier.PreProcess.process_wine import ProcessWineData
from SpectrogramClassifier.PreProcess.process_bacteria import ProcessBacteriaData
from SpectrogramClassifier.NN import NN
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean", type=int, default=5)
    args = parser.parse_args()

    mean = args.mean

    data = Data(ProcessWineData, mean_rows=mean, perc_val=0.1, perc_test=0.1) 
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
