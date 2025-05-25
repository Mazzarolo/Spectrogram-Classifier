from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from SpectrogramClassifier.NN import NN

class ClassifierModel():  
    def __init__(self, 
                data, 
                lr=1e-6, 
                max_epochs=10000, 
                log_every_n_steps=10, 
                deterministic=True, 
                enable_checkpointing=False,
                callbacks=[EarlyStopping(monitor="val_loss",mode='max',patience=25,stopping_threshold=1)]):
        self.data = data
        self.model = NN(data=data,lr=lr)
        seed_everything(42, workers=True)   
        self.trainer = Trainer(max_epochs=max_epochs, log_every_n_steps=log_every_n_steps, callbacks=callbacks, deterministic=deterministic, enable_checkpointing=enable_checkpointing) 

    def fit(self):
        self.trainer.fit(self.model, datamodule=self.data)

    def train(self):
        self.trainer.test(self.model, datamodule=self.data)