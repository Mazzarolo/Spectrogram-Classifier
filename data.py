import traceback
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import numpy as np
import pandas as pd
import os
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from pytorch_lightning import LightningDataModule
from copy import deepcopy
from process_bacteria import ProcessBacteriaData
from utils import SNV, StandardScaler_nn, pp_SNV, pp_StandardScaling
from sklearn import preprocessing
import conf

class Data(LightningDataModule):
    
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, 
                 batch_size = 256, 
                 v_batch_size = 128,
                 use_augmentation = False,
                 processor_class = ProcessBacteriaData,
                 perc_val = 0.15,
                 perc_test = 0.05,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.v_batch_size = v_batch_size
        self.use_augmentation = use_augmentation
        self.processor_class = processor_class
        self.perc_val = perc_val
        self.perc_test = perc_test
        self.extra_args = kwargs
        self.setup()
        
    def setup(self, stage: str = None):
        if stage == None:
            if not hasattr(self, "mean_np"):
                self.get_data()
                self.split()
                
                if self.use_augmentation==True:
                    self.augmentation_process()    

                self.finished_dataset()
                
    def get_data(self) -> None:
        self.processor = self.processor_class(**self.extra_args)
        
        X = self.processor.x_full_dataset
        Y = self.processor.y_full_dataset

        self.n_classes = len(np.unique(Y))
    
        self.X_clean = X
        self.Y_clean = Y
            
    def split(self) -> None:
        perc_val_test = self.perc_val + self.perc_test
        self.xcal, self.xval, self.ycal, self.yval = train_test_split(self.X_clean,self.Y_clean,test_size=perc_val_test,random_state=24)
        self.xval, self.xtest, self.yval, self.ytest = train_test_split(self.xval,self.yval,test_size=self.perc_test/perc_val_test,random_state=24)
       
    def augmentation_process(self) -> None:
        augmented_len = int(len(self.xcal) * 0.25)
        
        self.xcal = self.xcal
        self.ycal = self.ycal
        
    def finished_dataset(self)  -> None:
        pipeline = make_pipeline(pp_SNV(),pp_StandardScaling())
        pipeline.fit(self.xcal[:,:conf.N_FEATURES-conf.CUT_NOISE],self.ycal)
        if hasattr(self.processor, 'x_extra_test') and self.processor.x_extra_test is not None:
            self.xtest = np.concatenate([self.xtest, self.processor.x_extra_test])

        if hasattr(self.processor, 'y_extra_test') and self.processor.y_extra_test is not None:
            self.ytest = np.concatenate([self.ytest, self.processor.y_extra_test])

        torch.set_printoptions(precision=10)
        self.xcal = np.expand_dims(self.xcal, axis=-1)
              
        self.xval = np.expand_dims(self.xval, axis=-1)
        self.xtest = np.expand_dims(self.xtest, axis=-1)

        self.xcal = np.rollaxis(self.xcal, 2, 1)
        self.xval = np.rollaxis(self.xval, 2, 1)
        self.xtest = np.rollaxis(self.xtest, 2, 1)
        
        self.le = preprocessing.LabelEncoder()

        self.X_train = torch.from_numpy(self.xcal).float()
        self.Y_train = torch.as_tensor(self.le.fit_transform(self.ycal)).long()
        self.X_val = torch.from_numpy(self.xval).float()
        self.Y_val = torch.as_tensor(self.le.fit_transform(self.yval)).long()
        self.X_test = torch.from_numpy(self.xtest).float()
        self.Y_test = torch.as_tensor(self.le.fit_transform(self.ytest)).long()

        self.train_dataset = TensorDataset(self.X_train,self.Y_train)
        self.val_dataset = TensorDataset(self.X_val,self.Y_val)
        self.test_dataset = TensorDataset(self.X_test,self.Y_test)

        self.scale_np = np.expand_dims(np.expand_dims(pipeline[1].scale_,-1),-1).T
        self.mean_np = np.expand_dims(np.expand_dims(pipeline[1].mean_,-1),-1).T
        
    def get_X_Y_train(self) -> np.ndarray:
        return self.X_aug,self.Y_aug
        
    def get_X_Y_val(self) -> np.ndarray:
        return self.xval,self.yval
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,batch_size=self.batch_size, shuffle=True, num_workers=2, persistent_workers=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,batch_size=self.v_batch_size,num_workers=2, persistent_workers=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,batch_size=len(self.test_dataset),num_workers=5,persistent_workers=True)
    
    def get_dataloader_test(self) -> DataLoader:
        return self.test_dataloader
    
    def get_pipelines(self) -> np.ndarray:
        return (self.scale_np,self.mean_np)