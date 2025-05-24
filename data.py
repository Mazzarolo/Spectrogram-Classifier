import traceback
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
# import unidecode
import numpy as np
import pandas as pd
# import git
import os
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from pytorch_lightning import LightningDataModule
from copy import deepcopy
from process import Processed_data
from process_wine import Processed_wine_data
from utils import SNV, StandardScaler_nn, pp_SNV, pp_StandardScaling
from sklearn import preprocessing
import conf

class Data(LightningDataModule):
    
    _instance = None

    def __new__(cls, *args, **kwargs):
        # Define apenas uma instância independente de quantas vezes o construtor ser chamado
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    def __init__(self, 
                 batch_size = 256, 
                 v_batch_size = 128,
                 use_augmentation = False,
                 mean_rows = 5):
        super().__init__()
        self.batch_size = batch_size
        self.v_batch_size = v_batch_size
        self.use_augmentation = use_augmentation
        self.mean_rows = mean_rows
        self.setup()
        
    def setup(self, stage: str = None):
        if stage == None:
            # Checa se ele ja executou o setup em algum momento
            if not hasattr(self, "mean_np"):
                self.get_data()
                self.split()
                
                if self.use_augmentation==True:
                    self.augmentation_process()    

                self.finished_dataset()
                
                
    def get_data(self) -> None:
        #git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
        #git_root = git_repo.git.rev_parse("--show-toplevel")
        print(self.mean_rows)
        self.processor = Processed_wine_data(mean_rows=self.mean_rows)
        
        X = self.processor.x_full_dataset
        Y = self.processor.y_full_dataset
    
        self.X_clean = X
        self.Y_clean = Y
            


    def split(self) -> None:
        self.xcal, self.xval, self.ycal, self.yval = train_test_split(self.X_clean,self.Y_clean,test_size=0.2,random_state=24)
        self.xval, self.xtest, self.yval, self.ytest = train_test_split(self.xval,self.yval,test_size=0.25,random_state=24)
       

    def augmentation_process(self) -> None:
        augmented_len = int(len(self.xcal) * 0.25)
        # Seleciona um subset dos dados
        #shift = np.std(self.xcal)*0.25
        #X_new = deepcopy(self.xcal[0:augmented_len,:])
        #Y_new = deepcopy(self.ycal[0:augmented_len,:])
        #X_new = ChemUtils.dataaugment(X_new,betashift = shift, slopeshift = shift*0.2, multishift = shift)
        #self.X_aug = np.vstack([self.xcal, X_new])
        #self.Y_aug = np.vstack([self.ycal, Y_new])
        
        self.xcal = self.xcal
        self.ycal = self.ycal
        

        #self.X_aug = np.repeat(self.xcal, repeats=20, axis=0)
        #self.X_aug = ChemUtils.dataaugment(self.X_aug,betashift = shift, slopeshift = shift*0.6, multishift = shift)

        #self.Y_aug = np.repeat(self.ycal, repeats=20, axis=0) #y_train is simply repeated
        
    def finished_dataset(self)  -> None:
        pipeline = make_pipeline(pp_SNV(),pp_StandardScaling())
        pipeline.fit(self.xcal[:,:conf.N_FEATURES-conf.CUT_NOISE],self.ycal)
        #self.xtest = np.concatenate([self.xtest,self.processor.x_extra_test])       # pega os dados extra
        #self.ytest = np.concatenate([self.ytest,self.processor.y_extra_test])
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