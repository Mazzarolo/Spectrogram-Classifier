import torch
from torch import nn
import numpy as np

from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union
import pandas as pd
import sklearn.base as skbase
from sklearn.preprocessing import StandardScaler
import SpectrogramClassifier.conf as conf

class SNV(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        mean = torch.mean(x,dim=-1).unsqueeze(dim=1)
        std = torch.std(x,dim=-1,unbiased=False).unsqueeze(dim=1)
        return ((x - mean) / std)

class StandardScaler_nn(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.mean = torch.nn.Parameter(mean, requires_grad=False)
        self.scale =  torch.nn.Parameter(scale, requires_grad=False)
    
    def forward(self, x: torch.tensor):
        y = (x - self.mean.to(device=x.device)) / self.scale.to(device=x.device)
        return y
    
class CutEnds(nn.Module):
    def __init__(self,begin=0,end=conf.CUT_NOISE):
        super().__init__()
        self.begin = begin
        self.end = end
    def forward(self,x):
        if self.end == 0:
            return x[:, :, self.begin:]
        return x[:,:,self.begin:-self.end]
    
    
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training==True and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(x.size()[0],1,conf.N_FEATURES-conf.CUT_NOISE).float().normal_() * scale
            x = x + sampled_noise
        return x
    
def get_out_features(model: nn.ModuleList | nn.Sequential, input_shape: tuple[int, ...]) -> int:
    """
    Função que retorna as features de saída de um modelo para a construção de camadas densas
    args:
        model: ModuleList ou Sequential do PyTorch
        input: Shape de dados esperados
    return:
        out_feature: Tamanho da saída da última camada após passar por todo o modelo
    """
    if type(model) == nn.ModuleList:
        # Passa um input aleatório em todas as camadas
        size = model[0](torch.rand(*(input_shape))).data.shape
        for i, module in enumerate(model):
                if (i==0):
                    continue
                size = module(torch.rand(*(size))).data.shape

        out_features = np.prod(list(size))
        return out_features
    elif type(model) == nn.Sequential:
        # Passa um input aleatório pelo modelo
        size = model(torch.rand(*(input_shape))).data.shape

        out_features = np.prod(list(size))
        return out_features
    else:
        raise TypeError("Essa função só suporta os tipos ModuleList ou Sequential")
    
class Preprocessor(skbase.BaseEstimator, skbase.TransformerMixin):
    P = TypeVar('P', bound='Preprocessor')

    def transform(self: P, x: pd.DataFrame) -> np.ndarray: pass

    def fit(self: P, x: pd.DataFrame, y: pd.DataFrame = None) -> P: pass
    

class pp_SNV(Preprocessor):
    
    def __init__(self,ignore:bool = False):
        super().__init__()
        self.ignore = ignore

    def transform(self, x: np.ndarray) -> pd.DataFrame:
        if self.ignore : 
            return x
        input_data = x
        output_data = np.zeros_like(x)
        for i in range(input_data.shape[0]):
            output_data[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) \
                                / np.std(input_data[i, :])
        out = x.copy()
        out[:] = output_data
        return out

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> 'SNV':
        return self

class pp_StandardScaling(Preprocessor):
    def __init__(self, 
                scale_=None,
                mean_=None, 
                var_=None):
        super().__init__()
        self.scale_ = scale_
        self.mean_ = mean_
        self.var_ = var_
    def transform(self,X):
        return self.pipe.transform(X)
    def fit(self,X,y=None):
        self.pipe = StandardScaler().fit(X)
        scale = self.pipe.scale_
        mean = self.pipe.mean_
        var = self.pipe.var_
        self.scale_ = scale
        self.mean_ = mean
        self.var_ = var
        return self