import pytorch_lightning as pl
import torch
from torch import nn
from SpectrogramClassifier.utils import CutEnds, StandardScaler_nn, SNV, GaussianNoise, get_out_features
from torchmetrics.classification import MulticlassAccuracy
from torch.optim import Adam
from torcheval.metrics.functional import multiclass_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import SpectrogramClassifier.conf as conf
import csv

from copy import deepcopy
from SpectrogramClassifier.data import Data
import datetime

class NN(pl.LightningModule):
    def __init__(self,data=None,arc=None,lr=1e-6) -> nn.Sequential:
        super().__init__()
        if data:
            self.data = data
        else:
            self.data = Data()
        scale_np, mean_np = self.data.get_pipelines()
        self.lr=lr
        self.arc=arc
        self.best_model = (-1,None,None,None)
        self.n_classes=self.data.n_classes
        
        if self.arc==None:
            arc = nn.ModuleList()
            initialsL = nn.ModuleList()
            initialsL.append(CutEnds())
            initialsL.append(SNV())
            initialsL.append(StandardScaler_nn(torch.from_numpy(mean_np).float(),torch.from_numpy(scale_np).float()))
            initialsL.append(GaussianNoise(sigma = 0.01))
            out_features = get_out_features(initialsL, ((1, 1, conf.N_FEATURES)))
            arc.append(nn.Conv1d(in_channels = 1, out_channels = 16, kernel_size = 64,padding = 0))
            arc.append(nn.ELU())
            arc.append(nn.MaxPool1d(kernel_size=16, stride=2, padding=0, dilation=1, ceil_mode=False))
            arc.append(nn.Conv1d(in_channels = 16,out_channels = 32, kernel_size = 64,padding = 0))
            arc.append(nn.ELU())
            arc.append(nn.MaxPool1d(kernel_size=16, stride=2, padding=0, dilation=1, ceil_mode=False))

            out_features = get_out_features(arc, ((1, 1, out_features)))

            arc.append(nn.Flatten())
            arc.append(nn.Dropout(0.5))
            arc.append(nn.Linear(out_features,out_features//2))
            arc.append(nn.ELU())
            arc.append(nn.Linear(out_features//2,out_features//4))
            arc.append(nn.ELU())
            arc.append(nn.Linear(out_features//4, self.n_classes))
            
            arc.append(nn.Softmax(dim=0))
            self.arc = nn.Sequential(*initialsL, *arc)

        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.arc.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        z = z
        loss=0
    
        loss = nn.functional.cross_entropy(z, y)

        self.log("train_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x ,y = batch
        
        test_model = NN(data=self.data,arc=self.best_model[2])
        test_model.load_state_dict(self.best_model[1])
        test_model.training = False
        z=test_model.forward(x)
        
        accuracy = self.metric(z,y)
        self.accuracy_to_print = accuracy
        matrix = multiclass_confusion_matrix(z,y,self.n_classes)
        print(matrix)       
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix.cpu().numpy(),display_labels=self.data.le.inverse_transform(list(range(self.n_classes))))       
        disp.plot()
        plt.savefig(f"cms/confusion_matrix_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
        print("Accuracy of the NN: ",accuracy)
        self.log("Accuracy: ",accuracy)

        end_point = conf.START_POINT + conf.N_FEATURES

        with open("results/windows.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                conf.N_FEATURES, 
                563 + conf.START_POINT, 
                563 + end_point, 
                accuracy.item(),
                self.current_epoch,
                self.global_step
            ])

        preds = torch.argmax(z, dim=1)

        self.test_preds.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())

        report = classification_report(self.test_targets, self.test_preds, output_dict=True)
        accuracy = report["accuracy"]
        precision = report["macro avg"]["precision"]
        recall = report["macro avg"]["recall"]
        f1_score = report["macro avg"]["f1-score"]

        with open("results/metrics_results.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                accuracy,
                precision,
                recall,
                f1_score
            ])

        self.log("Final Accuracy", accuracy)
        self.log("Final Precision", precision)
        self.log("Final Recall", recall)
        self.log("Final F1-score", f1_score)
       
    def validation_step(self, batch, batch_idx):
        values =[]
        x, y = batch
        z = self.forward(x)
        z = z
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.metric = MulticlassAccuracy(num_classes=self.n_classes).to(device=device)
        val_loss = self.metric(z, y)

        if(val_loss > self.best_model[0] or self.best_model[0] < 0):
            self.best_model = (-1,None,None,None)
            self.best_model = (val_loss, deepcopy(self.state_dict()), self.arc, self.optimizer.state_dict())
            
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        self.optimizer = Adam(self.parameters(),lr=self.lr)
        return self.optimizer