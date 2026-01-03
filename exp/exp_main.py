import os
import torch
import warnings
import numpy as np
import torch.nn as nn

from torch import optim
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from exp.exp_basic import Exp_Basic
from models import TFEnhanced, FEDformer, OmniAnomaly, Autoformer, Informer
from data_provider.data_factory import data_provider

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'TFEnhanced': TFEnhanced,
            'FEDformer': FEDformer,
            'OmniAnomaly': OmniAnomaly,
            'Autoformer': Autoformer,
            'Informer': Informer
        }

        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        return data_provider(self.args, flag)
    
    def _select_optimizer(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return optimizer
    
    def _select_scheduler(self, optimizer):
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.args.step_size, self.args.lr_decay)
        return scheduler

    def _select_criterion(self):
        criterion = nn.MSELoss(reduction = 'none')
        return criterion
    

    def valid(self, epoch, valid_loader):
        criterion = self._select_criterion()

        self.model.eval()
        avg_loss_val = 0
        loop = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True)
        with torch.no_grad():
            for i, (x, y) in loop:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)

                _, L, D = y.shape
                y = y.view(-1, L * D)
                y_pred = y_pred.view(-1, L * D)
                mse = torch.mean(criterion(y_pred, y), axis=-1)
                loss = torch.mean(mse) 
                avg_loss_val += loss.cpu().item()

                loop.set_description(
                    f"Validation Epoch [{epoch}/{self.args.epochs}]"
                )
                loop.set_postfix(loss=loss.item(), avg_loss_val=avg_loss_val / (i + 1))


    def train(self, setting):
        train_loader = self._get_data(flag="train")
        valid_loader = self._get_data(flag="valid")

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = self._select_scheduler(optimizer)

        for epoch in range(1, 1 + self.args.epochs):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)

            for i, (x, y) in loop:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)

                _, L, D = y.shape
                y = y.view(-1, L * D)
                y_pred = y_pred.view(-1, L * D)
                mse = torch.mean(criterion(y_pred, y), axis=-1)
                loss = torch.mean(mse)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss += loss.cpu().item()
                loop.set_description(f"Training Epoch [{epoch}/{self.args.epochs}]")
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (i + 1))

            scheduler.step()
            self.valid(epoch, valid_loader)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        last_model_path = path + '/' + 'checkpoint.pth'
        torch.save(self.model.state_dict(), last_model_path)

    def test(self, setting, test=False):
        test_loader, test_label = self._get_data(flag="test")
        criterion = self._select_criterion()

        if test:
            print('Loading model...')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth'), map_location=self.device))

        self.model.eval()
        scores = []
        loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)

        with torch.no_grad():
            for i, (x, y) in loop:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)

                _, L, D = y.shape
                y = y.view(-1, L * D)
                y_pred = y_pred.view(-1, L * D)
                loss = torch.mean(criterion(y_pred, y), axis=-1)
                scores.append(loss.cpu())
                loop.set_description("Testing")
                loop.set_postfix(loss=loss.cpu().mean().item())

        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()

        output = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores.reshape(-1, 1)).ravel()
        thresh = np.percentile(output, 100 - self.args.threshold)

        pred = np.array((output >= thresh).astype(int))
        gt = np.array(test_label)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        auc = roc_auc_score(gt, output)
        result_str = "Accuracy: {:0.4f}, AUC : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F1-score : {:0.4f}".format(accuracy, auc, precision, recall, f_score)
        print(result_str)  

        if not test:
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            with open(os.path.join(folder_path, "results.txt"), "a") as f:
                f.write(result_str + "\n")  

    def online_detection(self, setting):
        test_loader, test_label = self._get_data(flag="test")
        criterion = self._select_criterion()

        print('Loading model...')
        try:
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))
        except:
            print('No model found. Please train a model first.')

        self.model.eval()
        scores = []
        loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)

        with torch.no_grad():
            for i, (x, y) in loop:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)

                _, L, D = y.shape
                y = y.view(-1, L * D)
                y_pred = y_pred.view(-1, L * D)
                loss = torch.mean(criterion(y_pred, y), axis=-1)
                scores.append(loss.cpu())
        
        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()
        output = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores.reshape(-1, 1)).ravel()
        thresh = np.percentile(output, 100 - self.args.threshold)
        prediction = np.array((output >= thresh).astype(int))

        folder_path = './results/Online/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(os.path.join(folder_path, f"{setting}.npy"), prediction)
        np.save(os.path.join(folder_path, f"label.npy"), test_label)