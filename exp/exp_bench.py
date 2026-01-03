import os
import torch
import warnings
import numpy as np
import torch.nn as nn

from torch import optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from exp.exp_basic import Exp_Basic
from models import AnomalyTransformer, BeatGAN, DCdetector, AMSL, MEMTO, THOC, DAGMM, OCSVM, IForest, MARPP
from data_provider.data_factory import data_provider

warnings.filterwarnings('ignore')

class Exp_Bench(Exp_Basic):
    def __init__(self, args):
        super(Exp_Bench, self).__init__(args)

    def _build_solver(self):
        model_dict = {
            'AnomalyTransformer': AnomalyTransformer,
            'BeatGAN': BeatGAN,
            'DCdetector': DCdetector,
            'AMSL': AMSL,
            'MEMTO': MEMTO,
            'THOC': THOC,
            'DAGMM': DAGMM,
            'OCSVM': OCSVM,
            'IForest': IForest,
            'MARPP': MARPP
        }

        solver = model_dict[self.args.model].Solver(self.args)
        return solver
    
    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def valid(self, valid_loader):
        return self.solver.valid(valid_loader)

    def train(self, setting):
        train_loader = self._get_data(flag="train")
        valid_loader = self._get_data(flag="valid")
        self.solver.fit(train_loader, valid_loader)

    def test(self, setting, test=False):
        test_loader, test_label = self._get_data(flag="test")

        scores = self.solver.decision_function(test_loader)
        scores = scores.ravel()

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

    def online_detection(self, setting, test=False):
        test_loader, test_label = self._get_data(flag="test")

        scores = self.solver.decision_function(test_loader)
        scores = scores.ravel()
        output = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores.reshape(-1, 1)).ravel()
        thresh = np.percentile(output, 100 - self.args.threshold)
        prediction = np.array((output >= thresh).astype(int))

        folder_path = './results/Online/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(os.path.join(folder_path, f"{setting}.npy"), prediction)
        np.save(os.path.join(folder_path, f"label.npy"), test_label)