import os 
import joblib
import numpy as np
from sklearn.svm import OneClassSVM

class Solver():
    def __init__(self, configs, kernel='rbf', nu=0.06, gamma='scale'):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.checkpoints = configs.checkpoints
        self.model_name = configs.model
        self.training = configs.task == 'train'

    def flatten_data(self, dataloader):
        data_list = []
        for x, _ in dataloader:
            x_flat = x.numpy().reshape(x.shape[0], -1)  
            data_list.append(x_flat)

        return np.vstack(data_list)  # shape: (N_train, 256)

    def load_model(self): 
        setting = self.model_name
        path = os.path.join(self.checkpoints, setting)
        last_model_path = path + '/' + 'checkpoint.joblib'
        self.model = joblib.load(last_model_path)

    def save_model(self):
        setting = self.model_name
        path = os.path.join(self.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        last_model_path = path + '/' + 'checkpoint.joblib'
        joblib.dump(self.model, last_model_path)

    def fit(self, train_loader, valid_loader):
        X_train = self.flatten_data(train_loader)
        self.model.fit(X_train)
        self.save_model()

    def decision_function(self, test_loader):
        if not self.training: 
            self.load_model()
        X_test = self.flatten_data(test_loader)
        return -self.model.decision_function(X_test)