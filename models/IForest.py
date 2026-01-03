import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

class Solver():
    def __init__(self, configs, n_estimators=100, max_samples='auto', contamination=0.06, random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1  # use all cores
        )
        self.checkpoints = configs.checkpoints
        self.model_name = configs.model
        self.training = configs.task == 'train'

    def flatten_data(self, dataloader):
        data_list = []
        for x, _ in dataloader:
            x_flat = x.numpy().reshape(x.shape[0], -1)  
            data_list.append(x_flat)
        return np.vstack(data_list)  # shape: (N, D)

    def load_model(self): 
        setting = self.model_name
        path = os.path.join(self.checkpoints, setting)
        last_model_path = os.path.join(path, 'checkpoint.joblib')
        self.model = joblib.load(last_model_path)

    def save_model(self):
        setting = self.model_name
        path = os.path.join(self.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        last_model_path = os.path.join(path, 'checkpoint.joblib')
        joblib.dump(self.model, last_model_path)

    def fit(self, train_loader, valid_loader):
        X_train = self.flatten_data(train_loader)
        self.model.fit(X_train)
        self.save_model()

    def decision_function(self, test_loader):
        if not self.training:
            self.load_model()
        X_test = self.flatten_data(test_loader)
        # IsolationForest.score_samples returns log-likelihood-like scores;
        # higher score => more "normal"
        # So anomaly score = -score_samples (higher => more anomalous)
        scores = -self.model.score_samples(X_test)
        return scores