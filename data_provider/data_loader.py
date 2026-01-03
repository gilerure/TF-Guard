import os
import numpy as np
from sklearn.model_selection import train_test_split


def getFoldK(data, fold, label, K = 5):
    normal_cnt = data.shape[0]
    fold_num = int(normal_cnt / K)
    fold_idx = fold * fold_num

    fold_data = data[fold_idx:fold_idx + fold_num]

    remain_data = np.concatenate([data[:fold_idx], data[fold_idx + fold_num:]])
    if label == 0:
        fold_data_y = np.zeros((fold_data.shape[0],), dtype=np.float32)
        remain_data_y = np.zeros((remain_data.shape[0],), dtype=np.float32)
    elif label == 1:
        fold_data_y = np.ones((fold_data.shape[0],), dtype=np.float32)
        remain_data_y = np.ones((remain_data.shape[0],), dtype=np.float32)
    else:
        raise Exception("label should be 0 or 1, get:{}".format(label))
    return fold_data, fold_data_y, remain_data, remain_data_y

def getPercent(data_x, data_y, percent, seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=percent, random_state=seed)
    return train_x, test_x, train_y, test_y

class SPRSegLoader(object):
    def __init__(self, data_path, mode="train"):
        self.mode = mode

        N_samples = np.load(os.path.join(data_path, "normal_samples.npy")) # shape: (27110, 1, 256)
        AN_samples = np.load(os.path.join(data_path, "abnormal_samples.npy")) # shape: (400, 1, 256)
        # print(N_samples.shape, AN_samples.shape)
        
        rng = np.random.default_rng(seed=42) 
        rng.shuffle(N_samples) 
        rng.shuffle(AN_samples)

        N_samples = N_samples.transpose(0, 2, 1) # shape: (N, L, 1)
        AN_samples = AN_samples.transpose(0, 2, 1)

        test_N, test_N_y, train_N, train_N_y = getFoldK(N_samples, 0, 0) # K-fold 前 1/5 用于测试
        test_AN, test_AN_y = AN_samples, np.ones((AN_samples.shape[0],), dtype=np.float32)
        train_N, val_N, train_N_y, val_N_y = getPercent(train_N, train_N_y, 0.1, 0) # 取 10% 作为验证集, train_N shape: (19519, 256, 1)
        test_AN, val_AN, test_AN_y, val_AN_y = getPercent(test_AN, test_AN_y, 0.1, 0)

        val_data = val_N
        val_y = val_N_y

        thre_data = np.concatenate((val_N, val_AN))
        thre_y = np.concatenate((val_N_y, val_AN_y))

        test_data = np.concatenate([test_N, test_AN, val_AN]) # test_N shape: (5422, 256, 1), test_AN shape: (360, 256, 1)
        test_y = np.concatenate([test_N_y, test_AN_y, val_AN_y])

        self.train, self.train_label = train_N, train_N_y
        self.val, self.val_label = val_data, val_y
        self.thre, self.thre_label = thre_data, thre_y
        self.test, self.test_label = test_data, test_y

        # print("\n----- Show Data Size-----")
        # print("train data size:{}".format(train_N.shape), flush=True) # (19519, 256, 1)
        # print("val data size:{}".format(val_data.shape), flush=True) # (2169, 256, 1)
        # print("thre data size:{}".format(thre_data.shape), flush=True) # (2209, 256, 1)
        # print("test N data size:{}".format(test_data.shape), flush=True) # (5822, 256, 1)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return self.train.shape[0]
        elif (self.mode == 'val'):
            return self.val.shape[0]
        elif (self.mode == 'test'):
            return self.test.shape[0]
        else:
            return self.thre.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_label[index])
        elif (self.mode == 'val'):
            return np.float32(self.val[index]), np.float32(self.val_label[index])
        elif (self.mode == 'test'):
            return np.float32(self.test[index]), np.float32(self.test_label[index])
        else:
            return np.float32(self.thre[index]), np.float32(self.thre_label[index])
        