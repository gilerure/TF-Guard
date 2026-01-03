import torch, random
import torch.utils.data
import numpy as np
from torch.utils.data import DataLoader
from data_provider.data_loader import SPRSegLoader

# seed
seed = 2025
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class ReconstructDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, stride=256, normalize=False):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max(0, (self.data.shape[0] - window_size) // stride + 1)
        self.samples, self.targets = self._generate_samples() # (19519, 256, 1)

    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        data = torch.tensor(self.data, dtype=torch.float32)

        if self.univariate:
            data = data.squeeze()
            X = torch.stack([data[i * self.stride : i * self.stride + self.window_size] for i in range(self.sample_num)]) # shape: (sample_num, window_size)
            X = X.unsqueeze(-1)
        else:
            X = torch.stack([data[i * self.stride : i * self.stride + self.window_size, :] for i in range(self.sample_num)])

        return X, X

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]


def data_provider(args, flag):
    data = SPRSegLoader(args.root_path)
    data_train = data.train.reshape(-1, data.train.shape[-1]) # (19519, 256, 1) -> (4996864, 1)
    data_val = data.val.reshape(-1, data.val.shape[-1]) # (2169, 256, 1) -> (555264, 1)
    data_test = data.test.reshape(-1, data.test.shape[-1]) # (5822, 256, 1) -> (1490432, 1)

    if flag == "train":
        return DataLoader(
        dataset=ReconstructDataset(data_train, window_size=args.win_size, stride=args.win_size),
        batch_size=args.batch_size,
        shuffle=True
    )
    elif flag == 'valid':
        return DataLoader(
        dataset=ReconstructDataset(data_val, window_size=args.win_size, stride=args.win_size),
        batch_size=args.batch_size,
        shuffle=False
    )        
    else:
        return DataLoader(
        dataset=ReconstructDataset(data_test, window_size=args.win_size, stride=args.win_size),
        batch_size=args.batch_size,
        shuffle=False
    ), data.test_label
