import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import tqdm
import numpy as np
import random
from random import choice
from scipy.signal import savgol_filter
from scipy.fft import dct

def bs_dct(signal):
    signal = dct(signal, type=2, norm='ortho', axis=1)
    return signal

def inter_data(hr, window=11):
    N = window
    time3 = savgol_filter(hr, window_length=N, polyorder=2)
    return time3

def noised(signal):
    SNR = 5
    noise = np.random.randn(signal.shape[0], signal.shape[1])
    noise = noise - np.mean(noise)
    signal_power = np.linalg.norm(signal) ** 2 / signal.size
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    signal_noise = noise + signal
    return signal_noise

def negated(signal):
    return signal * -1

def opposite_time(signal):
    return signal[::-1,:]

def permuted(signal, segment_length=32):
    signal_length = signal.shape[0]
    n_segments = signal_length // segment_length
    segment_indices = list(range(n_segments))
    random.shuffle(segment_indices)

    shuffled_signal = signal[segment_indices[0] * segment_length : (segment_indices[0]+1) * segment_length]
    for i in range(1,len(segment_indices)):
        segment = signal[segment_indices[i] * segment_length : (segment_indices[i]+1) * segment_length]
        shuffled_signal = np.vstack((shuffled_signal, segment))
    return shuffled_signal

def scale(signal):
    sc = [0.5, 2, 1.5, 0.8]
    s = choice(sc)
    return signal * s

def time_warp(signal):
    for i in range(signal.shape[1]):
        signal[:,i] = inter_data(signal[:,i],11)
    return signal

def transformation(x: torch.Tensor, device):
    dataX = x.detach().cpu().numpy()
    n = dataX.shape[0]

    transforms = [
        lambda a: a,          
        noised,
        negated,
        opposite_time,
        permuted,
        scale,
        time_warp
    ]

    augmented = []
    for fn in transforms:
        arr = np.empty_like(dataX)
        for i in range(n):
            arr[i] = fn(dataX[i].copy())
        augmented.append(arr)

    num_classes = len(transforms)
    labels = []
    for idx in range(num_classes):
        lab = np.zeros((n, num_classes), dtype=np.float32)
        lab[:, idx] = 1.0
        labels.append(lab)

    # 转回 tensor 并移到 device
    rt_x = [torch.from_numpy(arr).to(device) for arr in augmented]
    rt_labels = [torch.from_numpy(lab).to(device) for lab in labels]

    return rt_x, rt_labels

class MemoryModule(nn.Module):
    def __init__(self, mem_dim, n_mem):
        super(MemoryModule, self).__init__()
        self.mem_dim = mem_dim
        self.n_mem = n_mem
        self.weight = nn.Parameter(torch.FloatTensor(n_mem, mem_dim))
        self.std = 1. / math.sqrt(mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -self.std, self.std)

    def forward(self, x):
        batch_size, channel, x1_size, x2_size = x.shape
        x = x.permute(0,2,3,1).reshape(-1, channel)
        distance = torch.matmul(x, self.weight.t())
        att_weight = F.softmax(distance, dim=1)

        x = torch.matmul(att_weight, self.weight)
        x = x.reshape(batch_size, x1_size, x2_size, channel).permute(0,3,1,2)

        att_reshaped = att_weight.view(batch_size, -1)
        att_entropy_loss = -torch.sum(att_reshaped * torch.log(att_reshaped + 1e-12), dim=1, keepdim=True)

        return x, att_entropy_loss


class Encoder(nn.Module):
    def __init__(self, in_channel, hidden_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, hidden_dim, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d((2, 1), padding=0)
        self.conv2 = nn.Conv2d(hidden_dim, embedding_dim, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d((2, 1), padding=0)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        return x


class Decoder(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3, out_channel, embedding_dim):
        super(Decoder, self).__init__()
        self.convtrans1 = nn.ConvTranspose2d(embedding_dim, hidden_dim1, kernel_size=3, padding=1)
        self.convtrans2 = nn.ConvTranspose2d(hidden_dim1, hidden_dim2, kernel_size=(2, 1), stride=(2, 1))
        self.convtrans3 = nn.ConvTranspose2d(hidden_dim2, hidden_dim3, kernel_size=3, padding=1)
        self.convtrans4 = nn.ConvTranspose2d(hidden_dim3, out_channel, kernel_size=(2, 1), stride=(2, 1))

    def forward(self, x):
        x = F.relu(self.convtrans1(x))
        x = F.relu(self.convtrans2(x))
        x = F.relu(self.convtrans3(x))
        x = torch.sigmoid(self.convtrans4(x))
        x = x.squeeze(1)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv = nn.Conv2d(input_dim, 1, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class Fusion(nn.Module):
    def __init__(self, R):
        super(Fusion, self).__init__()
        self.linear = nn.Linear(1, 2 * R)
        self.batchnorm = nn.BatchNorm1d(2 * R)

    def forward(self, c):
        c = self.linear(c)
        c = self.batchnorm(c)
        c = torch.sigmoid(c)
        return c


class AMSLModel(nn.Module):
    def __init__(self,
                 feats,
                 n_mem,
                 R,
                 ):
        super(AMSLModel, self).__init__()
        self.R = R
        self.feats = feats
        self.n_mem = n_mem
        self.hidden_dim1 = 32
        self.embedding_dim = 64
        self.hidden_dim2 = 128

        self.argument_encoders = nn.ModuleList([])
        for i in range(self.R):
            self.argument_encoders.append(
                Encoder(self.feats, self.hidden_dim1, self.embedding_dim)
            )

        self.argument_decoders = nn.ModuleList([])
        for i in range(self.R):
            self.argument_decoders.append(
                Decoder(self.hidden_dim2, self.embedding_dim, self.hidden_dim1,
                        self.feats, 2 * self.embedding_dim)
            )

        self.classifier = Classifier(input_dim=self.embedding_dim, n_classes=self.R)

        self.global_memory = MemoryModule(mem_dim=self.embedding_dim, n_mem=self.n_mem)
        self.local_memories = nn.ModuleList([])
        for i in range(self.R):
            self.local_memories.append(
                MemoryModule(mem_dim=self.embedding_dim, n_mem=self.n_mem)
            )

        self.fusion = Fusion(R=self.R)

    def _argument_forward(self, modules, x):
        rt = None
        if isinstance(modules, list) or isinstance(modules, nn.ModuleList):
            if len(modules) != self.R:
                raise ValueError("Number of arguments must be equal to R")
            rt = [module(x[i]) for i, module in enumerate(modules)]
        elif isinstance(modules, nn.Module):
            rt = [modules(_x) for _x in x]

        if rt and isinstance(rt[0], tuple):
            unzipped_rt = []
            for i in range(len(rt[0])):
                elements = [item[i] for item in rt]
                unzipped_rt.append(elements)
            rt = unzipped_rt
        return rt

    def forward(self, x, c):
        if len(x) != self.R:
            raise ValueError("Number of arguments must be equal to R")

        x_encoded = self._argument_forward(self.argument_encoders, x)
        x_pred = self._argument_forward(self.classifier, x_encoded)
        x_global_memory, x_global_att_entropy_loss = self._argument_forward(self.global_memory, x_encoded)
        x_local_memory, x_local_att_entropy_loss = self._argument_forward(self.local_memories, x_encoded)
        fusion_att_weight = self.fusion(c)

        z = []
        for i, (memory_global, memory_local) in enumerate(zip(x_global_memory, x_local_memory)):
            z_global = memory_global * fusion_att_weight[:, 2*i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            z_local = memory_local * fusion_att_weight[:, 2*i+1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            z.append(torch.cat([x_encoded[i], z_global+z_local], dim=1))

        x_reconstructed = self._argument_forward(self.argument_decoders, z)

        return x_reconstructed, x_pred, x_global_att_entropy_loss, x_local_att_entropy_loss


class AMSLLoss(nn.Module):
    def __init__(self, lambd, gamma):
        super(AMSLLoss, self).__init__()
        self.lambd = lambd
        self.gamma = gamma

    def forward(self, x, x_rec, pred, gt, x_global_att_weight, x_local_att_weight):
        B = x[0].shape[0]

        device = x[0].device
        reconstruction_loss_per_sample = torch.zeros(B, device=device)
        classification_loss_per_sample = torch.zeros(B, device=device)
        sparse_loss_per_sample = torch.zeros(B, device=device)

        for x_rec_i, x_i in zip(x_rec, x):
            mse_elementwise = F.mse_loss(x_rec_i, x_i, reduction='none')
            per_sample_mse = mse_elementwise.view(B, -1).mean(dim=1)
            reconstruction_loss_per_sample += per_sample_mse

        for pred_i, gt_i in zip(pred, gt):
            if gt_i.dim() > 1 and gt_i.shape[1] > 1:
                labels = torch.argmax(gt_i, dim=1)
            else:

                labels = gt_i.view(-1).long()
            class_loss_per_sample = F.cross_entropy(pred_i, labels, reduction='none')
            classification_loss_per_sample += class_loss_per_sample

        for g_w, l_w in zip(x_global_att_weight, x_local_att_weight):
            per_sample_g = g_w.view(B, -1).mean(dim=1)
            per_sample_l = l_w.view(B, -1).mean(dim=1)
            sparse_loss_per_sample += (per_sample_g + per_sample_l)

        per_sample_total_loss = reconstruction_loss_per_sample + self.lambd * classification_loss_per_sample + self.gamma * sparse_loss_per_sample

        return per_sample_total_loss


class Solver():
    def __init__(self, configs):
        super().__init__()

        self.__anomaly_score = None

        self.lambd = 0.1
        self.gamma = 1
        self.R = 7
        self.n_mem = 500

        self.win_size = configs.win_size
        self.batch_size = configs.batch_size
        self.epochs = configs.epochs
        self.feats = configs.enc_in
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = AMSLModel(feats=self.feats, n_mem=self.n_mem, R=self.R).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=configs.learning_rate, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)
        self.criterion = AMSLLoss(self.lambd, self.gamma)

    def fit(self, train_loader, valid_loader):
    
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for idx, (d, _) in loop:
                d = d.to(self.device)
                c = torch.ones((d.size(0), 1)).to(self.device)
                d_argument, gt = transformation(d, self.device)

                d_rec, d_pred, x_global_att_entropy_loss, x_local_att_entropy_loss = self.model(d_argument, c)
                loss = self.criterion(d_argument, d_rec, d_pred, gt, x_global_att_entropy_loss, x_local_att_entropy_loss).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.cpu().item()
                loop.set_description(f"Training Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

            if len(valid_loader) > 0:
                self.model.eval()
                avg_loss_val = 0
                loop = tqdm.tqdm(
                    enumerate(valid_loader), total=len(valid_loader), leave=True
                )
                with torch.no_grad():
                    for idx, (d, _) in loop:
                        d = d.to(self.device)
                        c = torch.ones((d.size(0), 1)).to(self.device)
                        d_argument, gt = transformation(d, self.device)
                        d_rec, d_pred, x_global_att_entropy_loss, x_local_att_entropy_loss = self.model(d_argument,
                                                                                                        c)
                        loss = self.criterion(d_argument, d_rec, d_pred, gt, x_global_att_entropy_loss,
                                                x_local_att_entropy_loss).mean()

                        avg_loss_val += loss.cpu().item()
                        loop.set_description(
                            f"Validation Epoch [{epoch}/{self.epochs}]"
                        )
                        loop.set_postfix(loss=loss.item(), avg_loss_val=avg_loss_val / (idx + 1))

            self.scheduler.step()

    def decision_function(self, test_loader):
        self.model.eval()
        scores = []
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)

        with torch.no_grad():
            for idx, (d, _) in loop:
                d = d.to(self.device)
                c = torch.ones((d.size(0), 1)).to(self.device)
                d_argument, gt = transformation(d, self.device)
                d_rec, d_pred, _, _ = self.model(d_argument, c)
                loss = self.criterion(d_argument, d_rec, d_pred, gt, [], [])
                scores.append(loss.cpu())
        
        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()

        self.__anomaly_score = scores
        return self.__anomaly_score