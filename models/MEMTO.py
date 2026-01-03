"""
This function is adapted from [MEMTO] by [Junho Song et al.]
Original source: [https://github.com/gunny97/MEMTO]
"""

from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from kmeans_pytorch import kmeans
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import os
import time
from layers.Embed import TokenEmbedding, PositionalEmbedding
from layers.Transformer_EncDec import EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def k_means_clustering(x,n_mem,d_model):
    start = time.time()

    x = x.view([-1,d_model])
    print('running K Means Clustering. It takes few minutes to find clusters')
    # sckit-learn xxxx (cuda problem)
    _, cluster_centers = kmeans(X=x, num_clusters=n_mem, distance='euclidean', device=torch.device('cuda:0'))
    print("time for conducting Kmeans Clustering :", time.time() - start)
    print('K means clustering is done!!!')

    return cluster_centers


class ContrastiveLoss(nn.Module):
    def __init__(self, temp_param, eps=1e-12, reduce=True):
        super(ContrastiveLoss, self).__init__()
        self.temp_param = temp_param
        self.eps = eps
        self.reduce = reduce

    def get_score(self, query, key):
        score = torch.matmul(query, torch.t(key))  
        score = F.softmax(score, dim=1)  

        return score

    def forward(self, queries, items):

        batch_size = queries.size(0)
        d_model = queries.size(-1)

        # margin from 1.0
        loss = torch.nn.TripletMarginLoss(margin=1.0, reduce=self.reduce)

        queries = queries.contiguous().view(-1, d_model)  # (NxL) x C >> T x C
        score = self.get_score(queries, items)  # TxM

        # gather indices of nearest and second nearest item
        _, indices = torch.topk(score, 2, dim=1)

        # 1st and 2nd nearest items (l2 normalized)
        pos = items[indices[:, 0]]  # TxC
        neg = items[indices[:, 1]]  # TxC
        anc = queries  # TxC

        spread_loss = loss(anc, pos, neg)

        if self.reduce:
            return spread_loss

        spread_loss = spread_loss.contiguous().view(batch_size, -1)  # N x L

        return spread_loss  # N x L


class GatheringLoss(nn.Module):
    def __init__(self, reduce=True):
        super(GatheringLoss, self).__init__()
        self.reduce = reduce

    def get_score(self, query, key):
        score = torch.matmul(query, torch.t(key))  # Fea x Mem^T : (TXC) X (CXM) = TxM
        score = F.softmax(score, dim=1)  # TxM

        return score

    def forward(self, queries, items):
        batch_size = queries.size(0)
        d_model = queries.size(-1)

        loss_mse = torch.nn.MSELoss(reduce=self.reduce)

        queries = queries.contiguous().view(-1, d_model)  # (NxL) x C >> T x C
        score = self.get_score(queries, items)  # TxM

        _, indices = torch.topk(score, 1, dim=1)

        gathering_loss = loss_mse(queries, items[indices].squeeze(1))

        if self.reduce:
            return gathering_loss

        gathering_loss = torch.sum(gathering_loss, dim=-1)  # T
        gathering_loss = gathering_loss.contiguous().view(batch_size, -1)  # N x L

        return gathering_loss


class EntropyLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        loss = -1 * x * torch.log(x + self.eps)
        loss = torch.sum(loss, dim=-1)
        loss = torch.mean(loss)
        return loss


class NearestSim(nn.Module):
    def __init__(self):
        super(NearestSim, self).__init__()

    def get_score(self, query, key):
        qs = query.size()
        ks = key.size()

        score = F.linear(query, key)  # Fea x Mem^T : (TXC) X (CXM) = TxM
        score = F.softmax(score, dim=1)  # TxM

        return score

    def forward(self, queries, items):
        batch_size = queries.size(0)
        d_model = queries.size(-1)

        queries = queries.contiguous().view(-1, d_model)  # (NxL) x C >> T x C
        score = self.get_score(queries, items)  # TxM

        # gather indices of nearest and second nearest item
        _, indices = torch.topk(score, 2, dim=1)

        # 1st and 2nd nearest items (l2 normalized)
        pos = F.normalize(items[indices[:, 0]], p=2, dim=-1)  # TxC
        anc = F.normalize(queries, p=2, dim=-1)  # TxC

        similarity = -1 * torch.sum(pos * anc, dim=-1)  # T
        similarity = similarity.contiguous().view(batch_size, -1)  # N x L

        return similarity  # N x L

class InputEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, device, dropout=0.0):
        super(InputEmbedding, self).__init__()
        self.device = device
        self.token_embedding = TokenEmbedding(c_in=in_dim, d_model=d_model)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.token_embedding(x) + self.pos_embedding(x)
        return self.dropout(x)


class MemoryModule(nn.Module):
    def __init__(self, n_memory, fea_dim, shrink_thres=0.0025, device=None, memory_init_embedding=None, phase_type=None,
                 dataset_name=None):
        super(MemoryModule, self).__init__()
        self.n_memory = n_memory
        self.fea_dim = fea_dim  # C(=d_model)
        self.shrink_thres = shrink_thres
        self.device = device
        self.phase_type = phase_type
        self.memory_init_embedding = memory_init_embedding

        self.U = nn.Linear(fea_dim, fea_dim)
        self.W = nn.Linear(fea_dim, fea_dim)

        if self.memory_init_embedding == None:
            if self.phase_type == 'test':
                raise NotImplementedError(f'Phase test can not be execate without memory_init_embedding!')
            # first train
            print('loading memory item with random initilzation (for first train phase)')
            self.mem = F.normalize(torch.rand((self.n_memory, self.fea_dim), dtype=torch.float), dim=1)
        else:
            # second train
            if self.phase_type == 'second_train':
                print('second training (for second train phase)')
                self.mem = memory_init_embedding
            elif self.phase_type == 'test':
                self.mem = memory_init_embedding
                print('loading memory item vectors trained from kmeans (for test phase)')
            else:
                raise NotImplementedError(f'Phase {self.phase_type} type not implemented')

    # relu based hard shrinkage function, only works for positive values
    def hard_shrink_relu(self, input, lambd=0.0025, epsilon=1e-12):
        output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)

        return output

    def get_attn_score(self, query, key):
        '''
        Calculating attention score with sparsity regularization
        query (initial features) : (NxL) x C or N x C -> T x C
        key (memory items): M x C
        '''
        attn = torch.matmul(query, torch.t(key))  # (TxC) x (CxM) -> TxM
        attn = F.softmax(attn, dim=-1)

        if (self.shrink_thres > 0):
            attn = self.hard_shrink_relu(attn, self.shrink_thres)
            # re-normalize
            attn = F.normalize(attn, p=1, dim=1)

        return attn

    def read(self, query):
        '''
        query (initial features) : (NxL) x C or N x C -> T x C
        read memory items and get new robust features,
        while memory items(cluster centers) being fixed
        '''
        attn = self.get_attn_score(query, self.mem.detach())  # T x M
        add_memory = torch.matmul(attn, self.mem.detach())  # T x C

        # add_memory = F.normalize(add_memory, dim=1)
        read_query = torch.cat((query, add_memory), dim=1)  # T x 2C

        return {'output': read_query, 'attn': attn}

    def update(self, query):
        '''
        Update memory items(cluster centers)
        Fix Encoder parameters (detach)
        query (encoder output features) : (NxL) x C or N x C -> T x C
        '''
        query = query.to(self.device)
        self.mem = self.mem.to(self.device)
        attn = self.get_attn_score(self.mem, query.detach())  # M x T
        add_mem = torch.matmul(attn, query.detach())  # M x C

        # update gate : M x C
        update_gate = torch.sigmoid(self.U(self.mem) + self.W(add_mem))  # M x C
        self.mem = (1 - update_gate) * self.mem + update_gate * add_mem
        # self.mem = F.noramlize(self.mem + add_mem, dim=1)   # M x C

    def forward(self, query):
        '''
        query (encoder output features) : N x L x C or N x C
        '''
        s = query.data.shape
        l = len(s)

        query = query.contiguous()
        query = query.view(-1, s[-1])  # N x L x C or N x C -> T x C

        # Normalized encoder output features
        # query = F.normalize(query, dim=1)

        # update memory items(cluster centers), while encoder parameters being fixed
        if self.phase_type != 'test':
            self.update(query)

        # get new robust features, while memory items(cluster centers) being fixed
        outs = self.read(query)

        read_query, attn = outs['output'], outs['attn']

        if l == 2:
            pass
        elif l == 3:
            read_query = read_query.view(s[0], s[1], 2 * s[2])
            attn = attn.view(s[0], s[1], self.n_memory)
        else:
            raise TypeError('Wrong input dimension')
        '''
        output : N x L x 2C or N x 2C
        attn : N x L x M or N x M
        '''
        return {'output': read_query, 'attn': attn, 'memory_init_embedding': self.mem}


# Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x, _ = attn_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, c_out, d_ff=None, activation='relu', dropout=0.1):
        super(Decoder, self).__init__()
        self.out_linear = nn.Linear(d_model, c_out)
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.decoder_layer1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.decoder_layer2 = nn.Conv1d(in_channels=d_ff, out_channels=c_out, kernel_size=1)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = nn.BatchNorm1d(d_ff)

    def forward(self, x):
        out = self.out_linear(x)
        return out  


class TransformerVar(nn.Module):
    def __init__(self, win_size, enc_in, c_out, n_memory, shrink_thres=0, \
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu', \
                 device=None, memory_init_embedding=None, memory_initial=False, phase_type=None, dataset_name=None):
        super(TransformerVar, self).__init__()

        self.memory_initial = memory_initial

        # Encoding
        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout,
                                        device=device)  # N x L x C(=d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=dropout), d_model, n_heads
                    ), d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.mem_module = MemoryModule(n_memory=n_memory, fea_dim=d_model, shrink_thres=shrink_thres, device=device,
                                       memory_init_embedding=memory_init_embedding, phase_type=phase_type,
                                       dataset_name=dataset_name)

        # ours
        self.weak_decoder = Decoder(2 * d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)


    def forward(self, x):
        x = self.embedding(x)  # embeddin : N x L x C(=d_model)
        queries = out = self.encoder(x)  # encoder out : N x L x C(=d_model)

        if self.memory_initial:
            outputs = self.mem_module(out)
            out = torch.cat((out, out), dim=-1)
            mem = self.mem_module.mem
            out = self.weak_decoder(out)
            return {"out": out, "memory_item_embedding": None, "queries": queries, "mem": mem}
        else:
            outputs = self.mem_module(out)
            out, attn, memory_item_embedding = outputs['output'], outputs['attn'], outputs['memory_init_embedding']
            mem = self.mem_module.mem
            out = self.weak_decoder(out)

            return {"out": out, "memory_item_embedding": memory_item_embedding, "queries": queries, "mem": mem,
                    "attn": attn}


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class Solver():
    def __init__(self,
                 win_size=256,
                 batch_size=128,
                 epochs=10,
                 lr=0.01,
                 input_c=1,
                 output_c=1,
                 e_layers=3,
                 d_model=512,
                 n_memory=128,
                 memory_initial=True,
                 memory_init_embedding=None,
                 ):
        super().__init__()

        self.lambd = 0.01
        self.temperature = 0.1
        self.batch_size = batch_size
        self.win_size = win_size
        self.lr = lr
        self.epochs = epochs
        self.input_c = input_c
        self.output_c = output_c
        self.e_layers = e_layers
        self.d_model = d_model
        self.n_memory = n_memory
        self.memory_initial = memory_initial
        self.memory_init_embedding = memory_init_embedding

        if self.memory_initial == "False":

            self.memory_initial = False
        else:
            self.memory_initial = True

        self.memory_init_embedding = None

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.entropy_loss = EntropyLoss()
        self.criterion = nn.MSELoss()

    def build_model(self, memory_init_embedding, phase_type):
        self.model = TransformerVar(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c,
                                    e_layers=3, d_model=self.d_model, n_memory=self.n_memory, device=self.device,
                                    memory_initial=self.memory_initial, memory_init_embedding=memory_init_embedding,
                                    phase_type=phase_type).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def valid(self, training_type, vali_loader):
        self.model.eval()

        valid_loss_list = []
        valid_re_loss_list = []
        valid_entropy_loss_list = []

        if training_type == 'second_train':
            loop = tqdm(
                enumerate(vali_loader), total=len(vali_loader), leave=True
            )
            for i, (input_data, _) in loop:
                input = input_data.float().to(self.device)
                output_dict = self.model(input)
                output, queries, mem_items, attn = output_dict['out'], output_dict['queries'], output_dict['mem'], \
                output_dict['attn']

                rec_loss = self.criterion(output, input)
                entropy_loss = self.entropy_loss(attn)
                loss = rec_loss + self.lambd * entropy_loss

                valid_re_loss_list.append(rec_loss.detach().cpu().numpy())
                valid_entropy_loss_list.append(entropy_loss.detach().cpu().numpy())
                valid_loss_list.append(loss.detach().cpu().numpy())
        else:
            loop = tqdm(
                enumerate(vali_loader), total=len(vali_loader), leave=True
            )
            for i, (input_data, _) in loop:
                input = input_data.float().to(self.device)
                output_dict = self.model(input)
                output, queries, mem_items = output_dict['out'], output_dict['queries'], output_dict['mem']

                rec_loss = self.criterion(output, input)
                loss = rec_loss

                valid_re_loss_list.append(rec_loss.detach().cpu().numpy())
                valid_entropy_loss_list.append(0)
                valid_loss_list.append(loss.detach().cpu().numpy())

        return np.average(valid_loss_list), np.average(valid_re_loss_list), np.average(valid_entropy_loss_list)

    def _train(self, training_type, train_loader, vali_loader):
        time_now = time.time()
        if training_type != 'second_train':
            train_steps = len(train_loader)

            for epoch in range(self.epochs):
                iter_count = 0
                loss_list = []
                rec_loss_list = []

                epoch_time = time.time()
                self.model.train()
                loop = tqdm(
                    enumerate(train_loader), total=len(train_loader), leave=True
                )
                for i, (input_data, _) in loop:

                    self.optimizer.zero_grad()
                    iter_count += 1
                    input_data = input_data.float().to(self.device)
                    output_dict = self.model(input_data)

                    output, memory_item_embedding, queries, mem_items = output_dict['out'], output_dict[
                        'memory_item_embedding'], output_dict['queries'], output_dict["mem"]

                    rec_loss = self.criterion(output, input_data)
                    loss = rec_loss

                    loss_list.append(loss.detach().cpu().numpy())
                    rec_loss_list.append(rec_loss.detach().cpu().numpy())

                    if (i + 1) % 100 == 0:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    loss.mean().backward()
                    self.optimizer.step()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

                train_loss = np.average(loss_list)
                train_rec_loss = np.average(rec_loss_list)
                valid_loss, valid_re_loss_list, valid_entropy_loss_list = self.valid(training_type, vali_loader)

                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                        epoch + 1, train_steps, train_loss, valid_loss))
                print(
                    "Epoch: {0}, Steps: {1} | VALID reconstruction Loss: {3:.7f} Entropy loss Loss: {2:.7f}  ".format(
                        epoch + 1, train_steps, valid_re_loss_list, valid_entropy_loss_list))
        else:
            train_steps = len(train_loader)
            for epoch in range(self.epochs):
                iter_count = 0
                loss_list = []
                rec_loss_list = []
                entropy_loss_list = []

                epoch_time = time.time()
                self.model.train()
                loop = tqdm(
                    enumerate(train_loader), total=len(train_loader), leave=True
                )
                for i, (input_data, _) in loop:

                    self.optimizer.zero_grad()
                    iter_count += 1
                    input_data = input_data.float().to(self.device)
                    output_dict = self.model(input_data)

                    output, memory_item_embedding, queries, mem_items, attn = output_dict['out'], output_dict[
                        'memory_item_embedding'], output_dict['queries'], output_dict["mem"], output_dict['attn']

                    rec_loss = self.criterion(output, input_data)
                    entropy_loss = self.entropy_loss(attn)
                    loss = rec_loss + self.lambd * entropy_loss

                    loss_list.append(loss.detach().cpu().numpy())
                    entropy_loss_list.append(entropy_loss.detach().cpu().numpy())
                    rec_loss_list.append(rec_loss.detach().cpu().numpy())

                    if (i + 1) % 100 == 0:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    loss.mean().backward()
                    self.optimizer.step()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

                train_loss = np.average(loss_list)
                train_entropy_loss = np.average(entropy_loss_list)
                train_rec_loss = np.average(rec_loss_list)
                valid_loss, valid_re_loss_list, valid_entropy_loss_list = self.vali(training_type, vali_loader)

                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                        epoch + 1, train_steps, train_loss, valid_loss))
                print(
                    "Epoch: {0}, Steps: {1} | VALID reconstruction Loss: {3:.7f} Entropy loss Loss: {2:.7f}  ".format(
                        epoch + 1, train_steps, valid_re_loss_list, valid_entropy_loss_list))
                print(
                    "Epoch: {0}, Steps: {1} | TRAIN reconstruction Loss: {3:.7f} Entropy loss Loss: {2:.7f}  ".format(
                        epoch + 1, train_steps, train_rec_loss, train_entropy_loss))

        return memory_item_embedding
    def fit(self, train_loader, valid_loader):
        dataset = train_loader.dataset  
        num_total = len(dataset)
        num_samples = max(1, int(0.01 * num_total))  
        indices = torch.randperm(num_total).tolist()[:num_samples]
        subset = Subset(dataset, indices)
        k_loader = DataLoader(
            subset,
            batch_size=train_loader.batch_size,
            shuffle=False,  
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
        )

        print('========First Train========')
        train_type = None
        self.build_model(self.memory_init_embedding, train_type)
        self._train(train_type, train_loader, valid_loader)

        print('========Second Train========')
        train_type = 'second_train'
        self.model.eval()

        for i, (input_data, _) in enumerate(k_loader):
            input_data = input_data.float().to(self.device)
            if i == 0:
                output = self.model(input_data)['queries']
            else:
                output = torch.cat([output, self.model(input_data)['queries']], dim=0)

        self.memory_init_embedding = k_means_clustering(x=output, n_mem=self.n_memory, d_model=self.d_model)
        self.memory_initial = False
        self.build_model(memory_init_embedding=self.memory_init_embedding.detach(), phase_type=train_type)
        memory_item_embedding = self._train(train_type, train_loader, valid_loader)
        memory_item_embedding = memory_item_embedding[:int(self.n_memory), :]

    def decision_function(self, test_loader):
        print("======================TEST MODE======================")
        self.model.eval()
        criterion = nn.MSELoss(reduce=False)
        gathering_loss = GatheringLoss(reduce=False)
        temperature = self.temperature

        test_attens_energy = []
        loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        with torch.no_grad():
            for i, (input_data, _) in loop:
                input_data = input_data.float().to(self.device)
                output_dict = self.model(input_data)
                output, queries, mem_items = output_dict['out'], output_dict['queries'], output_dict['mem']

                rec_loss = torch.mean(criterion(input_data, output), dim=-1)
                latent_score = torch.softmax(gathering_loss(queries, mem_items) / temperature, dim=-1)
                loss = latent_score * rec_loss

                cri = loss.detach().cpu().numpy().sum(axis=-1)
                test_attens_energy.append(cri)

            test_attens_energy = np.concatenate(test_attens_energy, axis=0).reshape(-1)
            test_energy = np.array(test_attens_energy)
            scores = np.array(test_energy)

        assert scores.ndim == 1

        self.__anomaly_score = scores
        return self.__anomaly_score
