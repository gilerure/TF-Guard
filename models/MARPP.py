import numpy as np
import torch, math
from torch import nn
import tqdm, os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from layers.Embed import DataEmbedding

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = torch.nn.Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres= shrink_thres

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        att_weight = F.linear(x, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres>0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        return {'output': output, 'att': att_weight}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, x):
        y_and = self.memory(x)
        #
        y = y_and['output']
        att = y_and['att']
        return y, att

class InnerAutoencoder(nn.Module):
    def __init__(self,
                 n_features,
                 hidden_neurons=(128, 64),
                 dropout_rate=0.2,
                 n_mem=3,
                 hidden_dim=256):

        # initialize the super class
        super(InnerAutoencoder, self).__init__()

        # save the default values
        self.n_features = n_features
        self.dropout_rate = dropout_rate

        self.hidden_dim = hidden_dim
        self.enc_embedding = DataEmbedding(self.n_features, self.hidden_dim)
        self.lstm = nn.GRU(self.hidden_dim, self.hidden_dim, 2)

        # create the dimensions for the input and hidden layers
        self.layers_neurons_encoder_ = [self.n_features, *hidden_neurons]
        self.layers_neurons_decoder_ = self.layers_neurons_encoder_[::-1]
        self.layers_neurons_encoder_[0] = self.hidden_dim

        # get the object for the activations functions
        self.activation = nn.ReLU()

        # initialize encoder and decoder as a sequential
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        # fill the encoder sequential with hidden layers
        for idx, layer in enumerate(self.layers_neurons_encoder_[:-1]):

            # create a linear layer of neurons
            self.encoder.add_module(
                "linear" + str(idx),
                torch.nn.Linear(layer, self.layers_neurons_encoder_[idx + 1]))

            # add a batch norm per layer (leave out first layer)
            self.encoder.add_module("batch_norm" + str(idx),
                                    nn.BatchNorm1d(self.layers_neurons_encoder_[idx + 1]))

            # create the activation
            self.encoder.add_module("relu" + str(idx),
                                    self.activation)

            # create a dropout layer
            self.encoder.add_module("dropout" + str(idx),
                                    torch.nn.Dropout(dropout_rate))

        # fill the decoder layer
        for idx, layer in enumerate(self.layers_neurons_decoder_[:-1]):

            # create a linear layer of neurons
            self.decoder.add_module(
                "linear" + str(idx),
                torch.nn.Linear(layer,self.layers_neurons_decoder_[idx + 1]))

            # create a batch norm per layer if wanted (only if it is not the
            # last layer)
            if idx < len(self.layers_neurons_decoder_[:-1]) - 1:
                self.decoder.add_module("batch_norm" + str(idx),
                                        nn.BatchNorm1d(self.layers_neurons_decoder_[idx + 1]))

            # create the activation
            self.decoder.add_module("relu" + str(idx),
                                    self.activation)

            # create a dropout layer (only if it is not the last layer)
            if idx < len(self.layers_neurons_decoder_[:-1]) - 1:
                self.decoder.add_module("dropout" + str(idx),
                                        torch.nn.Dropout(dropout_rate))
        self.memory = MemModule(n_mem, self.layers_neurons_encoder_[-1])

    def forward(self, x, hidden = None):
        # we could return the latent representation here after the encoder
        # as the latent representation
        x = self.enc_embedding(x)
        B, L, D = x.shape
        x_ = x.contiguous().view(-1, B, D)
        hidden = torch.rand(2, B, self.hidden_dim).to(self.device) if hidden is not None else hidden
        out, hidden = self.lstm(x_, hidden)
        x = (out + x_).view(B * L, D)

        z = self.encoder(x)
        z_, att = self.memory(z)
        x = self.decoder(z_).view(L, B, self.n_features).permute(1, 0, 2)
        return {
                'x': x,
                'z': z,
                'z_': z_,
                'att': att
            }
    

class Solver():
    def __init__(self, configs,
                 hidden_neurons=None,
                 learning_rate=1e-3,
                 dropout_rate=0):
        super(Solver, self).__init__()

        self.n_mem = 500

        # save the initialization values
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.epochs = configs.epochs
        self.dropout_rate = dropout_rate
        self.weight_decay = configs.weight_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = torch.nn.MSELoss(reduction = 'none')
        self.checkpoints = configs.checkpoints
        self.model_name = configs.model
        self.training = configs.task == 'train'

        # default values for the amount of hidden neurons
        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32]

        # initialize the model
        self.model = InnerAutoencoder(
            n_features=configs.enc_in,
            hidden_neurons=self.hidden_neurons,
            dropout_rate=self.dropout_rate,
            n_mem = self.n_mem,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay)

    def load_model(self): 
        setting = self.model_name
        path = os.path.join(self.checkpoints, setting)
        last_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(last_model_path))

    def save_model(self):
        setting = self.model_name
        path = os.path.join(self.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        last_model_path = path + '/' + 'checkpoint.pth'
        torch.save(self.model.state_dict(), last_model_path)


    def fit(self, train_loader, valid_loader):
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for idx, (data, _) in loop:
                data = data.to(self.device)
                rt = self.model(data)
                preds, att = rt['x'], rt['att']
                loss = torch.mean(self.loss_fn(data, preds)) + torch.mean(-att * torch.log(att + 1e-12))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.cpu().item()
                loop.set_description(f"Training Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

            self.model.eval()
            avg_loss_val = 0
            loop = tqdm.tqdm(
                enumerate(valid_loader), total=len(valid_loader), leave=True
            )
            with torch.no_grad():
                for idx, (data, _) in loop:
                    data = data.to(self.device)
                    rt = self.model(data)
                    preds, att = rt['x'], rt['att']
                    loss = torch.mean(self.loss_fn(data, preds)) + torch.mean(-att * torch.log(att + 1e-12))
                    avg_loss_val += loss.cpu().item()
                    loop.set_description(f"Validation Epoch [{epoch}/{self.epochs}]")
                    loop.set_postfix(loss=loss.item(), avg_loss_val=avg_loss_val / (idx + 1))
        self.save_model()


    def decision_function(self, test_loader):
        if not self.training: 
            self.load_model()
        self.model.eval()
        scores = []
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)

        with torch.no_grad():
            for idx, (data, _) in loop:
                data = data.to(self.device)
                rt = self.model(data)
                preds, att = rt['x'], rt['att']

                _, L, D = data.shape
                data = data.reshape(-1, L * D)
                preds = preds.reshape(-1, L * D)
                loss = torch.mean(self.loss_fn(data, preds), axis=-1)
                scores.append(loss.cpu())

        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()
        self.__anomaly_score = scores
        return self.__anomaly_score
    