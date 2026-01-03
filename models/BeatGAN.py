import os,pickle
import numpy as np
import torch
import torch.nn as nn
import time,os,sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import tqdm


def weights_init(mod):
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(mod.weight.data)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find('Linear') !=-1 :
        torch.nn.init.xavier_uniform(mod.weight)
        mod.bias.data.fill_(0.01)


class Encoder(nn.Module):
    def __init__(self, nc,ndf,out_z,ngpu=1):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 256
            nn.Conv1d(nc,ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 16
            nn.Conv1d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 8

            nn.Conv1d(ndf * 16, out_z, 8, 1, 0, bias=False),
            # state size. (nz) x 1
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


class Decoder(nn.Module):
    def __init__(self,nc,nz,ngf,ngpu=1):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.main=nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(nz,ngf*16,8,1,0,bias=False),
            nn.BatchNorm1d(ngf*16),
            nn.ReLU(True),
            # state size. (ngf*16) x10
            nn.ConvTranspose1d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 20
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 40
            nn.ConvTranspose1d(ngf * 4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf) x 80
            nn.ConvTranspose1d(ngf * 2, ngf , 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf ),
            nn.ReLU(True),
            # state size. (ngf) x 160
            nn.ConvTranspose1d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 320
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


def normal(array,min_val,max_val):
    return (array-min_val)/(max_val-min_val)

class Discriminator(nn.Module):
    def __init__(self,nc,ndf):
        super(Discriminator, self).__init__()
        model = Encoder(nc,ndf,1,1)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        x = x.permute(0,2,1)
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features


class Generator(nn.Module):

    def __init__(self,nc,nz,ndf,ngf):
        super(Generator, self).__init__()
        self.encoder1 = Encoder(nc,ndf,nz,1)
        self.decoder = Decoder(nc,nz,ngf,1)

    def forward(self, x):
        x = x.permute(0,2,1)
        latent_i = self.encoder1(x)
        gen_x = self.decoder(latent_i)
        gen_x = gen_x.permute(0,2,1)
        return gen_x, latent_i


class Solver():
    def __init__(self, configs,
                 feats=1,
                 learning_rate=1e-4,
                 beta=0.4,
                 ):
        super(Solver, self).__init__()
        self.w_adv = 1
        self.nc = feats
        self.nz = configs.d_model
        self.ndf = configs.hidden_dim
        self.ngf = configs.hidden_dim
        self.epochs = configs.epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.learning_rate = learning_rate
        self.beta = beta

        self.G = Generator(self.nc,self.nz,self.ndf,self.ngf).to(self.device)
        self.G.apply(weights_init)

        self.D = Discriminator(self.nc,self.ndf).to(self.device)
        self.D.apply(weights_init)


        self.bce_criterion = nn.BCELoss()
        self.mse_criterion = nn.MSELoss()

        self.optimizerD = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.beta, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.beta, 0.999))

        self.real_label = 1
        self.fake_label= 0

        self.out_d_real = None
        self.feat_real = None

        self.fake = None
        self.latent_i = None
        self.out_d_fake = None
        self.feat_fake = None

    def fit(self, train_loader, valid_loader):
        for epoch in range(1, self.epochs + 1):
            self.G.train()
            self.D.train()
            avg_G_loss = 0
            avg_D_loss = 0

            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for idx, (d, _) in loop:
                d = d.to(self.device)
                err_d_real, err_d_fake, err_d = self.update_netd(d)
                err_g_adv, err_g_rec, err_g = self.update_netg(d)
                # If D loss too low, then re-initialize netD
                if err_d.item() < 5e-6:
                    self.reinitialize_netd()

                avg_D_loss += err_d.cpu().item()
                avg_G_loss += err_g_adv.cpu().item()
                loop.set_description(f"Training Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(D_loss=err_d.item(), G_loss=err_g.item(), avg_D_loss=avg_D_loss / (idx + 1), avg_G_loss=avg_G_loss / (idx + 1))

            if len(valid_loader) > 0:
                self.D.eval()
                self.G.eval()
                avg_G_loss_val = 0
                avg_D_loss_val = 0
                loop = tqdm.tqdm(
                    enumerate(valid_loader), total=len(valid_loader), leave=True
                )
                with torch.no_grad():
                    for idx, (d, _) in loop:
                        d = d.to(self.device)
                        err_d_real, err_d_fake, err_d = self.update_netd(d, optimize=False)
                        err_g_adv, err_g_rec, err_g = self.update_netg(d, optimize=False)
                        # If D loss too low, then re-initialize netD
                        if err_d.item() < 5e-6:
                            self.reinitialize_netd()

                        avg_D_loss_val += err_d.cpu().item()
                        avg_G_loss_val += err_g_adv.cpu().item()
                        loop.set_description(f"Validation Epoch [{epoch}/{self.epochs}]")
                        loop.set_postfix(D_loss=err_d.item(), G_loss=err_g.item(),
                                            avg_D_loss_val=avg_D_loss_val / (idx + 1), avg_G_loss_val=avg_G_loss_val / (idx + 1))


    def decision_function(self, test_loader):
        self.D.eval()
        self.G.eval()
        scores = []
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)

        with torch.no_grad():
            for idx, (x, _) in loop:
                x = x.float().to(self.device)
                fake, _ = self.G(x)
                error = torch.mean(
                    torch.pow((x.view(x.shape[0], -1) - fake.view(fake.shape[0], -1)), 2),
                    dim=1)
                scores.append(error.cpu())

        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()
        self.__anomaly_score = scores
        return self.__anomaly_score

    def update_netd(self, x, optimize=True):
        iter_batch_size = x.shape[0]
        self.D.zero_grad()
        # --
        # Train with real
        out_d_real, _ = self.D(x)
        # --
        # Train with fake
        fake, latent_i = self.G(x)
        out_d_fake, _ = self.D(fake)

        err_d_real = self.bce_criterion(out_d_real, torch.full((iter_batch_size,), self.real_label, device=self.device).float())
        err_d_fake = self.bce_criterion(out_d_fake, torch.full((iter_batch_size,), self.fake_label, device=self.device).float())

        err_d=err_d_real+err_d_fake

        if optimize:
            err_d.backward()
            self.optimizerD.step()

        return err_d_real, err_d_fake, err_d

    def update_netg(self, x, optimize=True):
        self.G.zero_grad()
        fake, _ = self.G(x)
        out_g, feat_fake = self.D(fake)
        _, feat_real = self.D(x)

        err_g_adv = self.mse_criterion(feat_fake, feat_real)  # loss for feature matching
        err_g_rec = self.mse_criterion(fake, x)  # constrain x' to look like x

        err_g =  err_g_rec + err_g_adv * self.w_adv

        if optimize:
            err_g.backward()
            self.optimizerG.step()

        return err_g_adv, err_g_rec, err_g

    ##
    def reinitialize_netd(self):
        """ Initialize the weights of netD
        """
        self.D.apply(weights_init)
        print('Reloading d net')