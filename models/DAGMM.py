import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import tqdm


class ComputeLoss:
    def __init__(self, model, lambda_energy, lambda_cov, device, n_gmm):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.device = device
        self.n_gmm = n_gmm

    def forward(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x - x_hat).pow(2))

        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        return Variable(loss, requires_grad=True)

    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1)) * eps).to(self.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2 * np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)

        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0) * E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if sample_mean == True:
            E_z = torch.mean(E_z)
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """
        # phi = D
        phi = torch.sum(gamma, dim=0) / gamma.size(0)

        # mu = KxD
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov


class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.linalg.cholesky(a)
        ctx.save_for_backward(l)
        return l

    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)


class DAGMMModel(nn.Module):
    def __init__(self, feats, n_gmm=2, z_dim=1):
        """Network for DAGMM (KDDCup99)"""
        super(DAGMMModel, self).__init__()
        # Encoder network
        self.fc0 = nn.Linear(feats, 118)
        self.fc1 = nn.Linear(118, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4 = nn.Linear(10, z_dim)

        # Decoder network
        self.fc5 = nn.Linear(z_dim, 10)
        self.fc6 = nn.Linear(10, 30)
        self.fc7 = nn.Linear(30, 60)
        self.fc8 = nn.Linear(60, 118)
        self.fc9 = nn.Linear(118, feats)

        # Estimation network
        self.fc10 = nn.Linear(z_dim + 2, 10)
        self.fc11 = nn.Linear(10, n_gmm)

    def encode(self, x):
        h = torch.tanh(self.fc0(x))
        h = torch.tanh(self.fc1(h))
        h = torch.tanh(self.fc2(h))
        h = torch.tanh(self.fc3(h))
        return self.fc4(h)

    def decode(self, x):
        h = torch.tanh(self.fc5(x))
        h = torch.tanh(self.fc6(h))
        h = torch.tanh(self.fc7(h))
        h = torch.tanh(self.fc8(h))
        return self.fc9(h)

    def estimate(self, z):
        h = F.dropout(torch.tanh(self.fc10(z)), 0.5)
        return F.softmax(self.fc11(h), dim=1)

    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity

    def forward(self, x):
        bs, length, dim = x.shape
        x = x. reshape(bs, -1)
        z_c = self.encode(x)
        x_hat = self.decode(z_c)
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        x_hat = x_hat.reshape(bs, length, dim)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        gamma = self.estimate(z)
        return z_c, x_hat, z, gamma


class Solver():
    def __init__(self, configs,
                 win_size=256,
                 feats=1,
                 learning_rate=1e-4,
                 epochs=20,
                 batch_size=128):
        super(Solver, self).__init__()
        self.n_gmm = 4
        self.latent_dim = 1
        self.lambda_energy = 0.1
        self.lambda_cov = 0.005

        self.phi = None
        self.mu = None
        self.cov = None

        self.win_size = win_size
        self.feats = feats
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DAGMMModel(feats=self.win_size*self.feats, n_gmm=self.n_gmm, z_dim=self.latent_dim).to(self.device)
        self.model.apply(weights_init_normal)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50], gamma=0.1)
        self.compute = ComputeLoss(self.model, self.lambda_energy, self.lambda_cov,
                                   self.device, self.n_gmm)

    def _get_decision_params(self, dataloader_train):
        self.model.eval()
        compute = ComputeLoss(self.model, None, None, self.device, self.n_gmm)
        with torch.no_grad():
            N_samples = 0
            gamma_sum = 0
            mu_sum = 0
            cov_sum = 0
            # Obtaining the parameters gamma, mu and cov using the trainin (clean) data.
            for x, _ in dataloader_train:
                x = x.float().to(self.device)

                _, _, z, gamma = self.model(x)
                phi_batch, mu_batch, cov_batch = compute.compute_params(z, gamma)

                batch_gamma_sum = torch.sum(gamma, dim=0)
                gamma_sum += batch_gamma_sum
                mu_sum += mu_batch * batch_gamma_sum.unsqueeze(-1)
                cov_sum += cov_batch * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)

                N_samples += x.size(0)

            train_phi = gamma_sum / N_samples
            train_mu = mu_sum / gamma_sum.unsqueeze(-1)
            train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        self.phi = train_phi
        self.mu = train_mu
        self.cov = train_cov

    def fit(self, train_loader, valid_loader):
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            n = epoch + 1
            avg_loss = 0

            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for idx, (d, _) in loop:
                d = d.to(self.device)
                # Zero the network parameter gradients
                self.optimizer.zero_grad()
                # Update network parameters via backpropagation: forward + backward + optimize
                _, x_hat, z, gamma = self.model(d)

                loss = self.compute.forward(d, x_hat, z, gamma)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
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
                        _, x_hat, z, gamma = self.model(d)
                        loss = self.compute.forward(d, x_hat, z, gamma)
                        avg_loss_val += loss.cpu().item()
                        loop.set_description(f"Validation Epoch [{epoch}/{self.epochs}]")
                        loop.set_postfix(loss=loss.item(), avg_loss_val=avg_loss_val / (idx + 1))

            self.scheduler.step()
        self._get_decision_params(train_loader)


    def decision_function(self, test_loader):
        self.model.eval()
        scores = []
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)

        with torch.no_grad():
            for idx, (x, _) in loop:
                x = x.float().to(self.device)
                _, _, z, gamma = self.model(x)
                sample_energy, cov_diag = self.compute.compute_energy(z, gamma, self.phi,
                                                                 self.mu, self.cov,
                                                                 sample_mean=False)
                scores.append(sample_energy.cpu())

        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()
        self.__anomaly_score = scores

        return self.__anomaly_score