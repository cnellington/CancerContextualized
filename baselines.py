import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as pl
from contextualized.dags.graph_utils import project_to_dag_torch


dag_pred = lambda X, W: torch.matmul(X.unsqueeze(1), W).squeeze(1)
mse_loss = lambda y_true, y_pred: ((y_true - y_pred)**2).mean()
l1_loss = lambda w, l1: l1 * torch.norm(w, p=1)
def dag_loss(w, alpha, rho):
    d = w.shape[-1]
    m = torch.linalg.matrix_exp(w * w)
    h = torch.trace(m) - d
    return alpha * h + 0.5 * rho * h * h


class NOTEARSTrainer(pl.Trainer):
    def predict(self, model, dataloader):
        preds = super().predict(model, dataloader)
        return torch.cat(preds)

    def predict_w(self, model, dataloader, project_to_dag=True):
        preds = self.predict(model, dataloader)
        W = model.W.detach() * model.diag_mask
        W = torch.tensor(project_to_dag_torch(W.numpy(force=True))[0])
        W_batch = W.unsqueeze(0).expand(len(preds), -1, -1)
        return W_batch.numpy()


class NOTEARS(pl.LightningModule):
    def __init__(self, x_dim, l1=1e-3, alpha=1e-8, rho=1e-8, learning_rate=1e-2):
        super().__init__()
        self.learning_rate = learning_rate
        self.l1 = l1
        self.alpha = alpha
        self.rho = rho
        diag_mask = torch.ones(x_dim, x_dim) - torch.eye(x_dim)
        self.register_buffer("diag_mask", diag_mask)
        init_mat = (torch.rand(x_dim, x_dim) * 2e-2 - 1e-2) * diag_mask
        self.W = nn.parameter.Parameter(init_mat, requires_grad=True)
        self.tolerance = 0.25
        self.prev_dag = 0.0
        
    def forward(self, X):
        W = self.W * self.diag_mask
        return dag_pred(X, W)
    
    def _batch_loss(self, batch, batch_idx):
        x_true = batch
        x_pred = self(x_true)
        mse = mse_loss(x_true, x_pred)
        l1 = l1_loss(self.W, self.l1)
        dag = dag_loss(self.W, self.alpha, self.rho)
        return 0.5 * mse + l1 + dag
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        loss = self._batch_loss(batch, batch_idx)
        self.log_dict({'train_loss': loss})
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_true = batch
        x_pred = self(x_true).detach()
        mse = mse_loss(x_true, x_pred)
        dag = dag_loss(self.W, 1e12, 1e12).detach()
        loss = mse + dag
        self.log_dict({'val_loss': loss})
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._batch_loss(batch, batch_idx)
        self.log_dict({'test_loss': loss})
        return loss
    
    def on_train_epoch_end(self, *args, **kwargs):
        dag = dag_loss(self.W, self.alpha, self.rho).item()
        if dag > self.tolerance * self.prev_dag and self.alpha < 1e12 and self.rho < 1e12:
            self.alpha = self.alpha + self.rho * dag
            self.rho = self.rho * 10
        self.prev_dag = dag
    
    def dataloader(self, X, **kwargs):
        kwargs['batch_size'] = kwargs.get('batch_size', 32)
        X_tensor = torch.Tensor(X).to(torch.float32)
        return DataLoader(dataset=X_tensor, **kwargs)


class CorrelationNetwork:
    def fit(self, X):
        self.p = X.shape[-1]
        self.regs = [[LinearRegression() for _ in range(self.p)] for _ in range(self.p)]
        for i in range(self.p):
            for j in range(self.p):
                self.regs[i][j].fit(X[:, j, np.newaxis], X[:, i, np.newaxis])
        return self
    
    def predict(self, n):
        betas = np.zeros((self.p, self.p))
        for i in range(self.p):
            for j in range(self.p):
                betas[i, j] = self.regs[i][j].coef_.squeeze()
        corrs = betas * betas.T
        return np.tile(np.expand_dims(corrs, axis=0), (n, 1, 1))
    
    def mses(self, X):
        mses = np.zeros(len(X))
        for i in range(self.p):
            for j in range(self.p):
                residual = self.regs[i][j].predict(X[:, j, np.newaxis]) - X[:, i, np.newaxis]
                residual = residual[:, 0]
                mses += (residual ** 2) / self.p**2
        return mses
    

class MarkovNetwork:
    def __init__(self, alpha=0.0, l1_ratio=1.0):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit(self, X):
        self.p = X.shape[-1]
        self.regs = [ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio) for _ in range(self.p)]
        for i in range(self.p):
            mask = np.ones_like(X)
            mask[:, i] = 0
            self.regs[i].fit(X * mask, X[:, i, np.newaxis])
        return self
    
    def predict(self, n):
        betas = np.zeros((self.p, self.p))
        for i in range(self.p):
            betas[i] = self.regs[i].coef_.squeeze()
            betas[i, i] = 0
        precision = - np.sign(betas) * np.sqrt(np.abs(betas * betas.T))
        return np.tile(np.expand_dims(precision, axis=0), (n, 1, 1))
    
    def mses(self, X):
        mses = np.zeros(len(X))
        for i in range(self.p):
            mask = np.ones_like(X)
            mask[:, i] = 0
            residual = self.regs[i].predict(X * mask) - X[:, i, np.newaxis]
            residual = residual[:, 0]
            mses += (residual ** 2) / self.p
        return mses


class BayesianNetwork:
    def fit(self, X):
        self.p = X.shape[-1]
        self.model = NOTEARS(
            self.p,
            l1=1e-3,
            alpha=1e-8,
            rho=1e-8,
            learning_rate=1e-2,
        )
        dataset = self.model.dataloader(X)
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.trainer = NOTEARSTrainer(max_epochs=50, auto_lr_find=True, accelerator=accelerator, devices=1)
        self.trainer.fit(self.model, dataset)
        return self

    def predict(self, n):
        dummy_X = np.zeros((n, self.p))
        dummy_dataset = self.model.dataloader(dummy_X)
        return self.trainer.predict_w(self.model, dummy_dataset)

    def mses(self, X):
        mses = np.zeros(len(X))
#         W_pred = self.model.W.detach()
        W_pred = torch.tensor(self.predict(1)[0], dtype=torch.float32)
        X_preds = dag_pred(torch.tensor(X, dtype=torch.float32), W_pred).detach().numpy()
        return ((X_preds - X)**2).mean(axis=1)


class GroupedNetworks:
    def __init__(self, model_class):
        self.model_class = model_class
    
    def fit(self, X, labels):
        self.models = {}
        self.p = X.shape[-1]
        for label in np.unique(labels):
            label_idx = labels == label
            X_label = X[label_idx]
            model = self.model_class().fit(X_label)
            self.models[label] = model
        return self
    
    def predict(self, labels):
        networks = np.zeros((len(labels), self.p, self.p))
        for label in np.unique(labels):
            label_idx = labels == label
            networks[label_idx] = self.models[label].predict(label_idx.sum())
        return networks
    
    def mses(self, X, labels):
        mses = np.zeros(len(X))
        for label in np.unique(labels):
            label_idx = labels == label
            X_label = X[label_idx]
            mses[label_idx] = self.models[label].mses(X_label)
        return mses


class ClusteredCorrelation:
    def __init__(self, K, clusterer=None):
        self.K = K
        if clusterer is None:
            self.kmeans = KMeans(n_clusters=K)
            self.prefit = False
        else:
            self.kmeans = clusterer
            self.prefit = True
        self.models = {k: PopulationCorrelation() for k in range(K)}
    
    def fit(self, C, X):
        self.p = X.shape[-1]
        if not self.prefit:
            self.kmeans.fit(C)
        labels = self.kmeans.predict(C)
        for k in range(self.K):
            k_idx = labels == k
            X_k, C_k = X[k_idx], C[k_idx]
            self.models[k].fit(C_k, X_k)
        return self
    
    def predict(self, C):
        labels = self.kmeans.predict(C)
        corrs = np.zeros((len(C), self.p, self.p))
        for label in np.unique(labels):
            l_idx = labels == label
            C_l = C[l_idx]
            corrs[l_idx] = self.models[label].predict(C_l)
        return corrs
    
    def mses(self, C, X):
        labels = self.kmeans.predict(C)
        mses = np.zeros(len(C))
        for label in np.unique(labels):
            l_idx = labels == label
            l_count = sum(l_idx)
            C_l, X_l = C[l_idx], X[l_idx]
            mses[l_idx] = self.models[label].mses(C_l, X_l)
        return mses
    

if __name__ == '__main__':
    n, x_dim = 100, 20
    labels = np.random.randint(0, 5, (n,))
    X = np.random.uniform(-1, 1, (n, x_dim))

    corr = CorrelationNetwork().fit(X)
    corr.predict(n)
    print(corr.mses(X).mean())

    grouped_corr = GroupedNetworks(CorrelationNetwork).fit(X, labels)
    grouped_corr.predict(labels)
    print(grouped_corr.mses(X, labels).mean())
    
    mark = MarkovNetwork().fit(X)
    mark.predict(n)
    print(mark.mses(X).mean())

    grouped_mark = GroupedNetworks(MarkovNetwork).fit(X, labels)
    grouped_mark.predict(labels)
    print(grouped_mark.mses(X, labels).mean())

    dag = BayesianNetwork().fit(X)
    dag.predict(n)
    print(dag.mses(X).mean())

    grouped_dag = GroupedNetworks(BayesianNetwork).fit(X, labels)
    grouped_dag.predict(labels)
    print(grouped_dag.mses(X, labels).mean())
