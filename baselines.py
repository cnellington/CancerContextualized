import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from contextualized.dags.graph_utils import project_to_dag_torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

dag_pred = lambda X, W: torch.matmul(X.unsqueeze(1), W).squeeze(1)
mse_loss = lambda y_true, y_pred: ((y_true - y_pred)**2).mean()
l1_loss = lambda w, l1: l1 * torch.norm(w, p=1)
def dag_loss(w, alpha, rho):
    d = w.shape[-1]
    m = torch.linalg.matrix_exp(w * w)
    h = torch.trace(m) - d
    return alpha * h + 0.5 * rho * h * h


class NeighborhoodSelectionModule(pl.LightningModule):
    def __init__(self, x_dim, fit_intercept=True, l1=1e-3, learning_rate=1e-2):
        super().__init__()
        self.learning_rate = learning_rate
        self.l1 = l1
        diag_mask = torch.ones(x_dim, x_dim) - torch.eye(x_dim)
        self.register_buffer("diag_mask", diag_mask)
        init_mat = (torch.rand(x_dim, x_dim) * 2e-2 - 1e-2) * diag_mask
        self.W = nn.parameter.Parameter(init_mat, requires_grad=True)
        if fit_intercept:
            init_mu = torch.rand(x_dim) * 2e-2 - 1e-2
            self.mu = nn.parameter.Parameter(init_mu, requires_grad=True)
        else:
            self.register_buffer("mu", torch.zeros(x_dim))

    def forward(self, X):
        X_centered = X - self.mu
        W = self.W * self.diag_mask
        return dag_pred(X_centered, W) + self.mu

    def _batch_loss(self, batch, batch_idx):
        x_true = batch
        x_pred = self(x_true)
        mse = mse_loss(x_true, x_pred)
        l1 = l1_loss(self.W, self.l1)
        return mse + l1

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
        self.log_dict({'val_loss': mse})
        return mse

    def test_step(self, batch, batch_idx):
        loss = self._batch_loss(batch, batch_idx)
        self.log_dict({'test_loss': loss})
        return loss

    def dataloader(self, X, batch_size=32, **kwargs):
        X_tensor = torch.Tensor(X).to(torch.float32)
        return DataLoader(dataset=X_tensor, batch_size=batch_size, **kwargs)

    def get_params(self, n, **kwargs):
        W = self.W.detach() * self.diag_mask
        W_batch = W.unsqueeze(0).expand(n, -1, -1)
        mu = self.mu.detach()
        mu_batch = mu.unsqueeze(0).expand(n, -1)
        return W_batch.numpy(), mu_batch.numpy()


class MarkovNetworkModule(NeighborhoodSelectionModule):
    def forward(self, X):
        X_centered = X - self.mu
        W = (self.W + self.W.T) * self.diag_mask
        return dag_pred(X_centered, W) + self.mu

    def _batch_loss(self, batch, batch_idx):
        x_true = batch
        x_pred = self(x_true)
        mse = mse_loss(x_true, x_pred)
        l1 = l1_loss((self.W + self.W.T), self.l1)
        return mse + l1

    def get_params(self, n, **kwargs):
        W = (self.W + self.W.T).detach() * self.diag_mask
        W_batch = W.unsqueeze(0).expand(n, -1, -1)
        mu = self.mu.detach()
        mu_batch = mu.unsqueeze(0).expand(n, -1)
        return W_batch.numpy(), mu_batch.numpy()


class CorrelationNetworkModule(NeighborhoodSelectionModule):
    def __init__(self, x_dim, fit_intercept=True,  learning_rate=1e-2):
        super().__init__(x_dim, l1=0, learning_rate=learning_rate)
        self.learning_rate = learning_rate
        diag_mask = torch.ones(x_dim, x_dim) - torch.eye(x_dim)
        self.register_buffer("diag_mask", diag_mask)
        init_mat = (torch.rand(x_dim, x_dim) * 2e-2 - 1e-2) * diag_mask
        self.W = nn.parameter.Parameter(init_mat, requires_grad=True)
        if fit_intercept:
            init_mu = torch.rand(x_dim) * 2e-2 - 1e-2
            self.mu = nn.parameter.Parameter(init_mu, requires_grad=True)
        else:
            self.register_buffer("mu", torch.zeros(x_dim))

    def forward(self, X):
        X_centered = X - self.mu
        X_tiled = X_centered.unsqueeze(1).expand(-1, X.shape[-1], -1)
        return X_tiled * self.W + self.mu.unsqueeze(-1).expand(-1, X.shape[-1])

    def _batch_loss(self, batch, batch_idx):
        x_true = batch
        x_pred_tiled = self(x_true)
        x_true_tiled = x_true.unsqueeze(-1).expand(-1, -1, x_true.shape[-1])
        mse = mse_loss(x_true_tiled, x_pred_tiled)
        return mse

    def validation_step(self, batch, batch_idx):
        loss = self._batch_loss(batch, batch_idx)
        self.log_dict({'val_loss': loss})
        return loss

    def get_params(self, n, **kwargs):
        W = self.W.detach()
        W[torch.sign(W) != torch.sign(W.T)] = 0.0
        corrs = torch.sqrt(W * W.T)
        corrs_batch = corrs.unsqueeze(0).expand(n, -1, -1)
        mu = self.mu.detach()
        mu_batch = mu.unsqueeze(0).expand(n, -1)
        return corrs_batch.numpy(), mu_batch.numpy()


class NOTEARS(NeighborhoodSelectionModule):
    def __init__(self, x_dim, l1=1e-3, alpha=1e-8, rho=1e-8, learning_rate=1e-2):
        super().__init__(x_dim, fit_intercept=False, l1=l1, learning_rate=learning_rate)
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
        self.register_buffer("mu", torch.zeros(x_dim))  # placeholder, but no intercept enabled for now
    
    def _batch_loss(self, batch, batch_idx):
        x_true = batch
        x_pred = self(x_true)
        mse = mse_loss(x_true, x_pred)
        l1 = l1_loss(self.W, self.l1)
        dag = dag_loss(self.W, self.alpha, self.rho)
        return 0.5 * mse + l1 + dag
    
    def validation_step(self, batch, batch_idx):
        x_true = batch
        x_pred = self(x_true).detach()
        mse = mse_loss(x_true, x_pred)
        dag = dag_loss(self.W, 1e12, 1e12).detach()
        loss = mse + dag
        self.log_dict({'val_loss': loss})
        return loss
    
    def on_train_epoch_end(self, *args, **kwargs):
        dag = dag_loss(self.W, self.alpha, self.rho).item()
        if dag > self.tolerance * self.prev_dag and self.alpha < 1e12 and self.rho < 1e12:
            self.alpha = self.alpha + self.rho * dag
            self.rho = self.rho * 10
        self.prev_dag = dag

    def get_params(self, n, project_to_dag=True, **kwargs):
        W = self.W.detach() * self.diag_mask
        if project_to_dag:
            W = torch.tensor(project_to_dag_torch(W.numpy(force=True))[0])
        W_batch = W.unsqueeze(0).expand(n, -1, -1)
        mu = self.mu.detach()
        mu_batch = mu.unsqueeze(0).expand(n, -1)
        return W_batch.numpy(), mu_batch.numpy()


class NeighborhoodSelection:
    def __init__(self, **kwargs):
        self.model_class = NeighborhoodSelectionModule
        self.kwargs = kwargs

    # def fit(self, X):  # no early stopping
    #     self.p = X.shape[-1]
    #     self.model = self.model_class(
    #         self.p,
    #         **self.kwargs,
    #     )
    #     dataset = self.model.dataloader(X)
    #     self.trainer = pl.Trainer(max_epochs=100, auto_lr_find=True, accelerator='auto', devices=1)
    #     self.trainer.fit(self.model, dataset)
    #     return self

    def fit(self, X, val_split=0.2):
        self.p = X.shape[-1]
        model = self.model_class(
            self.p,
            **self.kwargs,
        )
        if val_split > 0 and len(X) > 1:
            X_train, X_val = train_test_split(X, test_size=0.2, random_state=1)
            train_dataset = model.dataloader(X_train)
            val_dataset = model.dataloader(X_val)
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=f'checkpoints',  # hacky way to get unique dir
                filename='{epoch}-{val_loss:.2f}'
            )
            es_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.01,
                patience=5,
                verbose=True,
                mode="min"
            )
            self.trainer = pl.Trainer(
                max_epochs=100,
                accelerator='auto',
                devices=1,
                callbacks=[es_callback, checkpoint_callback]
            )
            self.trainer.fit(model, train_dataset, val_dataset)

            # Get best checkpoint by val_loss
            best_checkpoint = torch.load(checkpoint_callback.best_model_path)
            model.load_state_dict(best_checkpoint['state_dict'])
            self.model = model
        else:
            train_dataset = model.dataloader(X)
            checkpoint_callback = ModelCheckpoint(
                monitor="train_loss",
                dirpath=f'checkpoints',  # hacky way to get unique dir
                filename='{epoch}-{train_loss:.2f}'
            )
            es_callback = EarlyStopping(
                monitor="train_loss",
                min_delta=0.01,
                patience=5,
                verbose=True,
                mode="min"
            )
            self.trainer = pl.Trainer(
                max_epochs=100,
                accelerator='auto',
                devices=1,
                callbacks=[es_callback, checkpoint_callback]
            )
            self.trainer.fit(model, train_dataset)

            # Get best checkpoint by train_loss
            best_checkpoint = torch.load(checkpoint_callback.best_model_path)
            model.load_state_dict(best_checkpoint['state_dict'])
            self.model = model
        return self


    def predict(self, n):
        return self.model.get_params(n)[0]

    def mses(self, X):
        X_preds = torch.cat(self.trainer.predict(self.model, self.model.dataloader(X)), dim=0).detach().numpy()
        return ((X_preds - X) ** 2).mean(axis=1)


class CorrelationNetwork(NeighborhoodSelection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_class = CorrelationNetworkModule

    def mses(self, X):
        X_preds_tiled = torch.cat(self.trainer.predict(self.model, self.model.dataloader(X)), dim=0).detach().numpy()
        X_tiled = np.tile(np.expand_dims(X, axis=-1), (1, 1, X.shape[-1]))
        return ((X_preds_tiled - X_tiled) ** 2).mean(axis=(1, 2))


class MarkovNetwork(NeighborhoodSelection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_class = MarkovNetworkModule


class BayesianNetwork(NeighborhoodSelection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_class = NOTEARS

    def mses(self, X):
        W_pred = torch.tensor(self.model.get_params(1, project_to_dag=True)[0], dtype=torch.float32)
        X_preds = dag_pred(torch.tensor(X, dtype=torch.float32), W_pred).detach().numpy()
        return ((X_preds - X)**2).mean(axis=1)


class CorrelationNetworkSKLearn:
    def fit(self, X, **kwargs):
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
        corrs_batch = np.tile(np.expand_dims(corrs, axis=0), (n, 1, 1))
        mus_batch = np.zeros((n, self.p))  # dummy variable for consistency
        return np.tile(np.expand_dims(corrs, axis=0), (n, 1, 1)), mus_batch
    
    def mses(self, X):
        mses = np.zeros(len(X))
        for i in range(self.p):
            for j in range(self.p):
                residual = self.regs[i][j].predict(X[:, j, np.newaxis]) - X[:, i, np.newaxis]
                residual = residual[:, 0]
                mses += (residual ** 2) / self.p**2
        return mses
    

class NeighborhoodSelectionSKLearn:
    def __init__(self, alpha=0.0, l1_ratio=1.0):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        if alpha == 0:
            self.model_class = lambda: LinearRegression()
        elif l1_ratio == 1:
            self.model_class = lambda: Lasso(alpha=alpha)
        else:
            self.model_class = lambda: ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    def fit(self, X, **kwargs):
        self.p = X.shape[-1]
        self.regs = [self.model_class() for _ in range(self.p)]
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
        mus = np.array([reg.intercept_[0] for reg in self.regs])
        mus_batch = np.tile(np.expand_dims(mus, axis=0), (n, 1))
        return np.tile(np.expand_dims(precision, axis=0), (n, 1, 1)), mus_batch
    
    def mses(self, X):
        mses = np.zeros(len(X))
        for i in range(self.p):
            mask = np.ones_like(X)
            mask[:, i] = 0
            residual = self.regs[i].predict(X * mask) - X[:, i, np.newaxis]
            residual = residual[:, 0]
            mses += (residual ** 2) / self.p
        return mses


class GroupedNetworks:
    def __init__(self, model_class):
        self.model_class = model_class
    
    def fit(self, X, labels, **kwargs):
        self.models = {}
        self.p = X.shape[-1]
        for label in np.unique(labels):
            label_idx = labels == label
            X_label = X[label_idx]
            model = self.model_class().fit(X_label, **kwargs)
            self.models[label] = model
        return self
    
    def predict(self, labels):
        networks = np.zeros((len(labels), self.p, self.p))
        mus = np.zeros((len(labels), self.p))
        for label in np.unique(labels):
            label_idx = labels == label
            networks[label_idx], mus[label_idx] = self.models[label].predict(label_idx.sum())
        return networks
    
    def mses(self, X, labels):
        mses = np.zeros(len(X))
        for label in np.unique(labels):
            label_idx = labels == label
            X_label = X[label_idx]
            mses[label_idx] = self.models[label].mses(X_label)
        return mses
    

if __name__ == '__main__':
    n, x_dim = 10, 20
    labels = np.random.randint(0, 5, (n,))
    X = np.random.normal(0, 1, (n, x_dim))
    X_test = np.random.normal(0, 1, (n, x_dim))

    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    for model_class, name in [(NeighborhoodSelection, 'Neighborhood'),
                              (CorrelationNetwork, 'Correlation'),
                              (MarkovNetwork, 'Markov'),
                              (BayesianNetwork, 'DAG'),
                              (NeighborhoodSelectionSKLearn, 'NeighborhoodSKLearn'),
                              (CorrelationNetworkSKLearn, 'CorrelationSKLearn'),
                              ]:
        model = model_class().fit(X, val_split=0)
        model.predict(n)
        print(name, model.mses(X).mean(), model.mses(X_test).mean())

    # grouped_corr = GroupedNetworks(CorrelationNetwork).fit(X, labels)
    # grouped_corr.predict(labels)
    # print('Grouped Correlation', grouped_corr.mses(X, labels).mean())

    # grouped_neighborhood = GroupedNetworks(NeighborhoodSelection).fit(X, labels)
    # grouped_neighborhood.predict(labels)
    # print('Grouped Neighborhood', grouped_neighborhood.mses(X, labels).mean())

    # grouped_markov = GroupedNetworks(MarkovNetwork).fit(X, labels)
    # grouped_markov.predict(labels)
    # print('Grouped Markov', grouped_markov.mses(X, labels).mean())

    # grouped_dag = GroupedNetworks(BayesianNetwork).fit(X, labels)
    # grouped_dag.predict(labels)
    # print('Grouped DAG', grouped_dag.mses(X, labels).mean())