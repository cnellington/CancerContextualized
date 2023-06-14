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
    def __init__(self, x_dim, fit_intercept=False, l1=1e-3, learning_rate=1e-2):
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
    def __init__(self, x_dim, **kwargs):
        super().__init__(x_dim, l1=0, **kwargs)

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
        W_batch = W.unsqueeze(0).expand(n, -1, -1)
        # W[torch.sign(W) != torch.sign(W.T)] = 0.0
        # corrs = torch.sqrt(W * W.T)
        # corrs_batch = corrs.unsqueeze(0).expand(n, -1, -1)
        mu = self.mu.detach()
        mu_batch = mu.unsqueeze(0).expand(n, -1)
        return W_batch.numpy(), mu_batch.numpy()


class NOTEARS(NeighborhoodSelectionModule):
    def __init__(self, x_dim, alpha=1e-3, rho=1e-3, fit_intercept=False, **kwargs):
        if fit_intercept:
            print('intercept fitting not implemented for NOTEARS')
        super().__init__(x_dim, fit_intercept=False, **kwargs)
        self.alpha = alpha
        self.rho = rho
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
        dag = dag_loss(self.W, 1., 1.).detach()
        loss = mse + dag
        self.log_dict({'val_loss': loss})
        return loss
    
    def on_train_epoch_end(self, *args, **kwargs):
        dag = dag_loss(self.W, self.alpha, self.rho).item()
        if dag > self.tolerance * self.prev_dag and self.alpha < 1e12 and self.rho < 1e12:
            self.alpha = self.alpha + self.rho * dag
            self.rho = self.rho * 1.2
        self.prev_dag = dag

    def get_params(self, n, project_to_dag=False, **kwargs):
        W = self.W.detach() * self.diag_mask
        if project_to_dag:
            W = torch.tensor(project_to_dag_torch(W.numpy(force=True))[0])
        W_batch = W.unsqueeze(0).expand(n, -1, -1)
        mu = self.mu.detach()
        mu_batch = mu.unsqueeze(0).expand(n, -1)
        return W_batch.numpy(), mu_batch.numpy()


class NeighborhoodSelection:
    def __init__(self, verbose=False, **kwargs):
        self.model_class = NeighborhoodSelectionModule
        self.kwargs = kwargs
        self.verbose = verbose

    def fit(self, C, X, n_bootstraps=1, val_split=0.2):
        self.p = X.shape[-1]
        self.models = []
        self.trainers = []
        for boot_i in range(n_bootstraps):
            np.random.seed(boot_i)
            if n_bootstraps > 1:
                boot_idx = np.random.choice(len(X), size=len(X), replace=True)
            else:
                boot_idx = np.arange(len(X))
            X_boot = X[boot_idx]
            model = self.model_class(
                self.p,
                **self.kwargs,
            )
            if val_split > 0 and len(X) > 1:
                X_train, X_val = train_test_split(X_boot, test_size=0.2, random_state=1)
                train_dataset = model.dataloader(X_train)
                val_dataset = model.dataloader(X_val)
                checkpoint_callback = ModelCheckpoint(
                    monitor="val_loss",
                    filename='{epoch}-{val_loss:.5f}'
                )
                es_callback = EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-3,
                    patience=5,
                    verbose=True,
                    mode="min"
                )
                trainer = pl.Trainer(
                    max_epochs=100,
                    accelerator='auto',
                    devices=1,
                    callbacks=[es_callback, checkpoint_callback],
                    enable_progress_bar=self.verbose,
                )
                trainer.fit(model, train_dataset, val_dataset)

                # Get best checkpoint by val_loss
                best_checkpoint = torch.load(checkpoint_callback.best_model_path)
                model.load_state_dict(best_checkpoint['state_dict'])
                self.models.append(model)
                self.trainers.append(trainer)
            else:
                train_dataset = model.dataloader(X_boot)
                checkpoint_callback = ModelCheckpoint(
                    monitor="train_loss",
                    filename='{epoch}-{train_loss:.5f}'
                )
                es_callback = EarlyStopping(
                    monitor="train_loss",
                    min_delta=1e-3,
                    patience=5,
                    verbose=True,
                    mode="min"
                )
                trainer = pl.Trainer(
                    max_epochs=100,
                    accelerator='auto',
                    devices=1,
                    callbacks=[es_callback, checkpoint_callback],
                    enable_progress_bar=self.verbose,
                )
                trainer.fit(model, train_dataset)

                # Get best checkpoint by train_loss
                best_checkpoint = torch.load(checkpoint_callback.best_model_path)
                model.load_state_dict(best_checkpoint['state_dict'])
                self.models.append(model)
                self.trainers.append(trainer)
        return self

    def predict_networks(self, C, avg_bootstraps=True):
        n = len(C)
        Ws = []
        mus = []
        for model in self.models:
            W, mu = model.get_params(n)
            Ws.append(W)
            mus.append(mu)
        if avg_bootstraps:
            return np.mean(Ws, axis=0), np.mean(mus, axis=0)
        else:
            return np.array(Ws), np.array(mus)

    def predict(self, C, X, avg_bootstraps=True):
        X_preds = []
        for trainer, model in zip(self.trainers, self.models):
            X_pred = torch.cat(trainer.predict(model, model.dataloader(X)), dim=0).detach().numpy()
            X_preds.append(X_pred)
        if avg_bootstraps:
            return np.mean(X_preds, axis=0)
        else:
            return np.array(X_preds)

    def mses(self, C, X, avg_bootstraps=True):
        X_preds = self.predict(C, X, avg_bootstraps=False)
        if avg_bootstraps:
            return ((X_preds - X) ** 2).mean(axis=1)
        else:
            return np.array([((X_pred - X) ** 2).mean(axis=1) for X_pred in X_preds])


class CorrelationNetwork(NeighborhoodSelection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_class = CorrelationNetworkModule

    def predict(self, C, X, avg_bootstraps=True):
        X_preds_tiled = []
        for trainer, model in zip(self.trainers, self.models):
            X_pred_tiled = torch.cat(trainer.predict(model, model.dataloader(X)), dim=0).detach().numpy()
            X_preds_tiled.append(X_pred_tiled)
        if avg_bootstraps:
            return np.mean(X_preds_tiled, axis=0)
        else:
            return np.array(X_preds_tiled)

    def mses(self, C, X, avg_bootstraps=True):
        X_tiled = np.tile(np.expand_dims(X, axis=-1), (1, 1, X.shape[-1]))
        X_preds_tiled = self.predict(C, X, avg_bootstraps=False)
        if avg_bootstraps:
            return ((X_preds_tiled - X_tiled) ** 2).mean(axis=(1, 2))
        else:
            return np.array([((X_pred_tiled - X_tiled) ** 2).mean(axis=1) for X_pred_tiled in X_preds_tiled])


class MarkovNetwork(NeighborhoodSelection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_class = MarkovNetworkModule


class BayesianNetwork(NeighborhoodSelection):
    def __init__(self, project_to_dag=False, **kwargs):
        super().__init__(**kwargs)
        self.model_class = NOTEARS
        self.project_to_dag = project_to_dag

    def predict_networks(self, C, avg_bootstraps=True):
        n = len(C)
        Ws = [self.models[i].get_params(n)[0] for i in range(len(self.models))]
        if avg_bootstraps:
            Ws = np.mean(Ws, axis=0)
        return Ws

    def predict(self, C, X, avg_bootstraps=True):
        X_preds = []
        for model in self.models:
            W_pred = torch.tensor(model.get_params(1, project_to_dag=self.project_to_dag)[0], dtype=torch.float32)
            X_pred = dag_pred(torch.tensor(X, dtype=torch.float32), W_pred).detach().numpy()
            X_preds.append(X_pred)
        if avg_bootstraps:
            return np.mean(X_preds, axis=0)
        else:
            return np.array(X_preds)

    def mses(self, C, X, avg_bootstraps=True):
        X_preds = self.predict(C, X, avg_bootstraps=False)
        if avg_bootstraps:
            return ((X_preds - X) ** 2).mean(axis=1)
        else:
            return np.array([((X_pred - X) ** 2).mean(axis=1) for X_pred in X_preds])


class CorrelationNetworkSKLearn:
    def __init__(self, fit_intercept=False, verbose=None):
        self.fit_intercept = fit_intercept

    def fit(self, C, X, **kwargs):
        self.p = X.shape[-1]
        self.regs = [[LinearRegression(fit_intercept=self.fit_intercept) for _ in range(self.p)] for _ in range(self.p)]
        for i in range(self.p):
            for j in range(self.p):
                self.regs[i][j].fit(X[:, j, np.newaxis], X[:, i, np.newaxis])
        return self

    def predict_networks(self, C):
        n = len(C)
        betas = np.zeros((self.p, self.p))
        for i in range(self.p):
            for j in range(self.p):
                betas[i, j] = self.regs[i][j].coef_.squeeze()
        corrs = betas * betas.T
        corrs_batch = np.tile(np.expand_dims(corrs, axis=0), (n, 1, 1))
        mus_batch = np.zeros((n, self.p))  # dummy variable for consistency
        return np.tile(np.expand_dims(corrs, axis=0), (n, 1, 1)), mus_batch

    def predict(self, C, X):
        X_pred = np.zeros((len(C), self.p, self.p))
        for i in range(self.p):
            for j in range(self.p):
                X_pred[:, i, j] = self.regs[i][j].predict(X[:, j, np.newaxis])[:, 0]
        return X_pred
    
    def mses(self, C, X):
        X_pred = self.predict(C, X)
        residuals = X_pred - np.tile(np.expand_dims(X, axis=-1), (1, 1, self.p))
        return (residuals ** 2).mean(axis=(1, 2))
    

class NeighborhoodSelectionSKLearn:
    def __init__(self, fit_intercept=False, alpha=1e-3, l1_ratio=1.0, verbose=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        if alpha == 0:
            self.model_class = lambda: LinearRegression(fit_intercept=fit_intercept)
        elif l1_ratio == 1:
            self.model_class = lambda: Lasso(alpha=alpha, fit_intercept=fit_intercept)
        else:
            self.model_class = lambda: ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept)

    def fit(self, C, X, **kwargs):
        self.p = X.shape[-1]
        self.regs = [self.model_class() for _ in range(self.p)]
        for i in range(self.p):
            mask = np.ones_like(X)
            mask[:, i] = 0
            self.regs[i].fit(X * mask, X[:, i, np.newaxis])
        return self
    
    def predict_networks(self, C):
        n = len(C)
        betas = np.zeros((self.p, self.p))
        for i in range(self.p):
            betas[i] = self.regs[i].coef_.squeeze()
            betas[i, i] = 0
        precision = - np.sign(betas) * np.sqrt(np.abs(betas * betas.T))
        if self.fit_intercept:
            mus = np.array([reg.intercept_[0] for reg in self.regs])
        else:
            mus = np.zeros(self.p)
        mus_batch = np.tile(np.expand_dims(mus, axis=0), (n, 1))
        return np.tile(np.expand_dims(precision, axis=0), (n, 1, 1)), mus_batch

    def predict(self, C, X):
        X_pred = np.zeros_like(X)
        for i in range(self.p):
            mask = np.ones_like(X)
            mask[:, i] = 0
            X_pred[:, i] = self.regs[i].predict(X * mask)
        return X_pred

    def mses(self, C, X):
        X_pred = self.predict(C, X)
        return ((X_pred - X) ** 2).mean(axis=1)


class GroupedNetworks:
    def __init__(self, model_class):
        self.model_class = model_class
    
    def fit(self, C, X, **kwargs):
        labels = C
        self.models = {}
        self.p = X.shape[-1]
        for label in np.unique(labels):
            label_idx = labels == label
            model = self.model_class().fit(C[label_idx], X[label_idx], **kwargs)
            self.models[label] = model
        return self
    
    def predict_networks(self, C):
        labels = C
        networks = np.zeros((len(labels), self.p, self.p))
        mus = np.zeros((len(labels), self.p))
        for label in np.unique(labels):
            label_idx = labels == label
            networks[label_idx], mus[label_idx] = self.models[label].predict(label_idx.sum())
        return networks

    def predict(self, C, X):
        labels = C
        X_preds = np.zeros_like(X)
        for label in np.unique(labels):
            label_idx = labels == label
            X_pred = self.models[label].predict(C[label_idx], X[label_idx])
            if len(X_preds.shape) != len(X_pred.shape):  # make the return value consistent after seeing preds
                X_preds = np.zeros((len(X), X.shape[-1], X.shape[-1]))
            X_preds[label_idx] = X_pred
        return X_preds

    def mses(self, C, X):
        labels = C
        mses = np.zeros(len(X))
        for label in np.unique(labels):
            label_idx = labels == label
            mses[label_idx] = self.models[label].mses(C[label_idx], X[label_idx])
        return mses
    

if __name__ == '__main__':
    n, x_dim = 100, 50
    C = np.random.randint(0, 5, (n,))  # labels
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
        model = model_class(fit_intercept=False, verbose=True).fit(C, X, val_split=0)
        model.predict(C, X)
        model.predict_networks(C)
        print(name, model.mses(C, X).mean(), model.mses(X_test, X_test).mean())

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