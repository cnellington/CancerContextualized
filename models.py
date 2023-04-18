import dill as pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from contextualized.regression.lightning_modules import (
    ContextualizedMarkovGraph,
    ContextualizedNeighborhoodSelection,
    ContextualizedCorrelation,
)
from contextualized.regression.trainers import MarkovTrainer, CorrelationTrainer
from contextualized.dags.lightning_modules import NOTMAD, DEFAULT_ARCH_PARAMS
from contextualized.dags.trainers import GraphTrainer
from contextualized.regression.regularizers import REGULARIZERS
from contextualized.functions import LINK_FUNCTIONS
from contextualized.dags import graph_utils
from contextualized.dags.losses import mse_loss

class ContextualizedMarkovGraphWrapper:
    def __init__(self):
        self.subtype_kwargs = {
            'num_archetypes': 40,
            'encoder_type': 'mlp',
            'encoder_kwargs': {'width': 200, 'layers': 3, 'link_fn': LINK_FUNCTIONS['identity']},
            'model_regularizer': REGULARIZERS['l1'](alpha=1e-5, mu_ratio=0.0)
        }

    def fit(self, C, X):
        self.p = X.shape[-1]
        model = ContextualizedMarkovGraph(
            C.shape[-1],
            X.shape[-1],
            **self.subtype_kwargs
        )
        boot_train_idx, boot_val_idx = train_test_split(range(len(X)), test_size=0.2, random_state=1)
        C_train, C_val = C[boot_train_idx], C[boot_val_idx]
        X_train, X_val = X[boot_train_idx], X[boot_val_idx]
        train_dataset = model.dataloader(C_train, X_train, batch_size=10)
        val_dataset = model.dataloader(C_val, X_val, batch_size=10)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=f'checkpoints/markov_wrapper/bootstrap_{np.mean(X_train)}',  # hacky way to get unique dir
            filename='{epoch}-{val_loss:.2f}'
        )
        es_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.01,
            patience=5,
            verbose=True,
            mode="min"
        )
        self.trainer = MarkovTrainer(
            max_epochs=100,
            accelerator='auto',
            devices=1,
            callbacks=[es_callback, checkpoint_callback]
        )
        self.trainer.fit(model, train_dataset, val_dataset)

        # Get best checkpoint by val_loss, test and save
        best_checkpoint = torch.load(checkpoint_callback.best_model_path)
        model.load_state_dict(best_checkpoint['state_dict'])
        self.model = model
        return self

    def predict(self, C):
        betas, mus = self.trainer.predict_params(self.model, self.model.dataloader(C, np.zeros((len(C), self.p, self.p)), batch_size=10))
        return betas, mus

    def mses(self, C, X):
        betas, mus = self.predict(C)
        mses = np.zeros(len(C))
        for i in range(X.shape[-1]):
            residuals = X[:, i] - ((betas[:, i] * X).sum(axis=-1) + mus[:, i])
            mses += residuals ** 2 / X.shape[-1]
        return mses


class ContextualizedNeighborhoodSelectionWrapper:
    def __init__(self):
        self.subtype_kwargs = {
            'num_archetypes': 40,
            'encoder_type': 'mlp',
            'encoder_kwargs': {'width': 200, 'layers': 3, 'link_fn': LINK_FUNCTIONS['identity']},
            'model_regularizer': REGULARIZERS['l1'](alpha=1e-5, mu_ratio=0.0)
        }

    def fit(self, C, X):
        self.p = X.shape[-1]
        model = ContextualizedNeighborhoodSelection(
            C.shape[-1],
            X.shape[-1],
            **self.subtype_kwargs
        )
        boot_train_idx, boot_val_idx = train_test_split(range(len(X)), test_size=0.2, random_state=1)
        C_train, C_val = C[boot_train_idx], C[boot_val_idx]
        X_train, X_val = X[boot_train_idx], X[boot_val_idx]
        train_dataset = model.dataloader(C_train, X_train, batch_size=10)
        val_dataset = model.dataloader(C_val, X_val, batch_size=10)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=f'checkpoints/neighborhood_wrapper/bootstrap_{np.mean(X_train)}',  # hacky way to get unique dir
            filename='{epoch}-{val_loss:.2f}'
        )
        es_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.01,
            patience=5,
            verbose=True,
            mode="min"
        )
        self.trainer = MarkovTrainer(
            max_epochs=100,
            accelerator='auto',
            devices=1,
            callbacks=[es_callback, checkpoint_callback]
        )
        self.trainer.fit(model, train_dataset, val_dataset)

        # Get best checkpoint by val_loss, test and save
        best_checkpoint = torch.load(checkpoint_callback.best_model_path)
        model.load_state_dict(best_checkpoint['state_dict'])
        self.model = model
        return self

    def predict(self, C):
        betas, mus = self.trainer.predict_params(self.model, self.model.dataloader(C, np.zeros((len(C), self.p, self.p)), batch_size=10))
        return betas, mus

    def mses(self, C, X):
        betas, mus = self.predict(C)
        mses = np.zeros(len(C))
        for i in range(X.shape[-1]):
            residuals = X[:, i] - ((betas[:, i] * X).sum(axis=-1) + mus[:, i])
            mses += residuals ** 2 / X.shape[-1]
        return mses


class ContextualizedCorrelationWrapper:
    def __init__(self):
        self.subtype_kwargs = {
            'num_archetypes': 40,
            'encoder_type': 'mlp',
            'encoder_kwargs': {'width': 200, 'layers': 3, 'link_fn': LINK_FUNCTIONS['identity']},
        }

    def fit(self, C, X):
        self.p = X.shape[-1]
        model = ContextualizedCorrelation(
            C.shape[-1],
            X.shape[-1],
            **self.subtype_kwargs
        )
        boot_train_idx, boot_val_idx = train_test_split(range(len(X)), test_size=0.2, random_state=1)
        C_train, C_val = C[boot_train_idx], C[boot_val_idx]
        X_train, X_val = X[boot_train_idx], X[boot_val_idx]
        train_dataset = model.dataloader(C_train, X_train, batch_size=10)
        val_dataset = model.dataloader(C_val, X_val, batch_size=10)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=f'checkpoints/correlation_wrapper/bootstrap_{np.mean(X_train)}',  # hacky way to get unique dir
            filename='{epoch}-{val_loss:.2f}'
        )
        es_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.01,
            patience=5,
            verbose=True,
            mode="min"
        )
        self.trainer = CorrelationTrainer(
            max_epochs=100,
            accelerator='auto',
            devices=1,
            callbacks=[es_callback, checkpoint_callback]
        )
        self.trainer.fit(model, train_dataset, val_dataset)

        # Get best checkpoint by val_loss, test and save
        best_checkpoint = torch.load(checkpoint_callback.best_model_path)
        model.load_state_dict(best_checkpoint['state_dict'])
        self.model = model
        return self

    def predict(self, C):
        betas, mus = self.trainer.predict_params(self.model, self.model.dataloader(C, np.zeros((len(C), self.p, self.p)), batch_size=10))
        return betas, mus

    def mses(self, C, X):
        betas, mus = self.predict(C)
        # for i in range(X.shape[-1]):
        #     for j in range(X.shape[-1]):
        #         tiled_xi = np.array([X[:, i] for _ in range(len(betas))])
        #         tiled_xj = np.array([X[:, j] for _ in range(len(betas))])
        #         residuals = tiled_xi - betas[:, i, j] * tiled_xj - mus[:, i, j]
        #         mses += residuals ** 2 / (X.shape[-1] ** 2)
        X_tiled = np.tile(X[:, :, np.newaxis], (1, 1, X.shape[-1]))
        X_preds = (X_tiled * betas) + mus
        residuals = X_tiled - np.transpose(X_preds, (0, 2, 1))
        mses = (residuals ** 2).mean(axis=(1, 2))
        return mses


class ContextualizedBayesianNetworksWrapper:
    def __init__(self):
        self.subtype_kwargs = {
            'num_archetypes': 40,
            'encoder_kwargs': {
                'type': 'mlp',
                'params': {'width': 200, 'layers': 3, 'link_fn': LINK_FUNCTIONS['identity']},
            },
            'archetype_loss_params': DEFAULT_ARCH_PARAMS,
        }
        self.subtype_kwargs['archetype_loss_params']['num_archetypes'] = 40

    def fit(self, C, X):
        self.p = X.shape[-1]
        model = NOTMAD(
            C.shape[-1],
            X.shape[-1],
            **self.subtype_kwargs
        )
        boot_train_idx, boot_val_idx = train_test_split(range(len(X)), test_size=0.2, random_state=1)
        C_train, C_val = C[boot_train_idx], C[boot_val_idx]
        X_train, X_val = X[boot_train_idx], X[boot_val_idx]
        train_dataset = model.dataloader(C_train, X_train, batch_size=10)
        val_dataset = model.dataloader(C_val, X_val, batch_size=10)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=f'checkpoints/dag_wrapper/bootstrap_{np.mean(X_train)}',  # hacky way to get unique dir
            filename='{epoch}-{val_loss:.2f}'
        )
        es_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.01,
            patience=5,
            verbose=True,
            mode="min"
        )
        self.trainer = GraphTrainer(
            max_epochs=100,
            accelerator='auto',
            devices=1,
            callbacks=[es_callback, checkpoint_callback]
        )
        self.trainer.fit(model, train_dataset, val_dataset)

        # Get best checkpoint by val_loss, test and save
        best_checkpoint = torch.load(checkpoint_callback.best_model_path)
        model.load_state_dict(best_checkpoint['state_dict'])
        self.model = model
        return self

    def predict(self, C):
        betas = self.trainer.predict_params(
            self.model,
            self.model.dataloader(C, np.zeros((len(C), self.p, self.p)), batch_size=10),
            project_to_dag=True,
        )
        mus = np.zeros((len(C), self.p))
        return betas, mus

    def mses(self, C, X):
        betas, _ = self.predict(C)
        X_preds = graph_utils.dag_pred_np(X, betas)
        residuals = X - X_preds
        mses = (residuals ** 2).mean(axis=-1)
        return mses


if __name__ == '__main__':
    C = np.random.normal(size=(100, 10))
    X = np.random.normal(size=(100, 10))
    for wrapper in [
        ContextualizedNeighborhoodSelectionWrapper,
        ContextualizedMarkovGraphWrapper,
        ContextualizedCorrelationWrapper,
        ContextualizedBayesianNetworksWrapper
    ]:
        model = wrapper().fit(C, X)
        model.predict(C)
        print(model.mses(C, X).mean())