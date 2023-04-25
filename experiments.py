import os
import dill as pickle
import numpy as np
import pandas as pd
import seaborn as sns
import umap
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from contextualized import save, load
from models import (
    ContextualizedNeighborhoodSelectionWrapper,
    ContextualizedMarkovGraphWrapper,
    ContextualizedCorrelationWrapper,
    ContextualizedBayesianNetworksWrapper,
)
from baselines import (
    GroupedNetworks,
    NeighborhoodSelection,
    MarkovNetwork,
    CorrelationNetwork,
    BayesianNetwork,
    NeighborhoodSelectionSKLearn,
    CorrelationNetworkSKLearn,
)
from dataloader import load_data, load_toy_data


experiments = {
    'neighborhood': (NeighborhoodSelection, ContextualizedNeighborhoodSelectionWrapper),
    'markov': (MarkovNetwork, ContextualizedMarkovGraphWrapper),
    'correlation': (CorrelationNetworkSKLearn, ContextualizedCorrelationWrapper),
    'bayesian': (BayesianNetwork, ContextualizedBayesianNetworksWrapper),
}


class NeighborhoodExperiment:
    def __init__(
            self,
            base_dir=f'results/230424_neighborhoods',
            include_disease_labels=True,
            n_bootstraps=3,
            val_split=0.2,
            fit_intercept=False,
            load_saved=False,
            dry_run=False,
    ):
        self.baseline_class = lambda: NeighborhoodSelection(fit_intercept=fit_intercept)
        self.contextualized_class = lambda: ContextualizedNeighborhoodSelectionWrapper(fit_intercept=fit_intercept)
        self.include_disease_labels = include_disease_labels
        self.n_bootstraps = n_bootstraps
        self.val_split = val_split
        self.fit_intercept = fit_intercept
        self.load_saved = load_saved
        self.dry_run = dry_run
        self.savedir = f'{base_dir}-disease_labels={include_disease_labels}-val_split={val_split}-bootstraps={n_bootstraps}-intercept={fit_intercept}-dryrun={dry_run}'
        os.makedirs(self.savedir, exist_ok=True)
        self.mse_df_rows = []
        self.write_rows = lambda boot_i, tcga_ids, method, split, diseases, mses: self.mse_df_rows.extend(
            [(boot_i, tcga_id, method, split, disease, mse) for disease, tcga_id, mse in zip(diseases, tcga_ids, mses)])

        self.data_state = {
            'num_features': 100,
            'tumor_only': False,
            'hallmarks_only': False,
            'single_hallmark': None,
            'pretransform_norm': False,
            'transform': 'pca',
            'feature_selection': 'population',
            'disease_labels': include_disease_labels,
        }

        if dry_run:
            self.C_train, self.C_test, self.X_train, self.X_test, self.labels_train, self.labels_test, self.ids_train, self.ids_test, self.col_names = load_toy_data()
        else:
            self.C_train, self.C_test, self.X_train, self.X_test, self.labels_train, self.labels_test, self.ids_train, self.ids_test, self.col_names = load_data(
                **self.data_state)

    def run_model(self, model_name, n_bootstraps, val_split, save_networks=False, load_saved=False):
        all_train_preds = []
        all_test_preds = []
        all_models = []
        for boot_i in range(n_bootstraps):
            np.random.seed(boot_i)
            if n_bootstraps > 1:
                boot_idx = np.random.choice(len(self.C_train), size=len(self.C_train), replace=True)
            else:
                boot_idx = np.arange(len(self.C_train))
            C_boot, X_boot = self.C_train[boot_idx], self.X_train[boot_idx]
            if model_name == 'Population':
                C_boot, C_train, C_test = np.zeros_like(C_boot), np.zeros_like(self.C_train), np.zeros_like(self.C_test)
                model = self.baseline_class()
            elif model_name == 'Cluster-specific':
                kmeans = KMeans(n_clusters=len(np.unique(self.labels_train)), random_state=0).fit(C_boot)
                C_boot, C_train, C_test = kmeans.predict(C_boot), kmeans.predict(self.C_train), kmeans.predict(self.C_test)
                model = GroupedNetworks(self.baseline_class)
            elif model_name == 'Disease-specific':
                C_boot, C_train, C_test = self.labels_train[boot_idx], self.labels_train, self.labels_test
                model = GroupedNetworks(self.baseline_class)
            else:
                C_train, C_test = self.C_train, self.C_test
                model = self.contextualized_class()

            # Learn a single correlation model representing the whole population
            print(f'Training {model_name} model')
            if load_saved:
                model = load(f'{self.savedir}/saved_models/{model_name}_boot{boot_i}')
            else:
                model.fit(C_boot, X_boot, val_split=val_split)
                os.makedirs(f'{self.savedir}/saved_models/', exist_ok=True)
                save(model, f'{self.savedir}/saved_models/{model_name}_boot{boot_i}')
            all_models.append(model)
            boot_preds = model.predict(C_boot, X_boot)
            boot_mses = self.get_mses(boot_preds, X_boot)
            self.write_rows(boot_i, self.ids_train[boot_idx], model_name, 'Train', self.labels_train[boot_idx], boot_mses)
            train_preds = model.predict(C_train, self.X_train)
            train_mses = self.get_mses(train_preds, self.X_train)
            all_train_preds.append(train_preds)
            self.write_rows(boot_i, self.ids_train, model_name, 'Full Trainset', self.labels_train, train_mses)
            test_preds = model.predict(C_test, self.X_test)
            test_mses = self.get_mses(test_preds, self.X_test)
            all_test_preds.append(test_preds)
            self.write_rows(boot_i, self.ids_test, model_name, 'Test', self.labels_test, test_mses)
            print(
                boot_mses.mean(),
                train_mses.mean(),
                test_mses.mean(),
            )
        X_train_pred = np.mean(all_train_preds, axis=0)
        train_mses = self.get_mses(X_train_pred, self.X_train)
        self.write_rows("avg", self.ids_train, model_name, 'Train (Bootstrapped)', self.labels_train, train_mses)
        X_test_pred = np.mean(all_test_preds, axis=0)
        test_mses = self.get_mses(X_test_pred, self.X_test)
        self.write_rows("avg", self.ids_test, model_name, 'Test (Bootstrapped)', self.labels_test, test_mses)
        if save_networks:
            self.export_networks(all_models)

    def get_mses(self, X_preds, X_true):
        assert X_preds.shape == X_true.shape
        return np.mean((X_preds - X_true) ** 2, axis=1)

    def process_networks(self, networks):
        # do any processing necessary before using models
        pass

    def export_networks(self, all_models):
        all_train_networks = []
        all_test_networks = []
        for model in all_models:
            train_networks, _ = model.predict_networks(self.C_train)
            test_networks, _ = model.predict_networks(self.C_test)
            all_train_networks.append(train_networks)
            all_test_networks.append(test_networks)
        train_networks = np.mean(all_train_networks, axis=0)
        train_networks = train_networks.reshape((train_networks.shape[0], train_networks.shape[1] * train_networks.shape[2]))
        test_networks = np.mean(all_test_networks, axis=0)
        test_networks = test_networks.reshape((test_networks.shape[0], test_networks.shape[1] * test_networks.shape[2]))
        columns = np.array([[f'{col1}-{col2}' for col2 in self.col_names] for col1 in self.col_names]).reshape(
            (len(self.col_names) ** 2,))
        train_networks = np.concatenate([self.ids_train[:, np.newaxis], [['Train']] * len(train_networks), train_networks], axis=1)
        test_networks = np.concatenate([self.ids_test[:, np.newaxis], [['Test']] * len(test_networks), test_networks], axis=1)
        all_networks = np.concatenate([train_networks, test_networks], axis=0)
        columns = ['sample_id', 'Set'] + list(columns)
        networks_df = pd.DataFrame(data=all_networks, columns=columns)
        networks_df.to_csv(f'{self.savedir}/networks.csv', index=False)

        reducer = umap.UMAP()
        w = reducer.fit_transform(networks_df.drop(columns=['sample_id', 'Set']).values)
        x = reducer.fit_transform(self.X_train)

        fig, ax = plt.subplots(figsize=(10, 10))
        colors = mpl.colormaps['tab20'].colors + mpl.colormaps['tab20b'].colors
        for disease, color in zip(np.unique(self.labels_train), colors):
            label_idx = self.labels_train == disease
            ax.scatter(x[label_idx, 0], x[label_idx, 1], label=disease, s=1., c=[color])
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)
        plt.xlabel('UMAP 1', fontsize=18)
        plt.ylabel('UMAP 2', fontsize=18)
        plt.xticks([])
        plt.yticks([])
        plt.title('Transcriptomic Embedding', fontsize=22)
        plt.tight_layout()
        plt.savefig(f'{self.savedir}/transcriptomic_embedding.pdf')
        plt.clf()

        # %%
        fig, ax = plt.subplots(figsize=(10, 10))
        for disease, color in zip(np.unique(self.labels_train), colors):
            label_idx = np.array(list(self.labels_train) + list(self.labels_test)) == disease
            ax.scatter(w[label_idx, 0], w[label_idx, 1], label=disease, s=1., color=[color])
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)
        plt.xlabel('UMAP 1', fontsize=18)
        plt.ylabel('UMAP 2', fontsize=18)
        plt.xticks([])
        plt.yticks([])
        plt.title('Network Embedding', fontsize=22)
        plt.tight_layout()
        plt.savefig(f'{self.savedir}/network_embedding.pdf')
        plt.clf()

    def run(self):
        # Sets bootstrapped networks, bootstrapped mses
        self.run_model('Population', self.n_bootstraps, self.val_split, load_saved=self.load_saved)
        self.run_model('Cluster-specific', self.n_bootstraps, self.val_split, load_saved=self.load_saved)
        self.run_model('Disease-specific', self.n_bootstraps, self.val_split, load_saved=self.load_saved)
        self.run_model('Contextualized', self.n_bootstraps, self.val_split, load_saved=self.load_saved, save_networks=True)
        self.mse_df = pd.DataFrame(data=self.mse_df_rows, columns=['Bootstrap', 'sample_id', 'Model', 'Set', 'Disease', 'MSE'])
        self.mse_df.to_csv(f'{self.savedir}/mse_df.csv', index=False)
        self.plot_mses('Train')
        self.plot_mses('Test')
        self.plot_mses('Full Trainset')
        self.plot_mses('Train (Bootstrapped)')
        self.plot_mses('Test (Bootstrapped)')
        total_mses = self.mse_df.groupby(['Bootstrap', 'Model', 'Set']).mean().reset_index()
        total_mses = total_mses.groupby(['Model', 'Set']).agg({'MSE': ['mean', 'std']}).reset_index()[
            ['Model', 'Set', 'MSE']]
        total_mses.columns = [' '.join(col) for col in total_mses.columns]
        total_mses.to_csv(f'{self.savedir}/total_mses.csv', index=False)

    def plot_mses(self, set_label):
        # gets mses for each network
        datapoints = \
            self.mse_df[(self.mse_df['Set'] == 'Train') & (self.mse_df['Model'] == 'Contextualized') & (
                        self.mse_df['Bootstrap'] == 0)][
                'Disease'].value_counts().sort_index()
        mses_by_disease = self.mse_df.drop(columns='sample_id').groupby(
            ['Bootstrap', 'Model', 'Set', 'Disease']).mean().reset_index()
        plot_df = mses_by_disease[mses_by_disease['Set'] == set_label]

        def remake_errorbars(std_label):
            stds = mses_by_disease[mses_by_disease['Set'] == std_label].groupby(
                ['Model', 'Disease']).std().reset_index()
            means = plot_df.groupby(['Model', 'Disease']).mean().reset_index()
            plot_rows = []
            for (i, (model, disease, mean)), (_, (_, _, std)) in zip(means.iterrows(), stds.iterrows()):
                plot_rows.extend([
                    [model, disease, mean],
                    [model, disease, mean + std],
                    [model, disease, mean - std]
                ])
            return pd.DataFrame(data=plot_rows, columns=['Model', 'Disease', 'MSE'])

        if set_label == 'Train (Bootstrapped)':
            plot_df = remake_errorbars('Train')
        elif set_label == 'Test (Bootstrapped)':
            plot_df = remake_errorbars('Test')

        n_diseases = len(plot_df['Disease'].unique())
        fig, ax = plt.subplots(figsize=(n_diseases + 5, 5))
        sns.barplot(
            plot_df,
            x='Disease',
            y='MSE',
            hue='Model',
            hue_order=['Population', 'Cluster-specific', 'Disease-specific', 'Contextualized'],
            palette=['lightblue', 'deepskyblue', 'royalblue', 'orange'],
            errorbar='sd',
            capsize=0.05,
            ax=ax
        )
        plt.xlim(-1, n_diseases)
        plt.ylim(0, 2.5)
        ax.plot(list(range(-1, n_diseases + 1)), [1] * (n_diseases + 2), linestyle='dashed', color='lightgrey')

        labels = [f'{label} ({count})' for label, count in datapoints.iteritems()]
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=14)
        ax.set_yticklabels([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], fontsize=14)

        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)

        plt.xlabel('Disease Type (# Training Samples)', fontsize=18)
        plt.ylabel('MSE', fontsize=18)
        plt.title(f'{set_label} Errors by Disease Type', fontsize=22)
        plt.tight_layout()
        plt.savefig(f'{self.savedir}/mse_by_disease_{set_label.lower()}.pdf', dpi=300)
        plt.clf()


class MarkovExperiment(NeighborhoodExperiment):
    def __init__(
            self,
            base_dir=f'results/230424_markov',
            fit_intercept=False,
            **kwargs,
    ):
        super().__init__(base_dir=base_dir, fit_intercept=fit_intercept, **kwargs)
        self.baseline_class = lambda: MarkovNetwork(fit_intercept=fit_intercept)
        self.contextualized_class = lambda: ContextualizedMarkovGraphWrapper(fit_intercept=fit_intercept)


class CorrelationExperiment(NeighborhoodExperiment):
    def __init__(
            self,
            base_dir=f'results/230424_correlation',
            fit_intercept=False,
            **kwargs,
    ):
        super().__init__(base_dir=base_dir, fit_intercept=fit_intercept, **kwargs)
        self.baseline_class = lambda: CorrelationNetworkSKLearn(fit_intercept=fit_intercept)
        self.contextualized_class = lambda: ContextualizedCorrelationWrapper(fit_intercept=fit_intercept)

    def get_mses(self, X_preds, X_true):
        X_true = np.tile(np.expand_dims(X_true, axis=-1), (1, 1, X_true.shape[-1]))
        assert X_preds.shape == X_true.shape
        return np.mean((X_preds - X_true) ** 2, axis=(1, 2))


class BayesianExperiment(NeighborhoodExperiment):
    def __init__(
            self,
            base_dir=f'results/230424_bayesian',
            fit_intercept=False,
            project_to_dag=False,
            **kwargs,
    ):
        super().__init__(base_dir=base_dir, fit_intercept=False, **kwargs)
        self.baseline_class = lambda: BayesianNetwork(fit_intercept=False, project_to_dag=project_to_dag)
        self.contextualized_class = lambda: ContextualizedBayesianNetworksWrapper(fit_intercept=False, project_to_dag=project_to_dag)
        self.project_to_dag = project_to_dag
        self.savedir = f'{base_dir}-disease_labels={self.include_disease_labels}-val_split={self.val_split}-bootstraps={self.n_bootstraps}-intercept={self.fit_intercept}-project_to_dag={project_to_dag}-dryrun={self.dry_run}'
