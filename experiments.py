import os
import time
import argparse
import dill as pickle
import numpy as np
import pandas as pd
import seaborn as sns
import json
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
from dataloader import load_data, load_toy_data, DEFAULT_DATA_STATE, HALLMARK_GENES


experiments = {
    "neighborhood": (
        NeighborhoodSelectionSKLearn,
        ContextualizedNeighborhoodSelectionWrapper,
    ),
    "markov": (MarkovNetwork, ContextualizedMarkovGraphWrapper),
    "correlation": (CorrelationNetworkSKLearn, ContextualizedCorrelationWrapper),
    "bayesian": (BayesianNetwork, ContextualizedBayesianNetworksWrapper),
}


class NeighborhoodExperiment:
    def __init__(
        self,
        base_dir=f"results/test/",
        model="neighborhood",
        data_state=DEFAULT_DATA_STATE,
        n_bootstraps=3,
        start_bootstrap=0,
        val_split=0.2,
        max_epochs=100,
        fit_intercept=False,
        save_models=True,
        save_contextualized_networks=False,
        save_all_networks=False,
        save_all_bootstraps=False,
        load_saved=False,
        verbose=False,
    ):
        self.savedir = os.path.join(
            base_dir,
            f'{model}\
-fit_intercept={fit_intercept}\
-val_split={val_split}\
-max_epochs={max_epochs}\
-n_bootstraps={start_bootstrap}-{start_bootstrap + n_bootstraps - 1}\
-dryrun={data_state["dry_run"]}\
-test={data_state["test"]}\
-disease_test={data_state["disease_test"]}',
        )
        os.makedirs(self.savedir, exist_ok=True)
        self.disease_test = data_state["disease_test"]
        self.baseline_class = lambda: NeighborhoodSelection(
            fit_intercept=fit_intercept, verbose=verbose
        )
        self.contextualized_class = lambda: ContextualizedNeighborhoodSelectionWrapper(
            fit_intercept=fit_intercept, verbose=verbose
        )
        self.data_state = data_state
        self.n_bootstraps = n_bootstraps
        self.start_bootstrap = start_bootstrap
        self.val_split = val_split
        self.fit_intercept = fit_intercept
        self.save_models = save_models
        self.save_contextualized_networks = save_contextualized_networks
        self.save_all_networks = save_all_networks
        self.save_all_bootstraps = save_all_bootstraps
        self.load_saved = load_saved
        # self.savedir += '-'.join([f'{k}={v}' for k, v in data_state.items()])  # dirname too long
        with open(f"{self.savedir}/data_state.json", "w") as f:
            json.dump(self.data_state, f)
        (
            self.C_train,
            self.C_test,
            self.X_train,
            self.X_test,
            self.labels_train,
            self.labels_test,
            self.ids_train,
            self.ids_test,
            self.col_names,
        ) = load_data(**self.data_state)
        print(self.C_train.shape, self.C_test.shape)
        self.network_cols = np.array(
            [[f"{col1}-{col2}" for col2 in self.col_names] for col1 in self.col_names]
        ).reshape((len(self.col_names) ** 2,))
        self.mse_df_rows = []
        self.mse_df_cols = ["Bootstrap", "sample_id", "Model", "Set", "Disease", "MSE"]
        self.mse_by_feature_rows = []
        self.mse_by_feature_cols = [
            "Bootstrap",
            "sample_id",
            "Model",
            "Set",
            "Disease",
        ] + self.col_names.tolist()

    def write_rows(self, boot_i, tcga_ids, method, split, diseases, all_residuals):
        for disease, tcga_id, residuals in zip(diseases, tcga_ids, all_residuals):
            self.mse_df_rows.append(
                [boot_i, tcga_id, method, split, disease, residuals.mean()]
            )
        self.mse_by_feature_rows.append(
            [boot_i, "ID", method, split, "disease"]
            + all_residuals.mean(axis=0).flatten().tolist()
        )

    def get_mses(self, X_preds, X_true):
        assert X_preds.shape == X_true.shape
        return (X_preds - X_true) ** 2

    def run_model(
        self, model_name, start_bootstrap, n_bootstraps, val_split, save_networks=False, load_saved=False
    ):
        all_train_preds = []
        all_test_preds = []
        all_train_networks = []
        all_test_networks = []
        all_models = []
        for boot_i in range(start_bootstrap, start_bootstrap + n_bootstraps):
            np.random.seed(boot_i)
            if n_bootstraps > 1 and start_bootstrap != 0:
                boot_idx = np.random.choice(
                    len(self.C_train), size=len(self.C_train), replace=True
                )
            else:
                boot_idx = np.arange(len(self.C_train))
            C_boot, X_boot = self.C_train[boot_idx], self.X_train[boot_idx]
            if model_name == "Population":
                C_boot, C_train, C_test = (
                    np.zeros_like(C_boot),
                    np.zeros_like(self.C_train),
                    np.zeros_like(self.C_test),
                )
                model = self.baseline_class()
            elif model_name == "Cluster-specific":
                kmeans = KMeans(
                    n_clusters=len(np.unique(self.labels_train)), random_state=0
                ).fit(C_boot)
                C_boot, C_train, C_test = (
                    kmeans.predict(C_boot),
                    kmeans.predict(self.C_train),
                    kmeans.predict(self.C_test),
                )
                model = GroupedNetworks(self.baseline_class)
            elif model_name == "Disease-specific":
                C_boot, C_train, C_test = (
                    self.labels_train[boot_idx],
                    self.labels_train,
                    self.labels_test,
                )
                model = GroupedNetworks(self.baseline_class)
            else:
                C_train, C_test = self.C_train, self.C_test
                model = self.contextualized_class()

            # Learn a single correlation model representing the whole population
            print(f"Training {model_name} model")
            if load_saved:
                model = load(
                    f"{self.savedir}/saved_models/{model_name}_boot{boot_i}.pt"
                )
            else:
                model.fit(C_boot, X_boot, val_split=val_split)
                if self.save_models:
                    os.makedirs(f"{self.savedir}/saved_models/", exist_ok=True)
                    save(
                        model,
                        f"{self.savedir}/saved_models/{model_name}_boot{boot_i}.pt",
                    )
            all_models.append(model)
            boot_preds = model.predict(C_boot, X_boot)
            boot_mses = self.get_mses(boot_preds, X_boot)
            self.write_rows(
                boot_i,
                self.ids_train[boot_idx],
                model_name,
                "Train",
                self.labels_train[boot_idx],
                boot_mses,
            )
            train_preds = model.predict(C_train, self.X_train)
            train_networks, train_offsets = model.predict_networks(C_train)
            train_mses = self.get_mses(train_preds, self.X_train)
            all_train_preds.append(train_preds)
            all_train_networks.append(train_networks)
            self.write_rows(
                boot_i,
                self.ids_train,
                model_name,
                "Full Trainset",
                self.labels_train,
                train_mses,
            )
            test_preds = model.predict(C_test, self.X_test)
            test_networks, test_offsets = model.predict_networks(C_test)
            test_mses = self.get_mses(test_preds, self.X_test)
            all_test_preds.append(test_preds)
            all_test_networks.append(test_networks)
            self.write_rows(
                boot_i, self.ids_test, model_name, "Test", self.labels_test, test_mses
            )
            print(
                boot_mses.mean(),
                train_mses.mean(),
                test_mses.mean(),
            )
        X_train_pred = np.mean(all_train_preds, axis=0)
        train_mses = self.get_mses(X_train_pred, self.X_train)
        self.write_rows(
            "avg",
            self.ids_train,
            model_name,
            "Train (Bootstrapped)",
            self.labels_train,
            train_mses,
        )
        X_test_pred = np.mean(all_test_preds, axis=0)
        test_mses = self.get_mses(X_test_pred, self.X_test)
        self.write_rows(
            "avg",
            self.ids_test,
            model_name,
            "Test (Bootstrapped)",
            self.labels_test,
            test_mses,
        )
        if save_networks:
            self.export_networks(all_train_networks, all_test_networks, model_name)

    def process_networks(self, networks):
        # do any processing necessary before using models
        pass

    def export_networks(self, all_train_networks, all_test_networks, model_name=""):
        metadata_cols = ['sample_id', 'Set', 'Bootstrap']
        network_cols = np.array(
            [[f"{col1}-{col2}" for col2 in self.col_names] for col1 in self.col_names]
        ).reshape((len(self.col_names) ** 2,)).tolist()
        rows = []
        for i, (train_networks, test_networks) in enumerate(zip(all_train_networks, all_test_networks)): 
            train_networks = train_networks.reshape(len(train_networks), -1)
            test_networks = test_networks.reshape(len(test_networks), -1)
            # train_offsets = train_offsets.reshape(len(train_offsets), -1) 
            # test_offsets = test_offsets.reshape(len(test_offsets), -1) 
            train_metadata = np.array([self.ids_train, ['Train'] * len(train_networks), [i] * len(train_networks)]).T
            test_metadata = np.array([self.ids_test, ['Test'] * len(test_networks), [i] * len(test_networks)]).T
            rows.append(np.concatenate([train_metadata, train_networks], axis=1))
            rows.append(np.concatenate([test_metadata, test_networks], axis=1))
            # train_networks += train_networks_boot / len(all_models)
            # del train_networks_boot  # immediate cleanup to avoid OOM
            # test_networks += test_networks_boot / len(all_models)
            # del test_networks_boot  # immediate cleanup to avoid OOM
        all_networks = np.concatenate(rows, axis=0)
        all_networks_df = pd.DataFrame(all_networks, columns=metadata_cols + network_cols)
        if self.save_all_bootstraps:
            all_networks_df.to_csv(f"{self.savedir}/{model_name}-networks-allboots.csv", index=False)
        networks_df = all_networks_df.groupby(['sample_id', 'Set']).mean().reset_index().drop(columns='Bootstrap')
        networks_df.to_csv(f"{self.savedir}/{model_name}-networks.csv", index=False)

        reducer = umap.UMAP()
        w = reducer.fit_transform(networks_df.drop(columns=["sample_id", "Set"]).values)
        x = reducer.fit_transform(self.X_train)

        fig, ax = plt.subplots(figsize=(10, 10))
        colors = mpl.colormaps["tab20"].colors + mpl.colormaps["tab20b"].colors
        for disease, color in zip(np.unique(self.labels_train), colors):
            label_idx = self.labels_train == disease
            ax.scatter(
                x[label_idx, 0], x[label_idx, 1], label=disease, s=1.0, c=[color]
            )
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=14)
        plt.xlabel("UMAP 1", fontsize=18)
        plt.ylabel("UMAP 2", fontsize=18)
        plt.xticks([])
        plt.yticks([])
        plt.title("Transcriptomic Embedding", fontsize=22)
        plt.tight_layout()
        plt.savefig(f"{self.savedir}/transcriptomic_embedding.pdf", dpi=300)
        plt.clf()

        # %%
        fig, ax = plt.subplots(figsize=(10, 10))
        for disease, color in zip(np.unique(self.labels_train), colors):
            label_idx = (
                np.array(list(self.labels_train) + list(self.labels_test)) == disease
            )
            ax.scatter(
                w[label_idx, 0], w[label_idx, 1], label=disease, s=1.0, color=[color]
            )
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=14)
        plt.xlabel("UMAP 1", fontsize=18)
        plt.ylabel("UMAP 2", fontsize=18)
        plt.xticks([])
        plt.yticks([])
        plt.title("Network Embedding", fontsize=22)
        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{model_name}_network_embedding.pdf", dpi=300)
        plt.clf()

    def run(self):
        # Sets bootstrapped networks, bootstrapped mses
        self.run_model(
            "Population", self.start_bootstrap, self.n_bootstraps, self.val_split, load_saved=self.load_saved, save_networks=self.save_all_networks,
        )
        self.run_model(
            "Cluster-specific",
            self.start_bootstrap,
            self.n_bootstraps,
            self.val_split,
            load_saved=self.load_saved,
            save_networks=self.save_all_networks,
        )
        if self.disease_test is None:
            self.run_model(
                "Disease-specific",
                self.start_bootstrap,
                self.n_bootstraps,
                self.val_split,
                load_saved=self.load_saved,
                save_networks=self.save_all_networks,
            )
        self.run_model(
            "Contextualized",
            self.start_bootstrap,
            self.n_bootstraps,
            self.val_split,
            load_saved=self.load_saved,
            save_networks=self.save_contextualized_networks or self.save_all_networks,
        )
        self.mse_df = pd.DataFrame(data=self.mse_df_rows, columns=self.mse_df_cols)
        self.mse_df.to_csv(f"{self.savedir}/mse_df.csv", index=False)
        self.mse_by_feature_df = pd.DataFrame(
            data=self.mse_by_feature_rows, columns=self.mse_by_feature_cols
        )
        self.mse_by_feature_df.to_csv(f"{self.savedir}/mse_by_feature.csv", index=False)
        self.plot_mses("Train")
        self.plot_mses("Test")
        self.plot_mses("Full Trainset")
        self.plot_mses("Train (Bootstrapped)")
        self.plot_mses("Test (Bootstrapped)")
        total_mses = (
            self.mse_df.groupby(["Bootstrap", "Model", "Set"]).mean().reset_index()
        )
        total_mses = (
            total_mses.groupby(["Model", "Set"])
            .agg({"MSE": ["mean", "std"]})
            .reset_index()[["Model", "Set", "MSE"]]
        )
        total_mses.columns = [" ".join(col).strip(" ") for col in total_mses.columns]
        total_mses.to_csv(f"{self.savedir}/total_mses.csv", index=False)

    def plot_mses(self, set_label):
        # This inner function is used to create dataframes with the upper and lower bounds of the error bars for the plots.
        def remake_errorbars(mean_df, std_df, std_label, groupby=["Model", "Disease"]):
            # Calculate standard deviation for each group
            stds = std_df[std_df["Set"] == std_label].groupby(groupby).std().reset_index()
            # Calculate mean for each group
            means = mean_df.groupby(groupby).mean().reset_index()
            # Create upper bound of the error bar by adding standard deviation to the mean
            means_upper = means.copy()
            means_upper["MSE"] += stds["MSE"]
            # Create lower bound of the error bar by subtracting standard deviation from the mean
            means_lower = means.copy()
            means_lower["MSE"] -= stds["MSE"]
            # Combine means and their upper and lower bounds into a single dataframe
            ret = pd.concat([means, means_upper, means_lower], axis=0)
            return ret
        
        # Get population errors for plotting. 
        # For Train and Test, average all MSEs within each bootstrap, and use the standard deviation of the bootstraps as the error bar. 
        # For Train (Bootstrapped) and Test (Bootstrapped) the set was created by averaging bootstraps into a single prediction.
        # In this case, remake the error bars using the standard deviation of the Train or Test set bootstraps are the error bars.
        mses_by_disease = (
            self.mse_df.drop(columns="sample_id")
            .groupby(["Bootstrap", "Model", "Set"])
            .mean()
            .reset_index()
        )
        set_df = mses_by_disease[mses_by_disease["Set"] == set_label]
        if set_label == "Train (Bootstrapped)":
            plot_df = remake_errorbars(
                set_df, mses_by_disease, "Train", groupby=["Model"]
            )
        elif set_label == "Test (Bootstrapped)":
            plot_df = remake_errorbars(
                set_df, mses_by_disease, "Test", groupby=["Model"]
            )
        else:
            plot_df = set_df

        if self.disease_test is None:  # Remove Disease-specific when we don't have it
            order = [
                "Population",
                "Cluster-specific",
                "Disease-specific",
                "Contextualized",
            ]
            palette = ["lightblue", "deepskyblue", "royalblue", "orange"]
        else:
            order = ["Population", "Cluster-specific", "Contextualized"]
            palette = ["lightblue", "deepskyblue", "orange"]
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.barplot(
            plot_df,
            x="Model",
            y="MSE",
            order=order,
            palette=palette,
            errorbar="sd",
            capsize=0.05,
            ax=ax,
        )
        plt.xlim(-1, 4)
        plt.ylim(0, 2.5)
        ax.plot([-1, 5], [1, 1], linestyle="dashed", color="lightgrey")
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=14)
        ax.set_yticklabels([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], fontsize=14)
        plt.xlabel("Method", fontsize=18)
        plt.ylabel("MSE", fontsize=18)
        plt.title(f"{set_label} Errors", fontsize=22)
        plt.tight_layout()
        plt.savefig(f"{self.savedir}/mse_{set_label.lower()}.pdf", dpi=300)
        plt.clf()

        # Plot the contextualized model errors relative to the other models errors at the population level.
        # So if the Cluster-specific bar is at 0.56, then the Contextualized model average error is 0.56 * Cluster-specific error.
        # All basleines need to be below 1 for the contextualized model to be better than the baselines.
        # Contextualized will always have a relative error of 1, since contextualized is the reference
        plot_df["Relative MSE"] = (
            plot_df[plot_df["Model"] == "Contextualized"]["MSE"].mean() / plot_df["MSE"]
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.barplot(
            plot_df,
            x="Model",
            y="Relative MSE",
            order=order,
            palette=palette,
            errorbar="sd",
            capsize=0.05,
            ax=ax,
        )
        plt.xlim(-1, 4)
        plt.ylim(0, 2.5)
        ax.plot([-1, 5], [1, 1], linestyle="dashed", color="lightgrey")
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=14)
        ax.set_yticklabels([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], fontsize=14)
        plt.xlabel("Method", fontsize=18)
        plt.ylabel("Relative MSE", fontsize=18)
        plt.title(f"{set_label} Errors", fontsize=22)
        plt.tight_layout()
        plt.savefig(f"{self.savedir}/relative_mse_{set_label.lower()}.pdf", dpi=300)
        plt.clf()

        # Save the relative error values to quickly check the exact relative errors
        relative_df = (
            plot_df.groupby(["Model"])
            .agg({"Relative MSE": ["mean", "std"]})
            .reset_index()[["Model", "Relative MSE"]]
        )
        relative_df.columns = [" ".join(col).strip(" ") for col in relative_df.columns]
        relative_df.to_csv(
            f"{self.savedir}/relative_mses_{set_label.lower()}.csv", index=False
        )

        # Same as the population plot above, but for each disease type
        mses_by_disease = (
            self.mse_df.drop(columns="sample_id")
            .groupby(["Bootstrap", "Model", "Set", "Disease"])
            .mean()
            .reset_index()
        )
        set_df = mses_by_disease[mses_by_disease["Set"] == set_label]
        if set_label == "Train (Bootstrapped)":
            plot_df = remake_errorbars(
                set_df, mses_by_disease, "Train", groupby=["Model", "Disease"]
            )
        elif set_label == "Test (Bootstrapped)":
            plot_df = remake_errorbars(
                set_df, mses_by_disease, "Test", groupby=["Model", "Disease"]
            )
        else:
            plot_df = set_df
        n_diseases = len(plot_df["Disease"].unique())
        fig, ax = plt.subplots(figsize=(n_diseases + 5, 5))
        sns.barplot(
            plot_df,
            x="Disease",
            y="MSE",
            hue="Model",
            hue_order=order,
            palette=palette,
            errorbar="sd",
            capsize=0.05,
            ax=ax,
        )
        plt.xlim(-1, n_diseases)
        plt.ylim(0, 2.5)
        ax.plot(
            list(range(-1, n_diseases + 1)),
            [1] * (n_diseases + 2),
            linestyle="dashed",
            color="lightgrey",
        )

        # Show the number of training samples for each disease on the x-axis by remaking the labels with the disease training sample counts
        train_datapoints = (
            self.mse_df[
                (self.mse_df["Set"] == "Train")
                & (self.mse_df["Model"] == "Contextualized")
                & (self.mse_df["Bootstrap"] == 0)
            ]["Disease"]
            .value_counts()
            .sort_index()
        )
        train_disease_counts = {
            label: count for label, count in train_datapoints.iteritems()
        }
        test_datapoints = (
            self.mse_df[
                (self.mse_df["Set"] == "Test")
                & (self.mse_df["Model"] == "Contextualized")
                & (self.mse_df["Bootstrap"] == 0)
            ]["Disease"]
            .value_counts()
            .sort_index()
        )
        if "Train" in set_label:
            x_labels = [
                f"{label} ({count})" for label, count in train_datapoints.iteritems()
            ]
        else:
            x_labels = [
                f"{label} ({train_disease_counts[label] if label in train_disease_counts else 0})"
                for label, count in test_datapoints.iteritems()
            ]
        ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=14)
        ax.set_yticklabels([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], fontsize=14)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=14)
        plt.xlabel("Disease Type (# Training Samples)", fontsize=18)
        plt.ylabel("MSE", fontsize=18)
        plt.title(f"{set_label} Errors by Disease Type", fontsize=22)
        plt.tight_layout()
        plt.savefig(f"{self.savedir}/mse_by_disease_{set_label.lower()}.pdf", dpi=300)
        plt.clf()


class MarkovExperiment(NeighborhoodExperiment):
    def __init__(
        self,
        fit_intercept=False,
        verbose=False,
        **kwargs,
    ):
        super().__init__(model="markov", fit_intercept=fit_intercept, **kwargs)
        self.baseline_class = lambda: MarkovNetwork(
            fit_intercept=fit_intercept, verbose=verbose
        )
        self.contextualized_class = lambda: ContextualizedMarkovGraphWrapper(
            fit_intercept=fit_intercept, verbose=verbose
        )


class CorrelationExperiment(NeighborhoodExperiment):
    def __init__(
        self,
        fit_intercept=False,
        verbose=False,
        **kwargs,
    ):
        super().__init__(
            model="correlation", fit_intercept=fit_intercept, verbose=verbose, **kwargs
        )
        self.baseline_class = lambda: CorrelationNetworkSKLearn(
            fit_intercept=fit_intercept, verbose=verbose
        )
        self.contextualized_class = lambda: ContextualizedCorrelationWrapper(
            fit_intercept=fit_intercept, verbose=verbose
        )
        self.mse_by_feature_cols = [
            "Bootstrap",
            "sample_id",
            "Model",
            "Set",
            "Disease",
        ] + self.network_cols.tolist()

    def get_mses(self, X_preds, X_true):
        X_true = np.tile(np.expand_dims(X_true, axis=-1), (1, 1, X_true.shape[-1]))
        assert X_preds.shape == X_true.shape
        return (X_preds - X_true) ** 2


class BayesianExperiment(NeighborhoodExperiment):
    def __init__(
        self,
        fit_intercept=False,
        project_to_dag=False,
        verbose=False,
        **kwargs,
    ):
        super().__init__(
            model="bayesian", fit_intercept=False, verbose=verbose, **kwargs
        )
        self.baseline_class = lambda: BayesianNetwork(
            fit_intercept=False, project_to_dag=project_to_dag, verbose=verbose
        )
        self.contextualized_class = lambda: ContextualizedBayesianNetworksWrapper(
            fit_intercept=False, project_to_dag=project_to_dag, verbose=verbose
        )
        self.project_to_dag = project_to_dag
        # self.savedir += f"-project_to_dag={project_to_dag}"


def main(
    network_type="neighborhood",
    fit_intercept=True,
    val_split=0.2,
    n_bootstraps=30,
    start_bootstrap=0,
    save_models=False,
    load_saved=False,
    save_contextualized_networks=False,
    save_all_networks=False,
    save_all_bootstraps=False,
    result_dir="experiment",
    dryrun=False,
    num_features=50,
    covar_projection=-1,
    transform="pca",
    feature_selection="population",
    no_disease_labels=False,
    test=False,
    disease_test=None,
    project_to_dag=False,
    max_epochs=100,
):
    network_types = {
        "neighborhood": NeighborhoodExperiment,
        "markov": MarkovExperiment,
        "correlation": CorrelationExperiment,
        "bayesian": BayesianExperiment,
    }
    experiment_class = network_types[network_type]

    # Sanity check
    # sanity = experiment_class(n_bootstraps=2, data_state={'dry_run': True}, save_models=True, save_networks=True)
    # sanity.run()
    # sanity = experiment_class(n_bootstraps=2, data_state={'dry_run': True}, load_saved=True, save_networks=True)
    # sanity.run()
    # print('sanity check completed successfully <:)')

    # Setup global data and experiment parameters
    data_state = DEFAULT_DATA_STATE.copy()
    data_state.update(
        {
            "dry_run": dryrun,
            "num_features": num_features,
            "covar_projection": covar_projection,
            "transform": transform,
            "feature_selection": feature_selection,
            "test": test,
            "disease_test": disease_test,
            "disease_labels": not no_disease_labels,
        }
    )

    kwargs = {
        "base_dir": os.path.join("results", result_dir),
        "n_bootstraps": n_bootstraps,
        "start_bootstrap": start_bootstrap,
        "val_split": val_split,
        "fit_intercept": fit_intercept,
        "data_state": data_state,
        "save_models": save_models,
        "load_saved": load_saved,
        "save_contextualized_networks": save_contextualized_networks,
        "save_all_networks": save_all_networks,
        "save_all_bootstraps": save_all_bootstraps,
        "max_epochs": max_epochs,
    }
    if network_type == "bayesian":
        kwargs.update(
            {
                "project_to_dag": project_to_dag,
                "fit_intercept": False,
            }
        )
    experiment = experiment_class(**kwargs)
    print(f"Beginning {network_type} experiment")
    experiment.run()
    print("finished successfully <:)")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_type", type=str, default="neighborhood")
    parser.add_argument("--fit_intercept", default=False, action="store_true")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--n_bootstraps", type=int, default=2)
    parser.add_argument("--start_bootstrap", type=int, default=0)
    parser.add_argument("--save_models", default=False, action="store_true")
    parser.add_argument("--load_saved", default=False, action="store_true")
    parser.add_argument("--save_contextualized_networks", default=False, action="store_true")
    parser.add_argument("--save_all_networks", default=False, action="store_true")
    parser.add_argument("--save_all_bootstraps", default=False, action="store_true")
    parser.add_argument("--result_dir", type=str, default="experiment")
    parser.add_argument("--dryrun", default=False, action="store_true")
    parser.add_argument("--num_features", type=int, default=50)
    parser.add_argument("--covar_projection", type=int, default=200)
    parser.add_argument("--transform", type=str, default="pca")
    parser.add_argument("--feature_selection", type=str, default="population")
    parser.add_argument("--no_disease_labels", default=False, action="store_true")
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--disease_test", type=str, default=None)
    parser.add_argument("--project_to_dag", default=False, action="store_true")
    parser.add_argument("--max_epochs", type=int, default=100)
    args = parser.parse_args()
    # CLI overrides
    if args.disease_test is not None:
        args.test = True
    main(**vars(args))
