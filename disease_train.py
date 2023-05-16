import os
import dill as pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from experiments import NeighborhoodExperiment, CorrelationExperiment, MarkovExperiment, BayesianExperiment
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


#%%
# experiment = 'neighborhood'
# n_bootstraps = 3
# val_split = 0.2
def run_experiment(
    experiment,
    n_bootstraps,
    val_split,
    load_saved,
    train_test_data,
    disease
):
    experiments = {
        'neighborhood': (NeighborhoodSelectionSKLearn, ContextualizedNeighborhoodSelectionWrapper),
        'markov': (MarkovNetwork, ContextualizedMarkovGraphWrapper),
        'correlation': (CorrelationNetworkSKLearn, ContextualizedCorrelationWrapper),
        'bayesian': (BayesianNetwork, ContextualizedBayesianNetworksWrapper),
    }

    C_train, C_test, X_train, X_test, ids_train, ids_test, labels_train, labels_test, col_names = \
        train_test_data['C_train'], train_test_data['C_test'], train_test_data['X_train'], train_test_data['X_test'], train_test_data['tcga_ids_train'], train_test_data['tcga_ids_test'], train_test_data['labels_train'], train_test_data['labels_test'], train_test_data['col_names']
    # not sure what val_split is as of now
    baseline_class, contextualized_class = experiments[experiment]
    savedir = f'results/saved_models/{disease}-{experiment}-val_split={val_split}-bootstraps={n_bootstraps}'
    os.makedirs(savedir, exist_ok=True)

    mse_df_rows = []
    write_rows = lambda boot_i, tcga_ids, method, split, diseases, mses: mse_df_rows.extend([(boot_i, tcga_id, method, split, disease, mse) for disease, tcga_id, mse in zip(diseases, tcga_ids, mses)])

    all_pop = []
    all_cluster = []
    all_oracle = []
    all_contextualized = []
    
    # number of bootstraps
    for boot_i in range(n_bootstraps):
        np.random.seed(boot_i)
        if n_bootstraps > 1:
            boot_idx = np.random.choice(len(C_train), size=len(C_train), replace=True)
        else:
            boot_idx = np.arange(len(C_train))
        C_boot, X_boot = C_train[boot_idx], X_train[boot_idx] # disrupt the original order of the data or validation data

        # Learn a single correlation model representing the whole population
        print('Training population model')
        if load_saved:
            population_model = load(f'{savedir}/population_boot{boot_i}')
        else:  # added C_boot as an parameter 
            population_model = baseline_class().fit(np.zeros_like(C_boot), X_boot, val_split=val_split)
            save(population_model, f'{savedir}/population_boot{boot_i}')
        all_pop.append(population_model)
        boot_mses = population_model.mses(np.zeros_like(C_boot), X_boot)
        write_rows(boot_i, ids_train[boot_idx], 'Population', 'Train', labels_train[boot_idx], boot_mses)
        train_mses = population_model.mses(np.zeros_like(C_train), X_train)
        write_rows(boot_i, ids_train, 'Population', 'Full Trainset', labels_train, train_mses)
        test_mses = population_model.mses(np.zeros_like(C_test), X_test)
        write_rows(boot_i, ids_test, 'Population', 'Test', labels_test, test_mses)
        print(
            boot_mses.mean(),
            train_mses.mean(),
            test_mses.mean(),
        )

        # Learn context-clusters and learn a correlation model for each cluster
        print('Training clustered model')
        kmeans = KMeans(n_clusters=len(np.unique(labels_train)), random_state=0).fit(C_boot)
        cluster_labels_boot, cluster_labels_train, cluster_labels_test = kmeans.predict(C_boot), kmeans.predict(C_train), kmeans.predict(C_test)
        if load_saved:
            clustered_model = load(f'{savedir}/clustered_boot{boot_i}')
        else:
            clustered_model = GroupedNetworks(baseline_class).fit(cluster_labels_boot, X_boot)
            save(clustered_model, f'{savedir}/clustered_boot{boot_i}')
        all_cluster.append(clustered_model)
        boot_mses = clustered_model.mses(cluster_labels_boot, X_boot)
        write_rows(boot_i, ids_train[boot_idx], 'Cluster-specific', 'Train', labels_train[boot_idx], boot_mses)
        train_mses = clustered_model.mses(cluster_labels_train, X_train)
        write_rows(boot_i, ids_train, 'Cluster-specific', 'Full Trainset', labels_train, train_mses)
        test_mses = clustered_model.mses(cluster_labels_test, X_test)
        write_rows(boot_i, ids_test, 'Cluster-specific', 'Test', labels_test, test_mses)
        print(
            boot_mses.mean(),
            train_mses.mean(),
            test_mses.mean(),
        )

        # Learn a correlation model for each tissue
        # print('Training disease model')
        # if load_saved:
        #     oracle_model = load(f'{savedir}/disease_boot{boot_i}')
        # else:
        #     oracle_model = GroupedNetworks(baseline_class).fit(X_boot, labels_train[boot_idx], val_split=val_split)
        #     save(oracle_model, f'{savedir}/disease_boot{boot_i}')
        # all_oracle.append(oracle_model)
        # boot_mses = oracle_model.mses(X_boot, labels_train[boot_idx])
        # write_rows(boot_i, ids_train[boot_idx], 'Disease-specific', 'Train', labels_train[boot_idx], boot_mses)
        # train_mses = oracle_model.mses(X_train, labels_train)
        # write_rows(boot_i, ids_train, 'Disease-specific', 'Full Trainset', labels_train, train_mses)
        # test_mses = oracle_model.mses(X_test, labels_test)
        # write_rows(boot_i, ids_test, 'Disease-specific', 'Test', labels_test, test_mses)
        # print(
        #     boot_mses.mean(),
        #     train_mses.mean(),
        #     test_mses.mean(),
        # )

        print('Training contextualized model')
        torch.manual_seed(boot_i)
        if load_saved:
            contextualized_model = load(f'{savedir}/contextualized_boot{boot_i}')
        else:
            contextualized_model = contextualized_class().fit(C_boot, X_boot, val_split=val_split)
            save(contextualized_model, f'{savedir}/contextualized_boot{boot_i}')
        all_contextualized.append(contextualized_model)
        boot_mses = contextualized_model.mses(C_boot, X_boot)
        write_rows(boot_i, ids_train[boot_idx], 'Contextualized', 'Train', labels_train[boot_idx], boot_mses)
        train_mses = contextualized_model.mses(C_train, X_train)
        write_rows(boot_i, ids_train, 'Contextualized', 'Full Trainset', labels_train, train_mses)
        test_mses = contextualized_model.mses(C_test, X_test)
        write_rows(boot_i, ids_test, 'Contextualized', 'Test', labels_test, test_mses)
        print(
            boot_mses.mean(),
            train_mses.mean(),
            test_mses.mean(),
        )

    mse_df = pd.DataFrame(data=mse_df_rows, columns=['Bootstrap', 'sample_id', 'Model', 'Set', 'Disease', 'MSE'])
    mse_df.to_csv(f'{savedir}/mse_df.csv', index=False)

    def plot_mse_by_disease(mse_df, set_label):
        if set_label != 'Test':
            datapoints = \
            mse_df[(mse_df['Set'] == 'Train') & (mse_df['Model'] == 'Contextualized') & (mse_df['Bootstrap'] == 0)][
                'Disease'].value_counts().sort_index()
        else:
            datapoints = \
            mse_df[(mse_df['Set'] == 'Test') & (mse_df['Model'] == 'Contextualized') & (mse_df['Bootstrap'] == 0)][
                'Disease'].value_counts().sort_index()
        mses_by_disease = mse_df.drop(columns='sample_id').groupby(['Bootstrap', 'Model', 'Set', 'Disease']).mean().reset_index()
        plot_df = mses_by_disease[mses_by_disease['Set'] == set_label]

        n_diseases = len(plot_df['Disease'].unique())
        fig, ax = plt.subplots(figsize=(n_diseases + 5, 5))
        sns.barplot(
            plot_df,
            x='Disease',
            y='MSE',
            hue='Model',
            hue_order=['Population', 'Cluster-specific', 'Contextualized'],
            palette = ['lightblue', 'deepskyblue', 'orange'],
            errorbar='sd',
            capsize=0.05,
        #     edgecolor='black',
            ax=ax
        )
        plt.xlim(-1, n_diseases)
        plt.ylim(0, 3)
        ax.plot(list(range(-1, n_diseases + 1)), [1] * (n_diseases + 2), linestyle='dashed', color='lightgrey')

        if set_label != 'Test':
            labels = [f'{label} ({count})' for label, count in datapoints.iteritems()]
            fontsize = 18
        else:
            label = plot_df['Disease'].unique()[0]
            labels = [f'{label} ({datapoints[datapoints.index == label].values[0]})']
            fontsize = 10
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=14)
        #ax.set_yticklabels([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], fontsize=14)

        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)

        
        plt.xlabel('Disease Type (# Training Samples)', fontsize=fontsize)
        plt.ylabel('MSE', fontsize=fontsize)
        plt.title(f'{set_label} Errors by Disease Type (Disease Specific CV)', fontsize=fontsize)
        plt.tight_layout()

        plt.savefig(f'{savedir}/mse_by_disease_{set_label.lower()}.pdf', dpi=300)
        plt.clf()

    plot_mse_by_disease(mse_df, 'Train')
    plot_mse_by_disease(mse_df, 'Full Trainset')
    plot_mse_by_disease(mse_df, 'Test')

    mses_by_disease = mse_df.drop(columns='sample_id').groupby(['Bootstrap', 'Model', 'Set', 'Disease']).mean().reset_index()
    test_df = mses_by_disease[mses_by_disease['Set'] == 'Test']

    total_mses = mse_df.groupby(['Bootstrap', 'Model', 'Set']).mean().reset_index()
    total_mses = total_mses.groupby(['Model', 'Set']).agg({'MSE': ['mean', 'std']}).reset_index()[['Model', 'Set', 'MSE']]
    total_mses.columns = [' '.join(col) for col in total_mses.columns]
    total_mses.to_csv(f'{savedir}/total_mses.csv', index=False)

    return mse_df, test_df
    # w_train = np.array([contextualized.predict(C_train)[0] for contextualized in all_contextualized])
    # w_train = np.mean(w_train, axis=0)
    # w_train = w_train.reshape((w_train.shape[0], w_train.shape[1] * w_train.shape[2]))
    # w_test = np.array([contextualized.predict(C_test)[0] for contextualized in all_contextualized])
    # w_test = np.mean(w_test, axis=0)
    # w_test = w_test.reshape((w_test.shape[0], w_test.shape[1] * w_test.shape[2]))
    # columns = np.array([[f'{col1}-{col2}' for col2 in col_names] for col1 in col_names]).reshape((len(col_names) ** 2,))
    # train_networks = np.concatenate([ids_train[:, np.newaxis], [['Train']] * len(w_train), w_train], axis=1)
    # test_networks = np.concatenate([ids_test[:, np.newaxis], [['Test']] * len(w_test),  w_test], axis=1)
    # all_networks = np.concatenate([train_networks, test_networks], axis=0)
    # columns = ['sample_id', 'Set'] + list(columns)
    # networks_df = pd.DataFrame(data=all_networks, columns=columns)
    # networks_df.to_csv(f'{savedir}/networks.csv', index=False)

    # # %%
    # import umap

    # reducer = umap.UMAP()
    # w = reducer.fit_transform(networks_df.drop(columns=['sample_id', 'Set']).values)
    # x = reducer.fit_transform(X_train)
    # # %%
    # import matplotlib as mpl

    # fig, ax = plt.subplots(figsize=(10, 10))
    # colors = mpl.colormaps['tab20'].colors + mpl.colormaps['tab20b'].colors
    # for disease, color in zip(np.unique(labels_train), colors):
    #     label_idx = labels_train == disease
    #     ax.scatter(x[label_idx, 0], x[label_idx, 1], label=disease, s=1., c=[color])
    # ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)
    # plt.xlabel('UMAP 1', fontsize=18)
    # plt.ylabel('UMAP 2', fontsize=18)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Transcriptomic Embedding', fontsize=22)
    # plt.tight_layout()
    # plt.savefig(f'{savedir}/transcriptomic_embedding.pdf')
    # plt.clf()

    # # %%
    # fig, ax = plt.subplots(figsize=(10, 10))
    # for disease, color in zip(np.unique(labels_train), colors):
    #     label_idx = np.array(list(labels_train) + list(labels_test)) == disease
    #     ax.scatter(w[label_idx, 0], w[label_idx, 1], label=disease, s=1., color=[color])
    # ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)
    # plt.xlabel('UMAP 1', fontsize=18)
    # plt.ylabel('UMAP 2', fontsize=18)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Network Embedding', fontsize=22)
    # plt.tight_layout()
    # plt.savefig(f'{savedir}/network_embedding.pdf')
    # plt.clf()
    # %%


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--experiment', type=str, default='neighborhood')
#     parser.add_argument('--disease_labels', type=str, default='True')
#     parser.add_argument('--bootstraps', type=int, default=3)
#     parser.add_argument('--val_split', type=float, default=0.2)
#     parser.add_argument('--load_saved', default=False, action='store_true')
#     parser.add_argument('--dry_run', default=False, action='store_true')
#     experiment = parser.parse_args().experiment
#     include_disease_labels = parser.parse_args().disease_labels == 'True'
#     n_bootstraps = parser.parse_args().bootstraps
#     val_split = parser.parse_args().val_split
#     load_saved = parser.parse_args().load_saved
#     dry_run = parser.parse_args().dry_run
#     run_experiment(experiment, include_disease_labels, n_bootstraps, val_split, load_saved, dry_run)

    # Run everything
    # experiments = ['neighborhood', 'correlation', 'markov', 'bayesian']
    # include_disease_labels = [True, False]
    # n_bootstraps = 3
    # val_splits = [0.0, 0.2]
    # load_saved = False
    # dry_run = False
    # for experiment in experiments:
    #     for val_split in val_splits:
    #         for include_disease in include_disease_labels:
    #             run_experiment(experiment, include_disease, n_bootstraps, val_split, load_saved, dry_run)
