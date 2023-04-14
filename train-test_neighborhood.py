import os
import dill as pickle
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from contextualized import save, load
from models import ContextualizedNeighborhoodSelectionWrapper
from baselines import NeighborhoodSelection, GroupedNetworks
from dataloader import load_data, load_toy_data

data_state = {
    'num_features': 50,
    'tumor_only': False,
    'hallmarks_only': False,
    'single_hallmark': None,
    'pretransform_norm': False,
    'transform': 'pca',
    'feature_selection': 'population',
}
C_train, C_test, X_train, X_test, labels_train, labels_test, _, _ = load_data(**data_state)
# C_train, C_test, X_train, X_test, labels_train, labels_test, _, _ = load_toy_data()

kmeans = KMeans(n_clusters=len(np.unique(labels_train)), random_state=0).fit(C_train)
cluster_labels_train, cluster_labels_test = kmeans.predict(C_train), kmeans.predict(C_test)


savedir = f'saved_models/testneighborhood'
os.makedirs(savedir, exist_ok=True)
n_bootstraps = 1

all_pop = []
all_cluster = []
all_oracle = []
for boot_i in range(n_bootstraps):
    np.random.seed(boot_i)
    if n_bootstraps > 1:
        boot_idx = np.random.choice(len(C_train), size=len(C_train), replace=True)
    else:
        boot_idx = np.arange(len(C_train))
    C_boot, X_boot = C_train[boot_idx], X_train[boot_idx]

    # Learn a single correlation model representing the whole population
    print('Training population model')
    neighborhood_constructor = lambda: NeighborhoodSelection(alpha=1., l1_ratio=1.0)
    population_model = neighborhood_constructor().fit(X_boot)
    all_pop.append(population_model)
    save(population_model, f'{savedir}/population_boot{boot_i}')
    print(population_model.mses(X_boot).mean(), population_model.mses(X_test).mean())

    # Learn context-clusters and learn a correlation model for each cluster
    print('Training clustered model')
    clustered_model = GroupedNetworks(neighborhood_constructor).fit(X_boot, cluster_labels_train[boot_idx])
    all_cluster.append(clustered_model)
    save(clustered_model, f'{savedir}/clustered_boot{boot_i}')
    print(clustered_model.mses(X_boot, cluster_labels_train[boot_idx]).mean(),
          clustered_model.mses(X_test, cluster_labels_test).mean())

    # Learn a correlation model for each tissue
    print('Training oracle model')
    oracle_model = GroupedNetworks(neighborhood_constructor).fit(X_boot, labels_train[boot_idx])
    all_oracle.append(oracle_model)
    save(oracle_model, f'{savedir}/oracle_boot{boot_i}')
    print(oracle_model.mses(X_boot, labels_train[boot_idx]).mean(), oracle_model.mses(X_test, labels_test).mean())

all_contextualized = []
for boot_i in range(n_bootstraps):
    np.random.seed(boot_i)
    if n_bootstraps > 1:
        boot_idx = np.random.choice(len(C_train), size=len(C_train), replace=True)
    else:
        boot_idx = np.arange(len(C_train))
    C_boot, X_boot = C_train[boot_idx], X_train[boot_idx]

    torch.manual_seed(boot_i)
    contextualized_model = ContextualizedNeighborhoodSelectionWrapper().fit(C_boot, X_boot)
    save(contextualized_model, f'{savedir}/contextualized_boot{boot_i}')
    all_contextualized.append(contextualized_model)
    train_mses = contextualized_model.mses(C_train, X_train)
    test_mses = contextualized_model.mses(C_test, X_test)
    print(train_mses.mean(), train_mses.std(), test_mses.mean(), test_mses.std())

# savedir = f'results/saved_models/markov-pca50'
# n_bootstraps = 3
#
# all_pop = []
# all_cluster = []
# all_oracle = []
# all_contextualized = []
# for boot_i in range(n_bootstraps):
#     all_pop.append(load(f'{savedir}/population_boot{boot_i}'))
#     all_cluster.append(load(f'{savedir}/clustered_boot{boot_i}'))
#     all_oracle.append(load(f'{savedir}/oracle_boot{boot_i}'))
#     all_contextualized.append(load(f'{savedir}/contextualized_boot{boot_i}'))


def get_mses(experiment, model, C, X, labels=None):
    if experiment in ['pop']:
        return model.mses(X)
    elif experiment in ['cluster', 'oracle']:
        return model.mses(X, labels)
    elif experiment in ['sample specific']:
        return model.mses(C, X)
    return None


experiments = {
    'pop': all_pop,
    'cluster': all_cluster,
    'oracle': all_oracle,
    'sample specific': all_contextualized,
}

mses = {experiment: np.ones((n_bootstraps, len(C_test))) for experiment in experiments.keys()}
train_mses = {experiment: np.ones((n_bootstraps, len(C_train))) for experiment in experiments.keys()}

for experiment, models in experiments.items():
    for boot_i, model in enumerate(models):
        labels = cluster_labels_test
        if experiment == 'oracle':
            labels = labels_test
        mses[experiment][boot_i] = get_mses(experiment, model, C_test, X_test, labels)

        train_labels = cluster_labels_train
        if experiment == 'oracle':
            train_labels = labels_train
        train_mses[experiment][boot_i] = get_mses(experiment, model, C_train, X_train, train_labels)
    print(
        f"{experiment} Train MSE: {np.round(np.mean(train_mses[experiment]), 3)} +/- {np.var(np.mean(train_mses[experiment], axis=1))}")
    print(
        f"{experiment} Test MSE: {np.round(np.mean(mses[experiment]), 3)} +/- {np.std(np.mean(mses[experiment], axis=1))}")


import matplotlib as mpl
import matplotlib.pyplot as plt


extra_colors = ['lightskyblue', 'yellow', 'tomato', 'rosybrown', 'goldenrod', 'indigo',
                'paleturquoise', 'antiquewhite', 'palegreen', 'olive', 'peachpuff',
               'darkseagreen', 'azure', 'moccasin', 'khaki', 'coral', 'palevioletred']


def plot_mse(mses, group_labels, experiment_names, experiment_colors=None, train_labels=None, savepath=None):
    # make barchart
    ignore_idx = group_labels != 'ignore'
    mses = mses[:, :, ignore_idx].copy()
    group_labels = group_labels[ignore_idx].copy()
    groups = np.unique(group_labels)
    xticks = np.arange(len(groups))
    bars = len(mses)
    width = 3/4
    offset = width / 2
    bar_width = width / bars
    if experiment_colors is None:
        experiment_colors = list(mpl.colors.TABLEAU_COLORS.keys())[:len(mses)]

    fig, ax = plt.subplots(figsize=(int(len(groups) * 1.5), 7))
    for i, group in enumerate(groups):
        group_idx = group_labels == group
        with_legend = i == len(groups) - 1
        for plot_i, (experiment_mses, name, color) in enumerate(zip(mses, experiment_names, experiment_colors)):
            mse_boots = np.mean(experiment_mses[:, group_idx], axis=1)
            mse = np.mean(mse_boots)
            std = np.std(mse_boots)
            if with_legend:
                ax.bar(xticks[i] - offset + plot_i*bar_width, mse,
                    yerr=std, edgecolor='black', label=name, width=bar_width, color=color)
            else:
                ax.bar(xticks[i] - offset + plot_i*bar_width, mse,
                    yerr=std, edgecolor='black', width=bar_width, color=color)
    x_labels = groups
    if train_labels is not None:
        x_labels = [f"{group} ({int(np.sum(train_labels == group))})" for group in groups]
    plt.xticks(
        range(len(x_labels)),
        labels=x_labels,
        fontsize=14,
        rotation=30,
        rotation_mode='anchor',
        verticalalignment='baseline',
        horizontalalignment='right'
    )
    plt.legend(fontsize=16, bbox_to_anchor=(1.0, 0.7))
    plt.ylabel("MSE (Held-out Patients)", fontsize=28)
    plt.yticks(fontsize=16)
    plt.ylim(0, 2.5)

    # if save:
    #     fig.subplots_adjust(bottom=0.2, right=0.75)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches='tight', pad_inches=1.0)
    # plt.show()

experiment_metrics = []
for experiment, experiment_mses in mses.items():
#     print(experiment)  # sanity check for experiment dict order
    experiment_metrics.append(experiment_mses)
experiment_metrics = np.array(experiment_metrics)

experiment_names = [
#     'Zeros',
    'Population Model',
    'Context-clustered Models',
    'Oracle-clustered Models',
#     'One Archetype & Tissue-averaged Sample-specific Models',
#     'One Archetype Sample-specific Models',
#     'Oracle-averaged Sample-specific Models',
    'Sample-specific Models',
]
experiment_colors = [
#     'grey',
    'lightblue',
    'deepskyblue',
    'royalblue',
#     'indianred',
#     'tomato',
    'orange',
#     'gold',
]

# test_labels[test_labels == 'Kidney'] = 'ignore'
plot_mse(
    experiment_metrics,
    labels_test,
    experiment_names,
    experiment_colors=experiment_colors,
    train_labels=labels_train,
    savepath=f'{savedir}/test_mses_by_disease.png',
)


np.save(f'{savedir}/train_mses.npy', train_mses)
np.save(f'{savedir}/test_mses.npy', mses)

w_train = [contextualized.predict(C_train)[0] for contextualized in all_contextualized]
w_train = np.mean(w_train, axis=0)
w_test = [contextualized.predict(C_test)[0] for contextualized in all_contextualized]
w_test = np.mean(w_test, axis=0)
np.save(f'{savedir}/markov_networks_train.npy', w_train)
np.save(f'{savedir}/markov_networks_test.npy', w_test)

# %%
import umap

reducer = umap.UMAP()
w = reducer.fit_transform(w_train.reshape((len(w_train), w_train.shape[-1] * w_train.shape[-2])))
x = reducer.fit_transform(X_train)
# %%
import matplotlib as mpl

colors = mpl.colormaps['tab20'].colors + mpl.colormaps['tab20b'].colors
for disease, color in zip(np.unique(labels_train), colors):
    label_idx = labels_train == disease
    plt.scatter(x[label_idx, 0], x[label_idx, 1], label=disease, s=1., c=[color])
plt.legend()
plt.savefig(f'{savedir}/transcriptomic_embedding.pdf')

# %%
for disease, color in zip(np.unique(labels_train), colors):
    label_idx = labels_train == disease
    plt.scatter(w[label_idx, 0], w[label_idx, 1], label=disease, s=1., color=[color])
plt.legend()
plt.savefig(f'{savedir}/neighborhood_networks_embedding.pdf')
# %%