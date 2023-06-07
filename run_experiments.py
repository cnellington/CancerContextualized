import numpy as np
import torch
import time

torch.set_float32_matmul_precision('medium')

from experiments import NeighborhoodExperiment, CorrelationExperiment, MarkovExperiment, BayesianExperiment
from dataloader import DEFAULT_DATA_STATE, HALLMARK_GENES
contextual_genes = np.loadtxt('data/contextual_genes_sorted.txt', dtype=str).tolist()

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--experiment', type=str, default='neighborhood')
# parser.add_argument('--disease_labels', type=str, default='True')
# parser.add_argument('--bootstraps', type=int, default=3)
# parser.add_argument('--val_split', type=float, default=0.2)
# parser.add_argument('--load_saved', default=False, action='store_true')
# parser.add_argument('--dry_run', default=False, action='store_true')
# experiment = parser.parse_args().experiment
# include_disease_labels = parser.parse_args().disease_labels == 'True'
# n_bootstraps = parser.parse_args().bootstraps
# val_split = parser.parse_args().val_split
# load_saved = parser.parse_args().load_saved
# dry_run = parser.parse_args().dry_run
# run_experiment(experiment, include_disease_labels, n_bootstraps, val_split, load_saved, dry_run)

# 'num_features': 50,                 # number of expression features to consider
# 'tumor_only': False,                # remove healthy normal tissue samples if True
# 'hallmarks_only': False,            # reduce COSMIC genes to only the intersection between COSMIC and Hallmarks
# 'gene_list': None,                  # only use genes in this list
# 'pretransform_norm': False,         # normalize before the feature transformation
# 'transform': None,                  # None, or transform full expression profiles using 'pca' to num_features or 'hallmark avg'
# 'feature_selection': 'population',  # select genetic features according to population variance (population) or weighted disease-specific variance (disease)
# 'disease_labels': True,             # include disease labels and primary site information in covariates
# 'features_to_covars': -1,           # -1 to ignore, or a positive number of genomic features to add to covariates
# 'remove_covar_features': False,     # remove genomic features that have been added to covariates
# 'covar_projection': -1,             # -1 to ignore, or provide a positive n_components to project SNV and SCNA covariates using PCA to reduce dimensionality
# 'no_test': True,                    # Create a new "test" set from the training data to avoid hyperparam tuning on the real test set
# 'dry_run': False,                   # Return a sma


contextual_genes = np.loadtxt('data/contextual_genes_sorted.txt', dtype=str).tolist()
gene_lists = HALLMARK_GENES.copy()
gene_lists['Contextual Metagenes'] = contextual_genes

# Setup global data and experiment parameters
data_state = DEFAULT_DATA_STATE.copy()
data_state.update({
    # 'dry_run': True,
    'num_features': 50,
    'covar_projection': 200,
    # 'gene_list': gene_list,
    'transform': 'pca',
    'feature_selection': None,
    'no_test': False,
})
val_split = 0.2
n_bootstraps = 30
save_models = False
save_networks = True
base_dir = f'results/230518_metagenes_20boots/'

# Sanity check
sanity = NeighborhoodExperiment(n_bootstraps=2, data_state={'dry_run': True}, save_models=False, save_networks=True)
sanity.run()
sanity = CorrelationExperiment(n_bootstraps=2, data_state={'dry_run': True}, save_models=False, save_networks=True)
sanity.run()
print('sanity check completed successfully <:)')

# Run experiments
for fit_intercept in [True, False]:
    print('Neighborhood Experiment', fit_intercept)
    experiment = NeighborhoodExperiment(
        base_dir=base_dir,
        n_bootstraps=n_bootstraps,
        val_split=val_split,
        fit_intercept=fit_intercept,
        data_state=data_state,
        save_models=save_models,
        save_networks=save_networks,
    )
    experiment.run()

    print('Correlation Experiment', fit_intercept)
    experiment = CorrelationExperiment(
        base_dir=base_dir,
        n_bootstraps=n_bootstraps,
        val_split=val_split,
        fit_intercept=fit_intercept,
        data_state=data_state,
        save_models=save_models,
        save_networks=save_networks,
    )
    experiment.run()

    print('Markov Experiment', fit_intercept)
    experiment = MarkovExperiment(
        base_dir=base_dir,
        n_bootstraps=n_bootstraps,
        val_split=val_split,
        fit_intercept=fit_intercept,
        data_state=data_state,
        save_models=save_models,
        save_networks=save_networks,
    )
    experiment.run()

print('Finished Successfully <:)')
