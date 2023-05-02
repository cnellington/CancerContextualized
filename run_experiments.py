import torch
torch.set_float32_matmul_precision('medium')

from experiments import NeighborhoodExperiment, CorrelationExperiment, MarkovExperiment, BayesianExperiment
from dataloader import DEFAULT_DATA_STATE

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
# 'single_hallmark': None,            # reduce genes to a single hallmark set
# 'pretransform_norm': False,         # normalize before the feature transformation
# 'transform': None,                  # None, or transform full expression profiles using 'pca' to num_features or 'hallmark avg'
# 'feature_selection': 'population',  # select genetic features according to population variance (population) or weighted disease-specific variance (disease)
# 'disease_labels': True,             # include disease labels and primary site information in covariates
# 'features_to_covars': -1,           # -1 to ignore, or a positive number of genomic features to add to covariates
# 'remove_covar_features': False,     # remove genomic features that have been added to covariates
# 'covar_projection': -1,             # -1 to ignore, or provide a positive n_components to project SNV and SCNA covariates using PCA to reduce dimensionality
# 'no_test': True,                    # Create a new "test" set from the training data to avoid hyperparam tuning on the real test set
# 'dry_run': False,                   # Return a sma

base_dir = 'results/230428_100genes/'
data_state = DEFAULT_DATA_STATE.copy()
data_state.update({
    # 'dry_run': True,
    'num_features': 100,
    'covar_projection': 200,
    'transform': None,
    # 'features_to_covars': 10,
})
val_split = 0.2
n_bootstraps = 3

sanity = NeighborhoodExperiment(n_bootstraps=1, data_state={'dry_run': True})
sanity.run()
print('sanity check completed successfully <:)')

for fit_intercept in [True, False]:
    print('Correlation Experiment')
    experiment = CorrelationExperiment(
        base_dir=base_dir,
        n_bootstraps=n_bootstraps,
        val_split=val_split,
        fit_intercept=fit_intercept,
        data_state=data_state,
        # load_saved=True,  # only true to re-run results
    )
    experiment.run()

    print('Neighborhood Experiment')
    experiment = NeighborhoodExperiment(
        base_dir=base_dir,
        n_bootstraps=n_bootstraps,
        val_split=val_split,
        fit_intercept=fit_intercept,
        data_state=data_state,
    )
    experiment.run()

    print('Markov Experiment')
    experiment = MarkovExperiment(
        base_dir=base_dir,
        n_bootstraps=n_bootstraps,
        val_split=val_split,
        fit_intercept=fit_intercept,
        data_state=data_state,
    )
    experiment.run()

    # print('Bayesian Experiment')
    # experiment = BayesianExperiment(
    #     base_dir=base_dir,
    #     n_bootstraps=n_bootstraps,
    #     val_split=val_split,
    #     fit_intercept=fit_intercept,
    #     project_to_dag=True,
    #     data_state=data_state,
    # )
    # experiment.run()
    # exit()

print('Finished Successfully <:)')
# Run everything
# dry_run = False
# n_bootstraps = 3

# base_dir = 'results/230425_neighborhood'
# include_disease_labels = [True, False]
# val_splits = [0.0, 0.2]
# fit_intercepts = [False, True]
# for fit_intercept in fit_intercepts:
#     for val_split in val_splits:
#         for include_disease in include_disease_labels:
#             experiment = NeighborhoodExperiment(
#                 base_dir=base_dir,
#                 dry_run=dry_run,
#                 n_bootstraps=n_bootstraps,
#                 val_split=val_split,
#                 fit_intercept=fit_intercept,
#                 include_disease_labels=include_disease,
#             )
#             experiment.run()
#
#
# base_dir = 'results/230425_markov'
# include_disease_labels = [True, False]
# val_splits = [0.0, 0.2]
# fit_intercepts = [False, True]
# for fit_intercept in fit_intercepts:
#     for val_split in val_splits:
#         for include_disease in include_disease_labels:
#             experiment = MarkovExperiment(
#                 base_dir=base_dir,
#                 dry_run=dry_run,
#                 n_bootstraps=n_bootstraps,
#                 val_split=val_split,
#                 fit_intercept=fit_intercept,
#                 include_disease_labels=include_disease,
#             )
#             experiment.run()


# base_dir = 'results/230425_correlation'
# include_disease_labels = [True]
# val_splits = [0.2]
# fit_intercepts = [False]
# for fit_intercept in fit_intercepts:
#     for val_split in val_splits:
#         for include_disease in include_disease_labels:
#             experiment = CorrelationExperiment(
#                 base_dir=base_dir,
#                 dry_run=dry_run,
#                 n_bootstraps=n_bootstraps,
#                 val_split=val_split,
#                 fit_intercept=fit_intercept,
#                 include_disease_labels=include_disease,
#             )
#             experiment.run()

# base_dir = 'results/230425_bayesian'
# include_disease_labels = [True, False]
# val_splits = [0.0, 0.2]
# project_to_dags = [False, True]
# for project_to_dag in project_to_dags:  # no intercept, test projection
#     for val_split in val_splits:
#         for include_disease in include_disease_labels:
#             experiment = BayesianExperiment(
#                 base_dir=base_dir,
#                 dry_run=dry_run,
#                 n_bootstraps=n_bootstraps,
#                 val_split=val_split,
#                 project_to_dag=project_to_dag,
#                 include_disease_labels=include_disease,
#             )
#             experiment.run()
