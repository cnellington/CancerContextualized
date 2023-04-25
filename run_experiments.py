from experiments import NeighborhoodExperiment, CorrelationExperiment, MarkovExperiment, BayesianExperiment

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

# Run everything
dry_run = True
n_bootstraps = 1

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
#
#
# base_dir = 'results/230425_correlation'
# include_disease_labels = [True, False]
# val_splits = [0.0, 0.2]
# fit_intercepts = [False, True]
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

base_dir = 'results/230425_bayesian'
include_disease_labels = [True, False]
val_splits = [0.0, 0.2]
project_to_dags = [False, True]
for project_to_dag in project_to_dags:
    for val_split in val_splits:
        for include_disease in include_disease_labels:
            experiment = BayesianExperiment(
                base_dir=base_dir,
                dry_run=dry_run,
                n_bootstraps=n_bootstraps,
                val_split=val_split,
                project_to_dag=project_to_dag,
                include_disease_labels=include_disease,
            )
            experiment.run()
