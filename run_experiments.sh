DEFAULT_ARGS="\
--fit_intercept \
--val_split=0.2 \
--n_bootstraps=30 \
--save_networks \
--result_dir=230619_metagenes_30boots_intercept_notissuecontext \
--test \
--no_disease_labels \
"
python -u experiments.py --network_type='neighborhood' ${DEFAULT_ARGS}
python -u experiments.py --network_type='correlation' ${DEFAULT_ARGS}
python -u experiments.py --network_type='markov' ${DEFAULT_ARGS}
python -u experiments.py --network_type='bayesian' ${DEFAULT_ARGS}
