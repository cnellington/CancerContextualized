# Cancer Contextualized

Code used "[Learning to Estimate Sample-Specific Transcriptional Networks for 7000 Tumors](https://www.biorxiv.org/content/10.1101/2023.12.01.569658v1)"

Most of the training and evaluation code has since been improved and added to the  [Contextualized](https://contextualized.ml/) Python library.
For a simpler and more intuitive workflow with tutorials and interactive plots, please visit the updated [CancerContextualized](https://github.com/cnellington/CancerContextualized/) repository.


## Setup
1. Create a new conda environment with python=3.10
2. Run `pip install -r requirements.txt`
3. Afterward run `pip install pandas==1.5.1`
3. Download data from https://cmu.app.box.com/folder/203433039257 into `data/`

## Fit and Test on a Random Split
```bash
python -u experiments.py \
  --network_type='markov' \
  --fit_intercept \
  --val_split=0.2 
  --n_bootstraps=30 \
  --save_networks \
  --result_dir=results/markov \
  --test
```
Substitute `--network_type` with 'correlation' or 'neighborhood'

## Fit and Test on a Disease Hold-out Split
```bash
python -u experiments.py \
  --network_type='markov' \
  --fit_intercept \
  --val_split=0.2 
  --n_bootstraps=30 \
  --save_networks \
  --result_dir=results_holdout/markov/thca \
  --disease_labels=False \
  --disease_test='THCA' 
```

## Plot Errors
```bash
# Absolute Errors
python -u plot_mses.py \
  --mses results/markov/mse_df.csv \
  --savedir results/markov/mse_plots

# Relative Errors
python -u plot_mses.py \
  --mses results/markov/mse_df.csv \
  --relto 'Disease-specific' \
  --savedir results/markov/mse_plots_rel
```

## Plot Subtypes
```bash
python -u subtyping.py  \
  --data_dir 'data/'
  --networks results/markov/networks.csv 
```
Saves subtyping plots for all diseases and tissues in `results/markov/subtypes/`
