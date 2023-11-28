# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# %%
df = pd.read_csv('../results/230619_metagenes_30boots_intercept/markov-fit_intercept=True-val_split=0.2-n_bootstraps=30-dry_run=False-test=True-disease_test=None/subtyping/all_pvals.csv')
# %%
num_cols = ['mv_shared', 'pair_shared', 'mv_outer', 'pair_outer']
df_net = df[df['method'] == 'Network Subtypes']
df_tcga = df[df['method'] == 'CoCA Subtypes']
df_exp = df[df['method'] == 'Expression Subtypes']
# %%
df_net[num_cols] = df_net[num_cols].applymap(lambda x: -np.log10(x))
df_tcga[num_cols] = df_tcga[num_cols].applymap(lambda x: -np.log10(x))
df_exp[num_cols] = df_exp[num_cols].applymap(lambda x: -np.log10(x))
df_tcga.head()
# %%
df_tcga_diff = df_net.copy()
df_tcga_diff[num_cols] = df_net[num_cols].values - df_tcga[num_cols].values
df_exp_diff = df_net.copy()
df_exp_diff[num_cols] = df_net[num_cols].values - df_exp[num_cols].values
# %%
exp_diff = df_exp_diff[num_cols].values
tcga_diff = df_tcga_diff[num_cols].values
min_diff = np.minimum(exp_diff, tcga_diff)
max_diff = np.maximum(exp_diff, tcga_diff)
df_min = df_net.copy()
df_min[num_cols] = min_diff
df_max = df_net.copy()
df_max[num_cols] = max_diff
df_min.head()
# %%
df_min.sort_values(by='mv_shared')
# %%
df_diff = df_diff[~pd.isnull(df_diff).any(axis=1)]
df_diff.head()
# %%
df_diff['mv_shared'].max()
# %%
df_diff

# %%
