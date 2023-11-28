# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# %%
# def load_data(data_dir, networks_file, dryrun=False):
numeric_covars = None
data_dir = '../data/'
networks_file = '../results/230619_metagenes_30boots_intercept/markov-fit_intercept=True-val_split=0.2-n_bootstraps=30-dry_run=False-test=True-disease_test=None/networks.csv'
dryrun = False
covariate_files = [
    'clinical_covariates.csv',
    'snv_covariates.csv',
    'scna_other_covariates.csv',
    'scna_arm_covariates.csv',
    'scna_gene_covariates.csv',
]
covars = pd.read_csv(data_dir + covariate_files[0], header=0)
for covariate_file in covariate_files[1:]:
    covars = covars.merge(pd.read_csv(data_dir + covariate_file, header=0), on='sample_id', how='inner')

survival_df = pd.read_csv(data_dir + 'survival.csv')
gene_expression = pd.read_csv(data_dir + 'transcriptomic_features.csv')
metagene_expression = pd.read_csv(data_dir + 'metagene_expression.csv')

networks = pd.read_csv(networks_file).drop(columns='Set')
networks = covars.merge(networks, on='sample_id', how='inner')[networks.columns]
networks = networks.drop(columns=[col for col in networks.columns if len(networks[col].unique()) == 1])

known_subtypes_df = pd.read_csv(data_dir + 'tcga_subtypes_selected.csv')
covars['TCGA Subtype'] = covars.merge(known_subtypes_df, on='sample_id', how='left')['Subtype_Selected']

# make columns numeric
# global numeric_covars
numeric_covars = [
    'age_at_diagnosis',
    'percent_neutrophil_infiltration',
    'percent_monocyte_infiltration',
    # 'percent_normal_cells',
    # 'percent_tumor_nuclei',
    'percent_lymphocyte_infiltration',
    'percent_stromal_cells',
    # 'percent_tumor_cells',
    'Purity',
    'Ploidy',
    'WGD',
    'stage',
]
numeric_covars += [col for col in covars.columns if 'Mutated' in col or 'Gene' in col or 'Arm' in col or 'Allele' in col]
for col in numeric_covars:
    covars[col] = pd.to_numeric(covars[col], errors='coerce')

# Do a little data cleaning to make the plots nice
# Drop columns with 'X' 'Arm' and 'Loss of Heterozygosity' in the name
drop_x = [col for col in covars.columns if 'X' in col and 'Arm' in col and 'Loss of Heterozygosity' in col]
covars = covars.drop(columns=drop_x)
# Change duplicated to amplified by checking if number of copies is greater than 2 when WGD is true
genes = [col.split(' Gene')[0] for col in covars.columns if 'Gene' in col]
arms = [col.split(' Major Arm')[0] for col in covars.columns if 'Major Arm' in col]
# Ploidy = tumor_ploidy * purity + 2 * (1 - purity)
# (ploidy - 2 * (1 - purity)) / purity = tumor_ploidy
tumor_ploidy = (covars['Ploidy'] - 2 * (1 - covars['Purity'])) / covars['Purity']
gene_amplified_cols = []
for gene in genes:
    n_major_copies = covars[f'{gene} Major Allele Copies']
    n_minor_copies = covars[f'{gene} Minor Allele Copies']
    covars[f'{gene} Major Allele Amplified'] = n_major_copies > tumor_ploidy
    covars[f'{gene} Minor Allele Amplified'] = n_minor_copies > tumor_ploidy
    gene_amplified_cols.append(f'{gene} Major Allele Amplified')
    gene_amplified_cols.append(f'{gene} Minor Allele Amplified')
arm_amplified_cols = []
for arm in arms:
    n_major_copies = covars[f'{arm} Major Arm Copies']
    n_minor_copies = covars[f'{arm} Minor Arm Copies']
    covars[f'{arm} Major Arm Amplified'] = n_major_copies > tumor_ploidy
    covars[f'{arm} Minor Arm Amplified'] = n_minor_copies > tumor_ploidy
    arm_amplified_cols.append(f'{arm} Major Arm Amplified')
    arm_amplified_cols.append(f'{arm} Minor Arm Amplified') 
# Drop columns with 'Duplicated' in the name
drop_duplicated = [col for col in covars.columns if 'Duplicated' in col]
covars = covars.drop(columns=drop_duplicated)
# Drop columns with 'Copies' in the name
# drop_copies = [col for col in covars.columns if 'Copies' in col]
# covars = covars.drop(columns=drop_copies)
# Clean up numeric_covars
numeric_covars += gene_amplified_cols + arm_amplified_cols
numeric_covars = [col for col in numeric_covars if col not in drop_x + drop_duplicated]
# Convert age from days to years
covars['age_at_diagnosis'] = covars['age_at_diagnosis'] // 365.25
covars = covars.copy()  # Clean up fragmentation



if dryrun:
    dryrun_ids = covars[['sample_id', 'submitter_id']][covars['disease_type'].isin(['LGG', 'GBM'])]
    networks = networks[networks['sample_id'].isin(dryrun_ids['sample_id'])]
    covars = covars[covars['sample_id'].isin(dryrun_ids['sample_id'])]
    gene_expression = gene_expression[gene_expression['sample_id'].isin(dryrun_ids['sample_id'])]
    metagene_expression = metagene_expression[metagene_expression['sample_id'].isin(dryrun_ids['sample_id'])]
    survival_df = survival_df[survival_df['submitter_id'].isin(dryrun_ids['submitter_id'])]
    known_subtypes_df = known_subtypes_df[known_subtypes_df['sample_id'].isin(dryrun_ids['sample_id'])]
print('finished dataloading')
    # return networks, covars, gene_expression, metagene_expression, survival_df, known_subtypes_df


# %%
sns.heatmap(networks.drop(columns='sample_id').corr(), cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.show()

# %%
plt.figure(figsize=(1, 10))
nums = []
for i in range(3):
    nums.extend([i] * 2)
sns.heatmap(pd.DataFrame(nums), cmap='rainbow_r', cbar=False, xticklabels=False, yticklabels=False)
plt.show()

# %%
plt.figure(figsize=(10, 10))
data = covars[numeric_covars].astype(float).values
# Column normalize
data = data / np.linalg.norm(data, axis=0)
sns.heatmap(covars[numeric_covars].astype(float).iloc[:100], cbar=False, xticklabels=False, yticklabels=False)
plt.show()
# %%
