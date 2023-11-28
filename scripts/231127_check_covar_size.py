# %%
import pandas as pd
data_dir = '../data/.'
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
# %%
print('heello')

# %%
