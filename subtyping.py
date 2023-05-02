import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kaplanmeier as km
from matplotlib import cm


from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering

from plotting_utils import plot_dendrogram

data_dir = './data/'
result_dir = './results/' + '230424_bayesian-disease_labels=True-val_split=0-bootstraps=1-intercept=False-dryrun=False'

networks = pd.read_csv(f'{result_dir}/networks.csv').drop(columns='Set')

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

# Load known subtypes
known_subtypes_df = pd.read_csv(data_dir + 'tcga_subtypes.csv')
known_ids = known_subtypes_df['pan.samplesID'].values
formatted_ids = []
for known_id in known_ids:
    ret = known_id[:16]
    if len(ret) < 12:
        ret = np.nan
    elif len(ret) < 16:
        ret += '-01A'
    formatted_ids.append(ret)
known_subtypes_df['sample_id'] = formatted_ids
known_subtypes_df = known_subtypes_df[~known_subtypes_df['sample_id'].isnull()]
known_subtypes_df['disease_type'] = known_subtypes_df['cancer.type']
known_subtypes_df = known_subtypes_df[['sample_id', 'disease_type', 'Subtype_Selected']]
keep_idx = ['NA' not in subtype for subtype in known_subtypes_df['Subtype_Selected']]
known_subtypes_df = known_subtypes_df[keep_idx]

paad_subtypes_df = pd.read_csv(data_dir + 'tcga_paad_subtypes.csv')
paad_subtypes = ['squamous', 'immunogenic', 'progenitor', 'ADEX']
paad_subtypes_df['Subtype_Selected'] = ['PAAD.' + paad_subtypes[int(val) - 1] for val in paad_subtypes_df['mRNA Bailey Clusters (All 150 Samples) 1squamous 2immunogenic 3progenitor 4ADEX']]
paad_subtypes_df['sample_id'] = paad_subtypes_df['Tumor Sample ID']
paad_subtypes_df['disease_type'] = 'PAAD'
paad_subtypes_df = paad_subtypes_df[['sample_id', 'disease_type', 'Subtype_Selected']]
known_subtypes_df = known_subtypes_df.append(paad_subtypes_df)

uvm_subtypes_df = pd.read_csv(data_dir + 'tcga_uvm_subtypes.csv')
uvm_subtypes_df['Subtype_Selected'] = [f'UVM.SCNA{val}' for val in uvm_subtypes_df['SCNA Cluster No.']]
uvm_subtypes_df['sample_id'] = [patient[:12] + '-01A' for patient in uvm_subtypes_df['patient']]
uvm_subtypes_df['disease_type'] = 'UVM'
uvm_subtypes_df = uvm_subtypes_df[['sample_id', 'disease_type', 'Subtype_Selected']]
known_subtypes_df = known_subtypes_df.append(uvm_subtypes_df)
known_subtypes_df['disease_type'] = known_subtypes_df['disease_type'].apply(lambda x: x if x != 'OVCA' else 'OV')

covars['TCGA Subtype'] = covars.merge(known_subtypes_df, on='sample_id', how='left')['Subtype_Selected']

known_subtypes_df.to_csv(data_dir + 'tcga_subtypes_selected.csv', index=False)


# make columns numeric
numeric_covars = [
    'age_at_diagnosis',
    'percent_neutrophil_infiltration',
    'percent_monocyte_infiltration',
    'percent_normal_cells',
    'percent_tumor_nuclei',
    'percent_lymphocyte_infiltration',
    'percent_stromal_cells',
    'percent_tumor_cells',
    'stage',
]
numeric_covars += [col for col in covars.columns if 'Mutated' in col or 'Gene' in col or 'Arm' in col or 'Allele' in col]
for col in numeric_covars:
    covars[col] = pd.to_numeric(covars[col], errors='coerce')



savedir = f'{result_dir}/subtyping'
os.makedirs(savedir, exist_ok=True)


def get_rgblist(colors):
    rgblist = []
    for color in colors:
        colorscale = cm.get_cmap(color).colors
        rgblist += [f'rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})' for c in colorscale]
    return rgblist


def pancancer_dendrogram(dend_data, title):
    disease_rgblist = get_rgblist(['tab20b', 'tab20c'])
    site_rgblist = get_rgblist(['tab20'])

    disease_types = covars['disease_type'].values
    primary_sites = covars['primary_site'].values
    sample_types = covars['sample_type'].values
    spectrums = [
        disease_types,
        primary_sites,
        #         sample_types,
        #     clinical_covars['primary_site'].values
    ]
    spectrum_labels = [
        'Disease Type',
        'Primary Site',
        #         'Sample Type',
    ]
    spectrum_types = [
        'categorical',
        'categorical',
        #         'categorical',
    ]
    colors = [
        disease_rgblist[:len(np.unique(disease_types))],
        site_rgblist[:len(np.unique(primary_sites))],
        #         'Blues',
    ]
    show_legends = [
        True,
        True,
        #         True,
    ]
    plot_dendrogram(
        dend_data,
        title=title,
        method='ward',
        spectrums=spectrums,
        spectrum_labels=spectrum_labels,
        spectrum_types=spectrum_types,
        colors=colors,
        show_legends=show_legends,
        savepath=f'{savedir}/{title}.pdf',
    )

dend_features = networks.drop(columns=['sample_id']).values
pancancer_dendrogram(dend_features, title='Pancancer Network Organization')

exp_features = metagene_expression.drop(columns='sample_id').values
pancancer_dendrogram(exp_features, title='Pancancer Metagene Expression Organization')