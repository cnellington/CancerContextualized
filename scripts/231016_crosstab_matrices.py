import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--filepath', type=str, default=None)
# args = parser.parse_args()
# filepath = args.filepath

withna = True

# Individual diseases
filepath = '../results/230619_metagenes_30boots_intercept/markov-fit_intercept=True-val_split=0.2-n_bootstraps=30-dry_run=False-test=True-disease_test=None/subtyping/all_subtypes.csv'
df = pd.read_csv(filepath)
savedir = os.path.join(os.path.dirname(filepath), 'confusion' if not withna else 'confusion_na')
os.makedirs(savedir, exist_ok=True)
df = df.drop(columns='Subtype_Selected')

# Disease Groups
# filepath = '../results/230619_metagenes_30boots_intercept/markov-fit_intercept=True-val_split=0.2-n_bootstraps=30-dry_run=False-test=True-disease_test=None/subtyping/grouped/all_subtypes.csv'
# df = pd.read_csv(filepath)
# savedir = os.path.join(os.path.dirname(filepath), 'confusion' if not withna else 'confusion_na')
# os.makedirs(savedir, exist_ok=True)
# df = df.drop(columns='Subtype_Selected')

# Get TCGA subtypes from covars (only patients with expression)
# tcga_filepath = '../results/230619_metagenes_30boots_intercept/markov-fit_intercept=True-val_split=0.2-n_bootstraps=30-dry_run=False-test=True-disease_test=None/subtyping/all_subtypes.csv'
# tcga_df = pd.read_csv(tcga_filepath)
# tcga_df = tcga_df[['sample_id', 'Subtype_Selected']]
# df = df.drop(columns='Subtype_Selected')
# df = df.merge(tcga_df, on='sample_id')

# Get TCGA subtypes from file (all)
tcga_subtypes_df = pd.read_csv('../data/tcga_subtypes_selected.csv')
# tcga_subtypes_df = tcga_subtypes_df.drop(columns='disease_type')

# Define custom ordering for sorting column/row headers
# Split on '.' and take the last element as int. If no '.', sort as string. If 'NA', always last.
def custom_sort_networks(x):
    try:
        return f"{int(x.split('.')[-1]):04d}"
    except ValueError:
        if x == 'NA':
            return 'ZZZZZZ'
        else:
            return x


def custom_sort_tcga(x):
    if x == 'NA':
        return 'ZZZZZZ'
    else:
        return x


for disease_label in np.unique(df['disease_type']):
    diseases = disease_label.split('+')
    # diseases = ['GBM']
    disease_df = df[df['disease_type'] == disease_label].drop(columns='disease_type')
    tcga_disease_df =  tcga_subtypes_df[tcga_subtypes_df['disease_type'].isin(diseases)].drop(columns='disease_type')
    if withna:
        # Replace nulls with 'NA'
        disease_df = disease_df.merge(tcga_disease_df, on='sample_id', how='outer')
        disease_df = disease_df.fillna('NA')
    else:
        # drop rows with null values
        disease_df = disease_df.merge(tcga_disease_df, on='sample_id', how='inner')
        disease_df = disease_df.dropna()
    if len(disease_df) == 0:
        continue
    cols = disease_df['Subtype_Selected'].unique()
    rows = disease_df['network_subtypes'].unique()
    # Apply custom sort
    cols = sorted(cols, key=custom_sort_tcga)
    rows = sorted(rows, key=custom_sort_networks)
    confusion_mat = np.zeros((len(rows), len(cols)))
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            confusion_mat[i, j] = len(disease_df[(disease_df['Subtype_Selected'] == col) & (disease_df['network_subtypes'] == row)])
    # confusion = confusion_matrix(disease_df['network_subtypes'], disease_df['Subtype_Selected'])
    confusion_df = pd.DataFrame(confusion_mat, index=rows, columns=cols)
    confusion_df.index.name = 'Network Subtype'
    confusion_df.columns.name = 'TCGA Subtype'
    # Larger fontsize
    # plt.figure(figsize=(6, 6))
    sns.set(font_scale=1.1)
    sns.heatmap(confusion_df, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.title('Subtype Cross-Tabulation by Patient Counts')
    plt.savefig(os.path.join(savedir, f'{disease_label}_confusion.pdf'), bbox_inches='tight')
    plt.clf()
    plt.close()


# plt.show()

# Clustergram, cluster columns for neatness
# cg = sns.clustermap(
#     confusion_df, annot=True, cmap='Blues', fmt='g', cbar=False,
#     # figsize=(6, 8),
#     # dendrogram_ratio = 0.0,
#     # col_cluster=False,
# )
# cg.ax_row_dendrogram.set_visible(False)
# cg.ax_col_dendrogram.set_visible(False)
# # Turn off colorbar
# cg.cax.set_visible(False)
