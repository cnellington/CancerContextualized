# Whole lotta tech debt
import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import cm
from matplotlib.colors import ListedColormap
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import r_regression
from sklearn.cluster import KMeans, AgglomerativeClustering
from lifelines.statistics import logrank_test, multivariate_logrank_test, pairwise_logrank_test
from lifelines import KaplanMeierFitter
from lifelines.utils import qth_survival_times
import kaplanmeier as km
import umap

from plotting_utils import plot_dendrogram, cdist


numeric_covars = None


def load_data(data_dir, networks_file, dryrun=False):
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

    # known_subtypes_df.to_csv(data_dir + 'tcga_subtypes_selected.csv', index=False)


    # make columns numeric
    global numeric_covars
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
    return networks, covars, gene_expression, metagene_expression, survival_df, known_subtypes_df


def get_rgblist(colors):
    rgblist = []
    for color in colors:
        colorscale = cm.get_cmap(color).colors
        rgblist += [f'rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})' for c in colorscale]
    return rgblist


def pancancer_dendrogram(data, covars, title, savepath):
    plot_df = covars.merge(data, on='sample_id', how='inner')
    plot_df = plot_df.sort_values(by='disease_type')

    disease_rgblist = get_rgblist(['tab20b', 'tab20c'])
    site_rgblist = get_rgblist(['tab20'])

    disease_types = plot_df['disease_type'].values
    primary_sites = plot_df['primary_site'].values
    sample_types = plot_df['sample_type'].values
    spectrums = [
        disease_types,
        primary_sites,
        sample_types,
    ]
    spectrum_labels = [
        'Disease Type',
        'Primary Site',
        'Sample Type',
    ]
    spectrum_types = [
        'categorical',
        'categorical',
        'categorical',
    ]
    colors = [
        disease_rgblist[:len(np.unique(disease_types))],
        site_rgblist[:len(np.unique(primary_sites))],
        'Blues',
    ]
    show_legends = [
        True,
        True,
        True,
    ]
    plot_dendrogram(
        plot_df[data.columns].drop(columns=['sample_id']).values,
        title=title,
        method='ward',
        spectrums=spectrums,
        spectrum_labels=spectrum_labels,
        spectrum_types=spectrum_types,
        colors=colors,
        show_legends=show_legends,
        savepath=savepath,
    )
    savepath_split = savepath.split('.')
    savepath_noext = '.'.join(savepath_split[:-1])
    savepath_ext = savepath_split[-1]
    legend_savepath = savepath_noext + '_legend.' + savepath_ext
    plot_dendrogram(
        plot_df[data.columns].drop(columns=['sample_id']).values,
        title=title,
        method='ward',
        spectrums=spectrums,
        spectrum_labels=spectrum_labels,
        spectrum_types=spectrum_types,
        colors=colors,
        show_legends=show_legends,
        savepath=legend_savepath,
        dendro_height=1000,
    )
    plt.close('all')


survival_pvals = {}
def update_surv_pvals(split, experiment, diseases, pval):
    if split not in survival_pvals:
        survival_pvals[split] = {}
    if experiment not in survival_pvals[split]:
        survival_pvals[split][experiment] = {}
    survival_pvals[split][experiment]['+'.join(diseases)] = pval

    # for diseases in [
    #     ['HNSC', 'LUSC', 'LUAD'],
    #     ['LGG', 'GBM'],
    #     ['READ', 'COAD', 'STAD', 'ESCA'],
    #     ['UCEC', 'UCS', 'OV'],
    #     ['KICH', 'KIRC', 'KIRP'],
    #     ['THCA', 'PAAD'],
    # ]:


    # subtype_dfs = []
    # for diseases in single_diseases + disease_groups:
    # for diseases in [['UCEC', 'UCS', 'OV']]:
    # for diseases in [['OV'], ['CHOL']]:
    #     print(diseases)

def do_subtyping(diseases, subtyping_data, covars, known_subtypes_df, subtype_col, subtype_prefix, savedir=None):
    data_views = {
    #     'demographic': [],
        'biopsy': [
            'percent_neutrophil_infiltration',
            'percent_monocyte_infiltration',
            # 'percent_normal_cells',
            # 'percent_tumor_nuclei',
            'percent_lymphocyte_infiltration',
            'percent_stromal_cells',
            # 'percent_tumor_cells',
        ],
        'arm-level scna': [col for col in covars.columns if 'Arm' in col and 'Copies' not in col] + ['WGD'],
        'gene-level scna': [col for col in covars.columns if ('Gene' in col or 'Allele' in col) and ('Copies' not in col)],
        'snv': [col for col in covars.columns if 'Mutated' in col],
    }
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)

    # Only process samples in subtyping_data
    covars = covars.merge(subtyping_data[['sample_id']], on='sample_id', how='inner')[covars.columns]
    covars.reset_index(drop=True, inplace=True)
    subtyping_data.reset_index(drop=True, inplace=True)
    
    # multi-disease
    disease_idx = covars['disease_type'].values == diseases[0]
    for disease in diseases[1:]:
        disease_idx = np.logical_or(disease_idx, covars['disease_type'].values == disease)
    disease_idx = np.logical_and(disease_idx, covars['sample_type'] == 'Primary Tumor')  # Remove healthy
    disease_covars = covars[disease_idx]
    
    # Remove redundant features
    max_feature_overlap = 0.7
    redundant_features = set()
    for arm_col in data_views['arm-level scna']:
        for gene_col in data_views['gene-level scna']:
            x = disease_covars[arm_col].values[:, np.newaxis]
            y = disease_covars[gene_col].values
            corr = r_regression(x, y)
            if corr > max_feature_overlap:
                redundant_features.add(gene_col)

    x = disease_covars['WGD'].values[:, np.newaxis]
    for arm_col in data_views['arm-level scna']:
        y = disease_covars[arm_col].values
        corr = r_regression(x, y)
        if corr > max_feature_overlap:
            redundant_features.add(arm_col)
#     print(redundant_features)
    
    # Get known subtype labels
    known_subtype_labels = disease_covars.merge(known_subtypes_df, on='sample_id', how='left')['Subtype_Selected']
    unique_subtypes = disease_covars['TCGA Subtype'][~pd.isnull(disease_covars['TCGA Subtype'])].unique()
    num_known_subtypes = len(unique_subtypes)
    print('known subtypes', unique_subtypes, num_known_subtypes)
    
    # cluster networks and associte features with dendrogram splits
    method = 'ward'
    criterion = 'maxclust'
    network_samples = subtyping_data.drop(columns='sample_id')[disease_idx].values
    
    dist_array = cdist(network_samples)
    Z = linkage(dist_array, method=method)
    k = num_known_subtypes  # get as many network subtypes as known subtypes
    score = -1
    if num_known_subtypes < 2:
        for n_clusters in range(2, 10):
            subtypes = fcluster(Z, n_clusters, criterion=criterion)
            new_score = silhouette_score(squareform(dist_array), metric='precomputed', labels=subtypes)
            if new_score > score:
                score = new_score
                k = n_clusters
    subtypes = fcluster(Z, k, criterion=criterion)
    subtypes_idx = dendrogram(Z, no_plot=True)['leaves']
    # Increment by order of appearance
    subtype_names = {}
    for subtype in subtypes[subtypes_idx]:
        if subtype not in subtype_names:
            subtype_names[subtype] = f"{'+'.join(diseases)}.{subtype_prefix}.{len(subtype_names) + 1}"
    subtypes = np.array([subtype_names[subtype] for subtype in subtypes])
    score = silhouette_score(squareform(dist_array), metric='precomputed', labels=subtypes)
    disease_covars['Network Subtype'] = subtypes
    print('silhouette score', score)

    def get_pval(col):
        if col in redundant_features:
            return 1
        # Check if the column contains strings
        col_samples = disease_covars[col].values
        nan_idx = pd.isnull(col_samples)
        col_samples = col_samples[~nan_idx]
        col_subtypes = subtypes[~nan_idx]
        best_pval = 1
        if disease_covars[col].dtype == 'object':
            for subtype in np.unique(col_subtypes):
                for col_val in pd.unique(col_samples):
                    subtype_idx = col_subtypes == subtype
                    col_onehot = (col_samples == col_val).astype(int)
                    null_set = col_onehot[~subtype_idx]
                    test_set = col_onehot[subtype_idx]
                    _, pval = ttest_ind(test_set, null_set)
                    if np.isnan(pval):
                        pval = 1
                    if pval < best_pval:
                        best_pval = pval
        else:
            for subtype in np.unique(col_subtypes):
                subtype_idx = col_subtypes == subtype 
                null_set = col_samples[~subtype_idx]
                test_set = col_samples[subtype_idx]
                _, pval = ttest_ind(test_set, null_set)
                if np.isnan(pval):
                    pval = 1
                if pval < best_pval:
                    best_pval = pval
        return best_pval
    column_pvals = {col: get_pval(col) for col in numeric_covars}
    column_pvals.update({
        'race': get_pval('race'),
        'gender': get_pval('gender'),
    })

    # column_pvals = {col: 1 for col in numeric_covars}
    # for subtype in np.unique(subtypes):
    #     subtype_idx = subtypes == subtype
    #     for col in numeric_covars:
    #         if col in redundant_features:
    #             continue
    #         col_samples = disease_covars[col].values
    #         nan_idx = pd.isnull(col_samples)
    #         col_samples = col_samples[~nan_idx]
    #         col_subtype_idx = subtype_idx[~nan_idx]
    #         null_set = col_samples[~col_subtype_idx]
    #         test_set = col_samples[col_subtype_idx]
    #         _, pval = ttest_ind(test_set, null_set)
    #         if np.isnan(pval):
    #             pval = 1
    #         if pval < column_pvals[col]:
    #             column_pvals[col] = pval
    # sex_pval = 1.0
    # for subtype in np.nique
    
    max_view_features = 10
    max_feature_pval = 1e-4
    col_labels = []
    col_samples = []
    col_colors = []
    col_types = []
    col_legends = []
    pval_column = lambda col: f"{col} (-log(p) = {round(-np.log10(column_pvals[col]), 1)})" if col in column_pvals else col
    for (data_view, view_cols), color in zip(data_views.items(), ['Purples', 'Greens', 'Blues', 'Reds', 'Oranges']):
        view_pvals = np.array([column_pvals[col] for col in view_cols])
        view_idx = np.argsort(view_pvals)[:max_view_features]
        view_idx = view_idx[view_pvals[view_idx] < max_feature_pval]
        view_names = np.array(view_cols)[view_idx].tolist()
        view_pvals = view_pvals[view_idx].tolist()
        # view_labels = [f"{col} (-log(p) = {round(-np.log10(pval), 1)})" for col, pval in zip(view_names, view_pvals)]
        view_labels = [pval_column(col) for col in view_names]
        col_labels += view_labels
        col_samples += [disease_covars[col].values for col in view_names]
        col_colors += [color] * len(view_labels)
        if data_view == 'biopsy':
            col_types += ['continuous'] * len(view_labels)
            col_legends += [True] * len(view_labels)
        else:
            col_types += ['categorical'] * len(view_labels)
            col_legends += [False] * len(view_labels)
    # col_types = ['continuous'] * len(col_labels)
    # col_legends = [False] * len(col_labels)
#     print(list(zip(col_names, col_pvals, col_colors)))
    
    # add mandatory features to plot and concatenate
    always_names = ['Network Subtype', 'TCGA Subtype', 'disease_type', 'race', 'gender', 'age_at_diagnosis', 'stage', 'Purity', 'Ploidy']
    # for name in always_names:
    #     if name in column_pvals:
    #         always_pvals.append(column_pvals[col])
    #     else:
    #         always_pvals.append(1)
    always_titles = [pval_column(col) for col in always_names]
    title_labels = ['Network Subtype', 'TCGA Subtype', 'Disease Type', 'Race', 'Sex', 'Age at Diagnosis', 'Stage', 'Purity', 'Ploidy']
    always_labels = [title.replace(col, label) for col, title, label in zip(always_names, always_titles, title_labels)]
    # always_labels = [f"{title} (-log(p) = {round(-np.log10(pval), 1)})" if pval < 1 else title for title, pval in zip(always_titles, always_pvals)]
    always_colors = ['tab20b', 'tab20', 'Pastel1', 'rainbow', 'cool_r', 'Greys', 'Reds', 'Purples', 'Purples']
    always_samples = [disease_covars[col].values for col in always_names]
    always_types = ['categorical', 'categorical', 'categorical', 'categorical', 'categorical', 'continuous', 'categorical', 'continuous', 'continuous']
    always_legends = [True] * len(always_names)
    
    all_labels = always_labels + col_labels
    all_samples = always_samples + col_samples
    all_colors = always_colors + col_colors
    all_types = always_types + col_types
    all_legends = always_legends + col_legends
    
    # Remove boring features
    oncoplot_labels, oncoplot_samples, oncoplot_colors, oncoplot_types, oncoplot_legends = [], [], [], [], []
    for i in range(len(all_samples)):
        if len(pd.unique(all_samples[i])) > 1:
            oncoplot_labels.append(all_labels[i])
            oncoplot_samples.append(all_samples[i])
            oncoplot_colors.append(all_colors[i])
            oncoplot_types.append(all_types[i])
            oncoplot_legends.append(all_legends[i])

    # Get disease survival data
    disease_covars[subtype_col] = subtypes.astype(str)
    print(np.unique(subtypes))
    disease_subtypes = disease_covars[['sample_id', 'disease_type', subtype_col]]

    if savedir is not None:
        oncoplot = plot_dendrogram(
            network_samples,
            title=f'{"+".join(diseases)} Oncoplot',
            method=method,
            spectrums=oncoplot_samples,
            spectrum_labels=oncoplot_labels,
            spectrum_types=oncoplot_types,
            colors=oncoplot_colors,
            show_legends=oncoplot_legends,
            savepath=f'{savedir}/{diseases}-{subtype_col}-oncoplot.pdf' if savedir is not None else None,
            n_clusters=k,
        ) 
    plt.close('all')
    return disease_subtypes


def do_extra_plots(diseases, disease_subtypes, known_subtypes_df, survival_df, subtype_col, show=True, savedir=None):
    # Create a table with known subtypes
    subtypes_disease_idx = known_subtypes_df['disease_type'] == diseases[0]
    for disease in diseases:
        subtypes_disease_idx = np.logical_or(subtypes_disease_idx, known_subtypes_df['disease_type'] == disease)
    disease_known_subtypes = known_subtypes_df[subtypes_disease_idx]
    print(disease_known_subtypes['disease_type'].unique())

    disease_subtypes_inner = disease_subtypes.merge(disease_known_subtypes, on='sample_id', how='inner')
    disease_subtypes_inner['submitter_id'] = [sample_id[:12] for sample_id in disease_subtypes_inner['sample_id']]
    disease_survival_inner = disease_subtypes_inner.merge(survival_df, on='submitter_id', how='inner')
    disease_survival_inner.fillna('NA', inplace=True)

    disease_subtypes_outer = disease_subtypes.merge(disease_known_subtypes, on='sample_id', how='outer')
    disease_subtypes_outer['submitter_id'] = [sample_id[:12] for sample_id in disease_subtypes_outer['sample_id']]
    disease_survival_outer = disease_subtypes_outer.merge(survival_df, on='submitter_id', how='inner')
    disease_survival_outer.fillna('NA', inplace=True)

    # known_cmap = {subtype: color for subtype, color in zip(disease_survival['Subtype_Selected'].unique(), px.colors.qualitative.Plotly)}
    # network_cmap = {subtype: color for subtype, color in zip(disease_survival['network_subtype'].unique(), px.colors.qualitative.T10)}

    # Plot survival splits
    def plot_km(label_col, disease_survival_df, cmap, specifier=''):
    #     idx = ~disease_survival[label_col].isnull()
    #     if np.sum(idx) < 2:
    #         return False
    #     idx_df = disease_survival[idx]
        disease_survival_df = disease_survival_df[disease_survival_df[label_col] != 'NA']
        if len(disease_survival_df) == 0:
            return np.nan, np.nan
        
        multivariate_result = multivariate_logrank_test(disease_survival_df['time'].astype(float), disease_survival_df[label_col], disease_survival_df['died'])
        multivariate_pval = multivariate_result.p_value
        
        results = km.fit(disease_survival_df['time'], disease_survival_df['died'], disease_survival_df[label_col])#  != idx_df[label_col].values[0])
        colors = cmap.colors[:len(disease_survival_df[label_col].unique())]
        km.plot(results, title=f'{diseases} {label_col} Survival Function\n{specifier}\np-value {multivariate_result.p_value}', cmap=colors)
        plt.ylim(-0.05, 1.05)
    #     plt.tight_layout()
        if savedir is not None:
            plt.savefig(f'{savedir}/{diseases}-survival-{label_col}-{specifier}.pdf', bbox_inches='tight', pad_inches=.5)
        if show:
            plt.show()
        plt.clf()

        # Select only the subtypes with the strongest split
        pairwise_result = pairwise_logrank_test(disease_survival_df['time'].astype(float), disease_survival_df[label_col], disease_survival_df['died'])
        pairwise_tests = pairwise_result.summary.reset_index()
        best_row = pairwise_tests.iloc[pairwise_tests['p'].values.argmin()]
        best_subtype, worst_subtype, pairwise_min_pval = best_row[['level_0', 'level_1', 'p']]
        selected_df = disease_survival_df[disease_survival_df[label_col].isin([best_subtype, worst_subtype])]

        # Perform logrank test
    #     test = logrank_test(selected_df['time'].astype(float), selected_df['died'])
        test = multivariate_logrank_test(selected_df['time'].astype(float), selected_df[label_col], selected_df['died'])

        # Plot the survival function
        # for subtype in subtypes:
        #     subtype_df = disease_survival_df[disease_survival_df[label_col] == subtype]
        results = km.fit(selected_df['time'], selected_df['died'], selected_df[label_col])#  != idx_df[label_col].values[0])
        colors = cmap.colors[:len(selected_df[label_col].unique())]
        km.plot(results, title=f'{diseases} {label_col} Survival Function\n{specifier}\np-value {test.p_value}', cmap=colors)
        plt.ylim(-0.05, 1.05)
        if savedir is not None:
            plt.savefig(f'{savedir}/{diseases}-bestworstsurvival-{label_col}-{specifier}.pdf', bbox_inches='tight', pad_inches=.5)
        if show:
            plt.show()
        plt.clf()
        return multivariate_pval, pairwise_min_pval
        

    # Plot network and known subtype crosstabulation
    def plot_crosstab(group_col, count_col, disease_subtypes_df, cmap, specifier=''):
        plot_df = pd.get_dummies(disease_subtypes_df[count_col])
        plot_df[group_col] = disease_subtypes_df[group_col]
        plot_df = plot_df.groupby(group_col).sum().reset_index()
        plot_df.index = plot_df[group_col]
        new_cmap = ListedColormap(cmap.colors[:len(plot_df.columns) - 1])
        plot_df.drop(columns=group_col).plot(kind='bar', stacked=True, cmap=new_cmap)
        plt.title(f'{diseases} {count_col} by {group_col} Crosstabulation\n{specifier}')
    #     plt.tight_layout()
        if savedir is not None:
            plt.savefig(f'{savedir}/{diseases}-crosstab-{count_col}-{group_col}-{specifier}.pdf', bbox_inches='tight', pad_inches=.5)
        if show:
            plt.show()
        plt.clf()

    if len(disease_survival_inner) > 0:
        specifier = 'plotting shared samples only'
        known_mv_pval, known_pair_pval = plot_km('Subtype_Selected', disease_survival_inner, cm.get_cmap('tab20'), specifier=specifier)
        subtype_mv_pval, subtype_pair_pval = plot_km(subtype_col, disease_survival_inner, cm.get_cmap('tab20b'), specifier=specifier)
        plot_crosstab('Subtype_Selected', subtype_col, disease_survival_inner, cm.get_cmap('tab20b'), specifier=specifier)
        plot_crosstab(subtype_col, 'Subtype_Selected', disease_survival_inner, cm.get_cmap('tab20'), specifier=specifier)
    else:
        known_mv_pval, known_pair_pval = np.nan, np.nan
        subtype_mv_pval, subtype_pair_pval = np.nan, np.nan
        print('no inner plots')

    if len(disease_survival_outer) > 0:
        known_mv_pval_outer, known_pair_pval_outer = plot_km('Subtype_Selected', disease_survival_outer, cm.get_cmap('tab20'), specifier='plotting all samples with known subtypes')
        subtype_mv_pval_outer, subtype_pair_pval_outer = plot_km(subtype_col, disease_survival_outer, cm.get_cmap('tab20b'), specifier='plotting all samples with network subtypes')
        specifier = 'plotting all samples, NA = not shared between datasets'
        plot_crosstab('Subtype_Selected', subtype_col, disease_survival_outer, cm.get_cmap('tab20b'), specifier=specifier)
        plot_crosstab(subtype_col, 'Subtype_Selected', disease_survival_outer, cm.get_cmap('tab20'), specifier=specifier)
    else:
        known_mv_pval_outer, known_pair_pval_outer = np.nan, np.nan
        subtype_mv_pval_outer, subtype_pair_pval_outer = np.nan, np.nan
        print('no outer plots')

    plt.close('all') 
    return [
        subtype_mv_pval, 
        subtype_pair_pval, 
        subtype_mv_pval_outer, 
        subtype_pair_pval_outer, 
        known_mv_pval, 
        known_pair_pval, 
        known_mv_pval_outer, 
        known_pair_pval_outer, 
    ]


def main(data_dir, networks_file, dryrun = True):
    result_dir = '/'.join(networks_file.split('/')[:-1])
    savedir = f'{result_dir}/subtyping'
    os.makedirs(savedir, exist_ok=True)
    networks, covars, gene_expression, metagene_expression, survival_df, known_subtypes_df = load_data(data_dir, networks_file, dryrun=dryrun)

    # pancancer_dendrogram(networks, covars, 'Pancancer Network Organization', f"{savedir}/pancancer_network_dendrogram.pdf") 
    # pancancer_dendrogram(gene_expression, covars, 'Pancancer Transcriptomic Organization', f"{savedir}/pancancer_transcriptomic_dendrogram.pdf") 
    # pancancer_dendrogram(metagene_expression, covars, 'Pancancer Metagene Expression Organization', f"{savedir}/pancancer_metagene_dendrogram.pdf") 

    pvals_columns = ['disease', 'method', 'mv_shared', 'pair_shared', 'mv_outer', 'pair_outer']
    pvals_rows = []
    all_subtype_dfs = []
    for disease in np.unique(covars['disease_type'].values):
        subtype_col = 'network_subtypes'
        subtype_prefix = 'Net'
        disease_net_subtypes = do_subtyping([disease], networks, covars, known_subtypes_df, subtype_col, subtype_prefix, savedir=savedir)
        net_pvals_row = do_extra_plots([disease], disease_net_subtypes, known_subtypes_df, survival_df, subtype_col, show=False, savedir=savedir)
        pvals_rows.append([disease, 'CoCA Subtypes'] + net_pvals_row[4:])
        pvals_rows.append([disease, 'Network Subtypes'] + net_pvals_row[:4])
        subtype_col = 'expression_subtypes'
        subtype_prefix = 'Expr'
        disease_expr_subtypes = do_subtyping([disease], metagene_expression, covars, known_subtypes_df, subtype_col, subtype_prefix, savedir=savedir)
        expr_pvals_row = do_extra_plots([disease], disease_expr_subtypes, known_subtypes_df, survival_df, subtype_col, show=False, savedir=savedir)
        pvals_rows.append([disease, 'Expression Subtypes'] + expr_pvals_row[:4])
        disease_all_subtypes = disease_net_subtypes.drop(columns='disease_type').merge(disease_expr_subtypes.drop(columns='disease_type'), on='sample_id', how='outer').merge(known_subtypes_df[known_subtypes_df['disease_type'] == disease].drop(columns='disease_type'), on='sample_id', how='outer')
        disease_all_subtypes['disease_type'] = disease
        all_subtype_dfs.append(disease_all_subtypes)
    pd.DataFrame(data=pvals_rows, columns=pvals_columns).to_csv(f'{savedir}/all_pvals.csv', index=False)
    pd.concat(all_subtype_dfs, ignore_index=True).to_csv(f'{savedir}/all_subtypes.csv', index=False)


    disease_groups = [
        ['HNSC', 'LUSC', 'LUAD'],  # Respiratory
        ['LGG', 'GBM'],  # Brain
        ['READ', 'COAD', 'STAD', 'ESCA'],  # GI tract
        ['UCEC', 'UCS', 'OV'],  # Female reproductive organs
        ['KICH', 'KIRC', 'KIRP'],  # Kidney
        ['THCA', 'PAAD', 'OV'],  # Endocrine
        ['LIHC', 'CHOL'],  # Liver
    ]
    pvals_columns = ['disease', 'method', 'mv_shared', 'pair_shared', 'mv_outer', 'pair_outer']
    pvals_rows = []
    all_subtype_dfs = []
    savedir = os.path.join(savedir, 'grouped')
    os.makedirs(savedir, exist_ok=True)
    for diseases in disease_groups:
        disease_name = '+'.join(diseases)
        subtype_col = 'network_subtypes'
        subtype_prefix = 'Net'
        disease_net_subtypes = do_subtyping(diseases, networks, covars, known_subtypes_df, subtype_col, subtype_prefix, savedir=savedir)
        net_pvals_row = do_extra_plots(diseases, disease_net_subtypes, known_subtypes_df, survival_df, subtype_col, show=False, savedir=savedir)
        pvals_rows.append([disease_name, 'CoCA Subtypes'] + net_pvals_row[4:])
        pvals_rows.append([disease_name, 'Network Subtypes'] + net_pvals_row[:4])
        subtype_col = 'expression_subtypes'
        subtype_prefix = 'Expr'
        disease_expr_subtypes = do_subtyping(diseases, metagene_expression, covars, known_subtypes_df, subtype_col, subtype_prefix, savedir=savedir)
        expr_pvals_row = do_extra_plots(diseases, disease_expr_subtypes, known_subtypes_df, survival_df, subtype_col, show=False, savedir=savedir)
        pvals_rows.append([disease_name, 'Expression Subtypes'] + expr_pvals_row[:4])
        disease_all_subtypes = disease_net_subtypes.drop(columns='disease_type').merge(disease_expr_subtypes.drop(columns='disease_type'), on='sample_id', how='outer').merge(known_subtypes_df[known_subtypes_df['disease_type'] == disease_name].drop(columns='disease_type'), on='sample_id', how='outer')
        disease_all_subtypes['disease_type'] = disease_name
        all_subtype_dfs.append(disease_all_subtypes)
    pd.DataFrame(data=pvals_rows, columns=pvals_columns).to_csv(f'{savedir}/all_pvals.csv', index=False)
    pd.concat(all_subtype_dfs, ignore_index=True).to_csv(f'{savedir}/all_subtypes.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--networks', type=str)
    parser.add_argument('--dryrun', action='store_true')
    args = parser.parse_args()
    main(args.data_dir, args.networks, dryrun = args.dryrun)
