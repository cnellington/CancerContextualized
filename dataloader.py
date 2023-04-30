import os
#import dill as pickle
import pickle as pkl
import numpy as np
import pandas as pd
import warnings
import torch

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def load_toy_data(
        num_features=10,
        num_samples=100,
        num_labels=2,
        num_contexts=10,
):
    C = np.random.normal(0, 1, (num_samples, num_contexts))
    X = np.random.normal(0, 1, (num_samples, num_features))
    labels = np.random.choice(num_labels, size=num_samples)
    ids = np.arange(num_samples)
    C_train, C_test, X_train, X_test, labels_train, labels_test, ids_train, ids_test = train_test_split(C, X, labels, ids, test_size=0.2)
    col_names = [f'feature_{i}' for i in range(num_features)]
    return C_train, C_test, X_train, X_test, labels_train, labels_test, ids_train, ids_test, col_names


def load_data(
    num_features=50,        # number of expression features to consider
    tumor_only=False,        # remove healthy samples
    hallmarks_only=False,    # reduce COSMIC genes to only the intersection between COSMIC and Hallmarks
    single_hallmark=None,    # reduce genes to a single hallmark set
    pretransform_norm=False, # normalize before the feature transformation
    transform='pca',          # None, or transform expression using 'pca' or 'hallmark avg'
    feature_selection='population',   # select genetic features according to population variance (population) or weighted disease-specific variance (disease)
    disease_labels=True,
):
    # Get Data
    data_dir = './data/'
    covariate_files = [
        'clinical_covariates.csv',
        'snv_covariates.csv',
        'scna_other_covariates.csv',
        'scna_arm_covariates.csv',
        'scna_gene_covariates.csv',
    ]

    # load covariates files
    covars = pd.read_csv(data_dir + covariate_files[0], header=0)
    for covariate_file in covariate_files[1:]:
        covars = covars.merge(pd.read_csv(data_dir + covariate_file, header=0), on='sample_id', how='inner')
    genomic = pd.read_csv(data_dir + "transcriptomic_features.csv", header=0)
    # only keep the rows that are in covar
    genomic = covars.merge(genomic, on='sample_id', how='inner')[genomic.columns]
    
    tnames_full = np.load(data_dir + "transcript_names.npy")
    hallmark_genes = pd.read_csv(data_dir + 'h.all.v7.5.1.symbols.gmt', header=None, sep='\t')
    hallmark_dict = {}
    # for each row iin hallmark_genes, organize in a seperate dictionary entry
    for i, row in hallmark_genes.iterrows():
        hallmark_label = row[0]
        hallmark_set = set(row[2:])
        if np.nan in hallmark_set:
            hallmark_set.remove(np.nan)
        hallmark_dict[hallmark_label] = hallmark_set

    # Remove uninteresting features
    def drop_cols(df):
        drop_list = []
        for col in df.columns:
            if len(df[col].unique()) < 2: #drop columns with only one value
                drop_list.append(col)
        df = df.drop(columns=drop_list)
        print(f'Dropped singular columns: {drop_list}')
        return df
    covars = drop_cols(covars)

    # Remove non-cancerous samples, 2 types of samples: Solid Tissue Normal and Primary Tumor
    if tumor_only:
        cancer_idx = covars['sample_type'] != 'Solid Tissue Normal'
        covars = covars[cancer_idx]
        genomic = genomic[cancer_idx]

    # deterministic numeric encoding for contexts of interest
    context_df = covars.copy()
    # if the datatype are numeric or not
    numeric_covars = {
        'age_at_diagnosis': True,
        'gender': False,
        'year_of_birth': True,
        'race': False,
        'sample_type': False,
        'percent_stromal_cells': True,
        'percent_tumor_cells': True,
        'percent_normal_cells': True,
        'percent_neutrophil_infiltration': True,
        'percent_lymphocyte_infiltration': True,
        'percent_monocyte_infiltration': True,
        'Purity': True,
        'Ploidy': True,
        'WGD': False,
        'stage': True,
    }

    # if disease_labels set to true, then add the following to the numeric_covars
    # disease label determine if this is included in the context information
    if disease_labels:
        print('Using disease labels')
        numeric_covars.update({
            'primary_site': False,
            'disease_type': False,
        })

    
#     mut_cols = [col for col in covars.columns if 'Major' in col or 'Minor' in col or 'Mutated' in col]
    #get a list of column names with following keywords
    mut_cols = [col for col in covars.columns if 'Arm' in col or 'Gene' in col or 'Allele' in col or 'Mutated' in col]
    # update dictionary with the list of column names in mut_cols and the value for these new keys are True
    numeric_covars.update({f"{gene}": True for gene in mut_cols})

    numeric_headers = []
    for col, numeric in numeric_covars.items(): # iterate through each entry of the dictionary
        if numeric:
            # Convert numeric values to floats, replace NaN with column mean
            num_col = pd.to_numeric(context_df[col], errors='coerce')
            num_col = num_col.fillna(num_col.mean()) #imputation here
            num_header = col+"_numeric"
            numeric_headers.append(num_header)
            context_df[num_header] = num_col
        else:
            # Make one-hot encoded dummy variables for categorical variables
            dummies = pd.get_dummies(context_df[col], prefix=col)
            numeric_headers += dummies.columns.tolist()
            context_df = context_df.merge(dummies, left_index=True, right_index=True)
    context_df = context_df.drop(columns=covars.columns)
    context_df = context_df.copy()
    print(f"df contains NaN: {context_df.isnull().values.any()}")

    # Get dataset values
    C = context_df.values
    X = genomic.drop(columns='sample_id').values
    print(f"Context shape {C.shape}, Expression shape {X.shape}")


    # Remove uninteresting genes
    consistent_feats = np.mean(X > 0, axis=0) > 0.999
    X = X[:, consistent_feats]
    tnames_full = tnames_full[consistent_feats]

    # reduce to all hallmarks
    if hallmarks_only:
        hallmark_idx = np.isin(tnames_full, hallmark_genes.values[:, 2:])
        X = X[:, hallmark_idx]
        tnames_full = tnames_full[hallmark_idx]

    # Take a single hallmark set of genes
    if single_hallmark:
        hallmark_idx = np.isin(tnames_full, list(hallmark_dict[single_hallmark]))
        X = X[:, hallmark_idx]
        tnames_full = tnames_full[hallmark_idx]
        
    # Get train-test split
    labels = covars['disease_type'].values
    tcga_ids = covars['sample_id'].values
    train_idx, test_idx = train_test_split(range(len(X)), test_size=0.2, random_state=10)
    C_train, C_test = C[train_idx], C[test_idx]
    X_train, X_test = X[train_idx], X[test_idx]
    labels_train, labels_test = labels[train_idx], labels[test_idx]
    tcga_ids_train, tcga_ids_test = tcga_ids[train_idx], tcga_ids[test_idx]
    
    # Pre-transformation normalization
    if pretransform_norm:
        X_means = np.mean(X_train, axis=0)
        X_stds = np.std(X_train, axis=0)
        X_stds[X_stds == 0] = 1
        X_train -= X_means
        X_test -= X_means
        X_train /= X_stds
        X_test /= X_stds
    
    # Transform expression data
    if transform == 'pca':
        pca = PCA(n_components=None, random_state=1).fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        tnames_full = np.array([f"PC{i}" for i in range(X_train.shape[1])])
    if transform == 'hallmark':
        hallmark_tnames = []
        hallmark_train = []
        hallmark_test = []
        for i, (hallmark_name, geneset) in enumerate(hallmark_dict.items()):
            hallmark_idx = np.isin(tnames_full, list(geneset))
            if not hallmark_idx.any():
                continue
            hallmark_tnames.append(hallmark_name)
            hallmark_train.append(X_train[:, hallmark_idx].mean(axis=1))
            hallmark_test.append(X_test[:, hallmark_idx].mean(axis=1))
        tnames_full = np.array(hallmark_tnames)
        X_train = np.array(hallmark_train).T
        X_test = np.array(hallmark_test).T

    # rank features
    if feature_selection == 'disease':
        # Take a weighted average of intra-disease variance choose the highest scoring genes
        var_avg = np.zeros(len(tnames_full))
        for disease in np.unique(labels_train):
            disease_idx = labels_train == disease
            disease_vars = X_train[disease_idx].var(axis=0)
#             print(f"{disease} {np.sum(disease_idx)}")
            var_avg += disease_vars * (np.sum(disease_idx) / len(disease_idx))  # weight by tissue samples
        feature_ordering = np.argsort(-var_avg)
    if feature_selection == 'population':
        population_vars = X_train.var(axis=0)
        feature_ordering = np.argsort(-population_vars)
    # todo: logistic selection based on survival

    # trim to desired number of features
    feature_idx = feature_ordering[:num_features]
    X_train = X_train[:, feature_idx]
    X_test = X_test[:, feature_idx]
    tnames_full = tnames_full[feature_idx]

    # Normalize the data
    C_means = np.mean(C_train, axis=0)
    C_stds = np.std(C_train, axis=0)
    C_stds[C_stds == 0] = 1
    C_train -= C_means
    C_test -= C_means
    C_train /= C_stds
    C_test /= C_stds

    X_means = np.mean(X_train, axis=0)
    X_stds = np.std(X_train, axis=0)
    X_stds[X_stds == 0] = 1
    X_train -= X_means
    X_test -= X_means
    X_train /= X_stds
    X_test /= X_stds
        

    print(f"Covariates: {C_train.shape} {C_test.shape} {len(C)}")
    print(f"Features: {X_train.shape} {X_test.shape} {len(X)}")
    
    return C_train, C_test, X_train, X_test, labels_train, labels_test, tcga_ids_train, tcga_ids_test, tnames_full


if __name__ == '__main__':
#     data_dir = './data/'
#     args = [
#         data_dir,
#         1000,
#         False,
#         False,
#         'HALLMARK_TNFA_SIGNALING_VIA_NFKB',
#         True,
#         'hallmark',
#         'disease',
#     ]
#     print(args)
    load_data()
#     for num_features in [10, 100, 1000]:
#         for tumor_only in [True, False]:
#             for hallmarks_only in [True, False]:
#                 for single_hallmark in [None, 'HALLMARK_TNFA_SIGNALING_VIA_NFKB']:
#                     for pretransform_norm in [True, False]:
#                         for  transform in [None, 'pca', 'hallmark']:
#                             for feature_selection in ['disease', 'population']:
#                                 args = [
#                                     data_dir,
#                                     num_features,
#                                     tumor_only,
#                                     hallmarks_only,
#                                     single_hallmark,
#                                     pretransform_norm,
#                                     transform,
#                                     feature_selection,
#                                 ]
#                                 print(args)
#                                 load_data(*args)
