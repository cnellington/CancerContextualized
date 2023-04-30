import os
import pickle as pkl
import numpy as np
import pandas as pd
import warnings
from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#%% New Helper Function for splitting data by disease

# diesase split into pickles files because of no randomness and so we don't have to do this again
def disease_specific_split(labels, covars, tcga_ids, C, X, disease, tnames_full):
    """
    Produce train and test data for disease

    Args:
        labels (ndarray): the disease_type column of the covars dataframe
        covars (pd df): covars dataframe, contains all the covariates
        tcga_ids (ndarray): the sample_id column of the covars dataframe
        C (ndarray): the contextual data
        X (ndarray): the gene expression data
        disease (str): if not None, only split the data for this disease
    """

    if not os.path.exists('./disease_specific_data'):
        os.makedirs('./disease_specific_data')

    def train_test_disease (labels, covars, tcga_ids, C, X, disease, tnames_full):
        # get the context info first
        train_idx = covars[covars['disease_type'] != disease].index.values
        test_idx = covars[covars['disease_type'] == disease].index.values

        assert len(train_idx) + len(test_idx) == len(covars), "Error: train and test indices do not add up to total number of samples" 
        c_train = C[train_idx]
        c_test = C[test_idx]

        # get the expression info next
        x_train = X[train_idx]
        x_test = X[test_idx]

        assert x_train.shape[0] == c_train.shape[0], "Error: train expression and context do not have the same number of samples"
        assert x_test.shape[0] == c_test.shape[0], "Error: test expression and context do not have the same number of samples"

        # get the tcga_ids
        tcga_ids_train = tcga_ids[train_idx]
        tcga_ids_test = tcga_ids[test_idx]

        assert x_train.shape[0] == tcga_ids_train.shape[0], "Error: train expression and tcga_ids do not have the same number of samples"
        assert x_test.shape[0] == tcga_ids_test.shape[0], "Error: test expression and tcga_ids do not have the same number of samples"

        # get the labels:
        labels_train, labels_test = labels[train_idx], labels[test_idx]
        assert x_train.shape[0] == labels_train.shape[0], "Error: train expression and labels do not have the same number of samples"
        assert x_test.shape[0] == labels_test.shape[0], "Error: test expression and labels do not have the same number of samples"

        # save the files to a pickle file
        if not os.path.exists('./disease_specific_data'):
            os.makedirs('./disease_specific_data')
        with open(f'./disease_specific_data/data_{disease}.pkl', 'wb') as f:
            pkl.dump([c_train, c_test, x_train, x_test, tcga_ids_train, tcga_ids_test, labels_train, labels_test, tnames_full], f)

    # if interested in one disease
    if disease != None:
        if disease not in np.unique(labels):
            print(f"Error: {disease} is not a valid disease name")
            quit()
        print("Processing disease: ", disease)
        train_test_disease(labels, covars, tcga_ids, C, X, disease, tnames_full)
    else:
        for d in np.unique(labels):
            print("Processing disease: ", d)
            train_test_disease(labels, covars, tcga_ids, C, X, d, tnames_full)


#%% Input parameters
#original params
def disease_load_data(
    #tumor_only=True        
    hallmarks_only=False,    
    single_hallmark=None,
    disease = None, #if True, output pkl files for all diseases    
    #pretransform_norm=False        
    #disease_labels=False

    # new parmas for Disease_CV
    #disease_CV = True         # if True, doing disease context CV (hold out a specific disease in CV)    
):

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


    cancer_idx = covars['sample_type'] != 'Solid Tissue Normal'
    covars = covars[cancer_idx]
    genomic = genomic[cancer_idx]

    context_df = covars.copy()
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

    consistent_feats = np.mean(X > 0, axis=0) > 0.999
    X = X[:, consistent_feats]
    tnames_full = tnames_full[consistent_feats]

    
    if hallmarks_only:
        hallmark_idx = np.isin(tnames_full, hallmark_genes.values[:, 2:])
        X = X[:, hallmark_idx]
        tnames_full = tnames_full[hallmark_idx]

    # Take a single hallmark set of genes
    if single_hallmark:
        hallmark_idx = np.isin(tnames_full, list(hallmark_dict[single_hallmark]))
        X = X[:, hallmark_idx]
        tnames_full = tnames_full[hallmark_idx]

    
    labels = covars['disease_type'].values
    tcga_ids = covars['sample_id'].values

    disease_specific_split(labels, covars, tcga_ids, C, X, disease, tnames_full)

#%% normalize data

def disease_data_transformation(num_features, pretransform_norm, transform, feature_selection, disease_label):
    """
    All original code with additino of loading the pickle file and some print statements. 

    Args:
        num_features (int): number of features interested
        pretransform_norm (bool): if performing pretransform normalization
        transform (str): pca or hallmark
        feature_selection (str): disease or population (no disease for now)
        disease_label (str): the disease label 

    Returns:
        ndarray: seven ndarrays for the train and testing datasets
    """

    #open the pickle file
    with open('./disease_specific_data/data_' + disease_label + '.pkl', 'rb') as f:
        C_train, C_test, X_train, X_test, tcga_ids_train, tcga_ids_test, labels_train, labels_test, tnames_full = pkl.load(f)
    
    print(f"Finished loading data for {disease_label}, now transforming...")
    print(f"C_train: {C_train.shape}, C_test: {C_test.shape}")
    print(f"X_train: {X_train.shape}, X_test {X_test.shape}")
    print(f"tcga_ids_train: {tcga_ids_train.shape}, tcga_ids_test: {tcga_ids_test.shape}")
    print(f"transcript_names: {tnames_full.shape}")


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
    #     if feature_selection == 'disease':
    #         # Take a weighted average of intra-disease variance choose the highest scoring genes
    #         var_avg = np.zeros(len(tnames_full))
    #         for disease in np.unique(labels_train):
    #             disease_idx = labels_train == disease
    #             disease_vars = X_train[disease_idx].var(axis=0)
    # #             print(f"{disease} {np.sum(disease_idx)}")
    #             var_avg += disease_vars * (np.sum(disease_idx) / len(disease_idx))  # weight by tissue samples
    #         feature_ordering = np.argsort(-var_avg)
    
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
        
    print(f"Finished transforming data for {disease_label}. Printing new shapes...")
    print(f"Covariates: {C_train.shape} {C_test.shape}")
    print(f"Features: {X_train.shape} {X_test.shape}")
    print(f"transcript names: {tnames_full.shape}")

    return C_train, C_test, X_train, X_test, tcga_ids_train, tcga_ids_test, labels_train, labels_test, tnames_full

