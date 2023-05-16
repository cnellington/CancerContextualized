import os
import numpy as np
#import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from disease_dataloader import disease_load_data, disease_data_transformation
from disease_train import run_experiment

#%% Helper FUnctions

def check_path(): 
    """
    Check if the path that holds the disase specific data exists
    Path currently hard coded. Subroutine of check_disease_data().
    
    Returns:
        Bool: T if path exist otherwise return F.
    """
    if not os.path.exists('./disease_specific_data'):
        return False
    else:
        return True



def check_disease_data(labels):
    """
    Check if the disease specific data exists. 
    If it does, check if the diseases are the same as the labels. If not, recreate the data.
    
    Args:
        labels (ndarray): Array of unique disease labels from clinical_covariates.csv
    """
    if check_path() == False:
        print("Disease Specific Data does not exist. Creating data for all diseases now...")
        disease_load_data(hallmarks_only = False, 
                          single_hallmark = None, 
                          disease = None)
    else:
        disease_files = os.listdir('./disease_specific_data')
        disease_files = [x.split('_')[1].split('.')[0] for x in disease_files]
        if sum(np.sort(disease_files) == np.sort(labels)) != len(labels):
            print("Something is wrong with the current disease data. Recreating data for all diseases now...")
            disease_load_data(hallmarks_only = False, 
                              single_hallmark = None, 
                              disease = None) 
        print("Disease Specific Data exists. All diseases are up to date.")
        

def plot_concat_mse(plot_df):
    """plottng the final concateenated dataframe consists of all testing performances for all diseases

    Args:
        plot_df (pandas df): dataframe that holds all testing performances for all diseases
    """
    n_diseases = len(plot_df['Disease'].unique())
    fig, ax = plt.subplots(figsize=(n_diseases + 5, 5))
    sns.barplot(
        plot_df,
        x='Disease',
        y='MSE',
        hue='Model',
        hue_order=['Population', 'Cluster-specific', 'Contextualized'],
        palette = ['lightblue', 'deepskyblue', 'orange'],
        errorbar='sd',
        capsize=0.05,
    #     edgecolor='black',
        ax=ax
    )
    plt.xlim(-1, n_diseases)
    plt.ylim(0, 6)
    ax.plot(list(range(-1, n_diseases + 1)), [1] * (n_diseases + 2), linestyle='dashed', color='lightgrey')


    labels = [f'{label}' for label in plot_df['Disease'].unique()]
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=14)
    #ax.set_yticklabels([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], fontsize=14)

    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)


    plt.xlabel('Disease Type (# Training Samples)', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.title('Test Errors by Disease Type (Disease Specific CV)', fontsize=14)
    plt.tight_layout()

    #plt.savefig(f'./results/test.pdf', dpi=300)
    plt.show()
    plt.clf()




#%%
def load_disease_data(data_state, experiment_params):
    # unfinished function  needs more work depending on how to integrate the actual running model part...

    data_dir = './data/' # hard coded for now
    covars = pd.read_csv(data_dir + 'clinical_covariates.csv', header=0)
    labels = np.unique(covars['disease_type'].values)

    check_disease_data(labels) 

    test_df_list = []
    for d in labels:

        data_state["disease_label"] = d
        C_train, C_test, X_train, X_test, tcga_ids_train, tcga_ids_test, labels_train, labels_test, col_names = disease_data_transformation(**data_state)
        train_test_data = {
            'C_train': C_train,
            'C_test': C_test,
            'X_train': X_train,
            'X_test': X_test,
            'tcga_ids_train': tcga_ids_train,
            'tcga_ids_test': tcga_ids_test,
            'labels_train': labels_train,
            'labels_test': labels_test,
            'col_names': col_names,
        }

        experiment = experiment_params['experiment']
        n_bootstraps = experiment_params['n_bootstraps']
        val_split = experiment_params['val_split']
        load_saved = experiment_params['load_saved']

        test_df = run_experiment(experiment = experiment, n_bootstraps = n_bootstraps, 
                                 val_split = val_split, load_saved = load_saved, 
                                 train_test_data = train_test_data, disease = d)
        test_df_list.append(test_df)
    

    test_df_concat = pd.concat(test_df_list)
    test_df_concat.to_csv('./results/test_df_concat.csv', index = False) # save the concatenated file to directory
    plot_concat_mse(test_df_concat) # plot the concatenated file
    

#%%
data_state = {
        'num_features': 50,
        'pretransform_norm': False,
        'transform': 'pca',
        'feature_selection': 'population',
    }

experiment_params = {
    'experiment': "neighborhood",
    'n_bootstraps': 3,
    'val_split': 0.2,
    'load_saved': False,
}
    
#%%
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--num_features', type=int, default=50)
    parse.add_argument('--pretransform_norm', type=bool, default=False)
    parse.add_argument('--transform', type=str, default='pca')
    parse.add_argument('--feature_selection', type=str, default='population')
    parse.add_argument('--experiment', type=str, default='neighborhood')
    parse.add_argument('--n_bootstraps', type=int, default=3)
    parse.add_argument('--val_split', type=float, default=0.2)
    parse.add_argument('--load_saved', type=bool, default=False)

    data_state["num_features"] = parse.parse_args().num_features
    data_state["pretransform_norm"] = parse.parse_args().pretransform_norm
    data_state["transform"] = parse.parse_args().transform
    data_state["feature_selection"] = parse.parse_args().feature_selection
    experiment_params["experiment"] = parse.parse_args().experiment
    experiment_params["n_bootstraps"] = parse.parse_args().n_bootstraps
    experiment_params['val_split'] = parse.parse_args().val_split
    experiment_params['load_saved'] = parse.parse_args().load_saved

    load_disease_data(data_state, experiment_params)