import os
import numpy as np
#import pickle as pkl
import pandas as pd
from disease_dataloader import disease_load_data, disease_data_transformation

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
        



#%%
def load_disease_data(data_state):
    # unfinished function  needs more work depending on how to integrate the actual running model part...

    data_dir = './data/' # hard coded for now
    covars = pd.read_csv(data_dir + 'clinical_covariates.csv', header=0)
    labels = np.unique(covars['disease_type'].values)

    check_disease_data(labels) 

    for d in labels:
        # data_state = {
        #         'num_features': 50,
        #         'pretransform_norm': False,
        #         'transform': 'pca',
        #         'feature_selection': 'population',
        #         "disease_label": d,
        #     }

        data_state["disease_label"] = d
        C_train, C_test, X_train, X_test, tcga_ids_train, tcga_ids_test, labels_train, labels_test, col_names = disease_data_transformation(**data_state)

        # add code here for training model and calculating performance metrics...
    

#%%
data_state = {
        'num_features': 50,
        'pretransform_norm': False,
        'transform': 'pca',
        'feature_selection': 'population',
    }


load_disease_data(data_state)
    

