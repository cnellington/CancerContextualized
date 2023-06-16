import pandas as pd
import experiments
import numpy as np
import argparse
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os


#%% if ever need to run the entire model, start here (work in progress)
# class Input:
#     def __init__(self, dict_args):
#         self.network_type = dict_args["network_type"]
#         self.fit_intercept = dict_args["fit_intercept"]
#         self.val_split = dict_args["val_split"]
#         self.n_bootstraps = dict_args["n_bootstraps"]
#         self.save_models = dict_args["save_models"]
#         self.load_saved = dict_args["load_saved"]
#         self.save_networks = dict_args["save_networks"]
#         self.result_dir = dict_args["result_dir"]
#         self.dry_run = dict_args["dry_run"]
#         self.num_features = dict_args["num_features"]
#         self.covar_projection = dict_args["covar_projection"]
#         self.transform = dict_args["transform"]
#         self.feature_selection = dict_args["feature_selection"]
#         self.test = dict_args["test"]
#         self.disease_test = dict_args["disease_test"]
#         self.project_to_dag = dict_args["project_to_dag"]



# def GetAllDiseaseLabels():
#     data_dir = './data/'
#     covars = pd.read_csv(data_dir + 'clinical_covariates.csv', header=0)
#     labels = np.unique(covars['disease_type'].values)
#     return labels


# def main(args):
#     all_mses = []
#     #print(args["disease_test"])
#     if args["disease_test"] == "all":
#         labels = GetAllDiseaseLabels()
#         for d in labels:
#             args['disease_test'] = d
#             dict_args = Input(args)
#             #print(vars(dict_args))
#             mse_df = experiments.main(**vars(dict_args))
#             all_mses.append(mse_df)
#     else:
#         dict_args = Input(args)
#         experiments.main(**vars(dict_args))
#%% subroutines
def remake_errorbars_disease(mean_df, std_df):
    means = mean_df.groupby("Model").mean().reset_index()
    means_upper = means.copy()
    means_upper["MSE"] += std_df["MSE"]
    means_lower = means.copy()
    means_lower["MSE"] -= std_df["MSE"]
    ret = pd.concat([means, means_upper, means_lower], axis=0)
    return ret

def write_plot_dfs(mean_df, std_df, d):
     # this function outputs the df that describes the plot (not necessarily the df that is plotted)
    std_df = std_df.rename(columns={'MSE': 'Std'})
    disease_cv_df = mean_df.drop(columns=['Bootstrap', 'Set']).reset_index()
    disease_cv_df = disease_cv_df.rename(columns={'MSE': 'Mean'})
    disease_cv_df['Disease'] = [d] * len(disease_cv_df)
    disease_cv_df = pd.merge(disease_cv_df, std_df, on='Model')
    return disease_cv_df

def plot_disease_cv(final_plot_df, result_dir):
    n_diseases = final_plot_df['Disease'].nunique()
    fig, ax = plt.subplots(figsize=(n_diseases + 5, 5))
    sns.barplot(
        final_plot_df,
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
    plt.ylim(0, 10)
    ax.plot(list(range(-1, n_diseases + 1)), [1] * (n_diseases + 2), linestyle='dashed', color='lightgrey')

    labels = [f'{label}' for label in final_plot_df['Disease'].unique()]
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=14)
    #ax.set_yticklabels([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], fontsize=14)

    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)


    plt.xlabel('Disease Type (# Training Samples)', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.title('Test Errors by Disease Type (Disease Specific CV)', fontsize=14)
    plt.tight_layout()

    plt.savefig(f'{result_dir}/test_diseaseCV.pdf', dpi=300)
    plt.show()
    plt.clf()

#%%
def main(args):
    #print(args)
    parent_folder = args["input_dir"]
    result_dir = args["result_dir"]

    mse_df_list = []
    for f in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, f)
        if os.path.isdir(folder_path): 
            for file_name in os.listdir(folder_path):
                if file_name == "mse_df.csv":
                    mse_df = pd.read_csv(os.path.join(folder_path, file_name))
                    mse_df_list.append(mse_df)

    print(len(mse_df_list))
    for i in range(len(mse_df_list)):
        df = mse_df_list[i]
        df = df.loc[df['Set'].isin(["Test", "Test (Bootstrapped)"])]
        assert sum(df["Set"].unique() == ["Test", "Test (Bootstrapped)"]) == 2, "Error in Set column..."
        mse_df_list[i] = df
    all_mse_df = pd.concat(mse_df_list)

    plot_dfs = []
    disease_cv_dfs = []
    for d in all_mse_df["Disease"].unique():
        disease_df = all_mse_df.loc[all_mse_df["Disease"] == d]
        mses_by_disease = (
                    disease_df.drop(columns="sample_id")
                    .groupby(["Bootstrap", "Model", "Set"])
                    .mean()
                    .reset_index()
                )
        # get the bar height
        mean_df = mses_by_disease[mses_by_disease["Set"] == 'Test (Bootstrapped)']
        std_df = mses_by_disease[mses_by_disease["Set"] == 'Test'].groupby(['Model']).std().reset_index()
        
        # make the df for plotting
        plot_df = remake_errorbars_disease(mean_df, std_df)
        plot_df["Disease"] = [d] * len(plot_df)
        plot_dfs.append(plot_df)

        # make the df to output that describes the plot
        disease_cv_df = write_plot_dfs(mean_df, std_df, d)
        disease_cv_dfs.append(disease_cv_df.drop(columns=['index']))


        final_plot_df = pd.concat(plot_dfs)
        final_disease_cv_df = pd.concat(disease_cv_dfs)
        final_disease_cv_df.to_csv(f"{result_dir}/disease_cv_df.csv", index=False)

        plot_disease_cv(final_plot_df, result_dir)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    parser = argparse.ArgumentParser()
    # parser.add_argument("--network_type", type=str, default="neighborhood")
    # parser.add_argument("--fit_intercept", default=False, action="store_true")
    # parser.add_argument("--val_split", type=float, default=0.2)
    # parser.add_argument("--n_bootstraps", type=int, default=2)
    # parser.add_argument("--save_models", default=False, action="store_true")
    # parser.add_argument("--load_saved", default=False, action="store_true")
    # parser.add_argument("--save_networks", default=False, action="store_true")
    parser.add_argument("--input_dir", type=str) # a folder that contains subfolder with resutls for each disease
    parser.add_argument("--result_dir", type=str, default="experiment")
    # parser.add_argument("--dry_run", default=False, action="store_true")
    # parser.add_argument("--num_features", type=int, default=50)
    # parser.add_argument("--covar_projection", type=int, default=200)
    # parser.add_argument("--transform", type=str, default="pca")
    # parser.add_argument("--feature_selection", type=str, default="population")
    # parser.add_argument("--test", default=False, action="store_true")
    # # for disease_test, either put in "all" or a disease name
    # parser.add_argument("--disease_test", type=str, default=None)
    # parser.add_argument("--project_to_dag", default=False, action="store_true")
    #parser.add_argument("--run_all_disease", default=False, action="store_true")
    args = parser.parse_args()
    # CLI overrides
    #if args.disease_test is not None:
    #args.test = True
    print(vars(args))
    main(vars(args)) # call main with all the arguments
    


#%%

