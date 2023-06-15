import pandas as pd
import experiments
import numpy as np
import argparse
import torch
import seaborn as sns
import matplotlib.pyplot as plt

#get all disease labels

class Input:
    def __init__(self, dict_args):
        self.network_type = dict_args["network_type"]
        self.fit_intercept = dict_args["fit_intercept"]
        self.val_split = dict_args["val_split"]
        self.n_bootstraps = dict_args["n_bootstraps"]
        self.save_models = dict_args["save_models"]
        self.load_saved = dict_args["load_saved"]
        self.save_networks = dict_args["save_networks"]
        self.result_dir = dict_args["result_dir"]
        self.dry_run = dict_args["dry_run"]
        self.num_features = dict_args["num_features"]
        self.covar_projection = dict_args["covar_projection"]
        self.transform = dict_args["transform"]
        self.feature_selection = dict_args["feature_selection"]
        self.test = dict_args["test"]
        self.disease_test = dict_args["disease_test"]
        self.project_to_dag = dict_args["project_to_dag"]



def GetAllDiseaseLabels():
    data_dir = './data/'
    covars = pd.read_csv(data_dir + 'clinical_covariates.csv', header=0)
    labels = np.unique(covars['disease_type'].values)
    return labels


def main(args):
    all_mses = []
    #print(args["disease_test"])
    if args["disease_test"] == "all":
        labels = GetAllDiseaseLabels()
        for d in labels:
            args['disease_test'] = d
            dict_args = Input(args)
            #print(vars(dict_args))
            mse_df = experiments.main(**vars(dict_args))
            all_mses.append(mse_df)
    else:
        dict_args = Input(args)
        experiments.main(**vars(dict_args))
    #print(dict_args)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_type", type=str, default="neighborhood")
    parser.add_argument("--fit_intercept", default=False, action="store_true")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--n_bootstraps", type=int, default=2)
    parser.add_argument("--save_models", default=False, action="store_true")
    parser.add_argument("--load_saved", default=False, action="store_true")
    parser.add_argument("--save_networks", default=False, action="store_true")
    parser.add_argument("--result_dir", type=str, default="experiment")
    parser.add_argument("--dry_run", default=False, action="store_true")
    parser.add_argument("--num_features", type=int, default=50)
    parser.add_argument("--covar_projection", type=int, default=200)
    parser.add_argument("--transform", type=str, default="pca")
    parser.add_argument("--feature_selection", type=str, default="population")
    parser.add_argument("--test", default=False, action="store_true")
    # for disease_test, either put in "all" or a disease name
    parser.add_argument("--disease_test", type=str, default=None)
    parser.add_argument("--project_to_dag", default=False, action="store_true")
    #parser.add_argument("--run_all_disease", default=False, action="store_true")
    args = parser.parse_args()
    # CLI overrides
    #if args.disease_test is not None:
    args.test = True
    print(vars(args))
    main(vars(args)) # call main with all the arguments
    



#%%

mse_df_1 = pd.read_csv('/ix/djishnu/Hanxi/CancerContextualized/results/disease_holdout_test/neighborhood-fit_intercept=False-val_split=0.2-n_bootstraps=2-dry_run=False-test=True-disease_test=BLCA/mse_df.csv')
mse_df_2 = pd.read_csv('/ix/djishnu/Hanxi/CancerContextualized/results/disease_holdout_test/neighborhood-fit_intercept=True-val_split=0.2-n_bootstraps=2-dry_run=False-test=True-disease_test=LGG/mse_df.csv')
mse_df_list = [mse_df_1, mse_df_2]

# first concatename the input mse_df from each disease into 1 df
for i in range(len(mse_df_list)):
    df = mse_df_list[i]
    df = df.loc[df['Set'].isin(["Test", "Test (Bootstrapped)"])]
    assert sum(df["Set"].unique() == ["Test", "Test (Bootstrapped)"]) == 2, "Error in Set column..."
    mse_df_list[i] = df
all_mse_df = pd.concat(mse_df_list)

# heavy lifting code starts here
plot_dfs = []
for d in all_mse_df["Disease"].unique():
    # first deal with each disease
    disease_df = all_mse_df.loc[all_mse_df["Disease"] == d]
    # only use bootstrapped samples for bar height
    mean_df = disease_df[disease_df["Set"] == "Test (Bootstrapped)"]
    mean_df = mean_df.groupby('Model').mean().reset_index()

    # use Test samples for error bart
    std_df = disease_df[disease_df["Set"] == "Test"]
    err = []
    for m in range(len(std_df['Model'].unique())):
        tmp = std_df[std_df['Model'] == std_df['Model'].unique()[m]]
        tmp = tmp.groupby('Bootstrap').mean().reset_index()
        err.append(tmp['MSE'].std())

    mean_df["Err"] = err
    mean_df["Disease"] = [d] * len(all_mse_df["Model"].unique())
    plot_dfs.append(mean_df)

plot_df = pd.concat(plot_dfs)

# plot the bar plot
vals = plot_df.pivot(index='Disease', columns='Model', values='MSE')
yerr = plot_df.pivot(index='Disease', columns='Model', values='Err')
ax = vals.plot(kind='bar', yerr=yerr, logy=True, rot=0, figsize=(6, 5))
_ = ax.legend(title='Sample Set', bbox_to_anchor=(1, 1.02), loc='upper left')