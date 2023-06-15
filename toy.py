import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#test_df = pd.read_csv('./results/test_df_concat.csv')

#pop_df = test_df[test_df["Model"] == "Population"]

boot_means = test_df.groupby(['Model', 'Bootstrap']).mean().reset_index()
boot_means.groupby(['Model'])['MSE'].mean()
boot_means.groupby(['Model'])['MSE'].std()
plot_df = boot_means

fig, ax = plt.subplots(figsize=(6, 5))
sns.barplot(
    plot_df,
    x='Model',
    y='MSE',
    order = ['Population', 'Cluster-specific', 'Contextualized'],
    palette = ['lightblue', 'deepskyblue', 'orange'],
    errorbar='sd',
    capsize=0.05,
#     edgecolor='black',
    ax=ax
)

plt.title("Test Errors by Model (Disease Specific CV)", fontsize=14)
plt.tight_layout()
#plt.savefig(f'./results/test_model_aggregated.pdf', dpi=300)
plt.show()


average_mse = boot_means.groupby(['Model'])['MSE'].mean()
average_mse = average_mse.reset_index()
average_mse['std'] = boot_means.groupby(['Model'])['MSE'].std().values
average_mse.to_csv('./results/test_model_aggregated_v2.csv', index=False)


#%%


test_df_disease = pd.read_csv('/ix/djishnu/Hanxi/CancerContextualized/results/disease_holdouts/neighborhood-fit_intercept=True-val_split=0.2-n_bootstraps=2-dry_run=False-test=True-disease_test=LGG/mse_df.csv')

#%%
for d in all_diseases:
    # d will be LGG for example
    mse_df = experiment.main(d)
    disease_mse_df = mse_df[mse_df["Disease"] == d]

final_disease_mse_df = concatenate all disease_mse_df from each disease

grouped_bar_plot = use final_disease_mse_df to make grouped bar plot 
                   where each disease will be a group and consist of 3 bars (one for each model)
                    bar height = final_disease_mse_df["disease" = "LGG"]["MSE"].mean()
                    bar std = final_disease_mse_df["disease" = "LGG"]["MSE"].std()

aggregated_bar_plot = use final_disease_mse_df to make bar plot (not grouped anymore)
                      this plot will have 3 bars (one for each model)
                      bar height = final_disease_mse_df.groupby(["Model"])["MSE"].mean()
                      bar std = final_disease_mse_df.groupby(["Model"])["MSE"].std()

