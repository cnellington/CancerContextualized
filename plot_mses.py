import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def plot_mses(mse_df, set_label, savedir, relative_to='Contextualized', ymax=None, ticksize=20, labelsize=32, order=['Population', 'Cluster-specific', 'Disease-specific', 'Contextualized'], nocount=False):
    palette_map = {
        'Population': 'lightblue',
        'Cluster-specific': 'deepskyblue',
        'Disease-specific': 'royalblue',
        'Contextualized': 'orange',
    }
    # This inner function is used to create dataframes with the upper and lower bounds of the error bars for the plots.
    def remake_errorbars(mean_df, std_df, std_label, groupby=["Model", "Disease"]):
        # Calculate standard deviation for each group
        stds = std_df[std_df["Set"] == std_label].groupby(groupby).std().reset_index()
        # Calculate mean for each group
        means = mean_df.groupby(groupby).mean().reset_index()
        # Create upper bound of the error bar by adding standard deviation to the mean
        means_upper = means.copy()
        means_upper["MSE"] += stds["MSE"]
        # Create lower bound of the error bar by subtracting standard deviation from the mean
        means_lower = means.copy()
        means_lower["MSE"] -= stds["MSE"]
        # Combine means and their upper and lower bounds into a single dataframe
        ret = pd.concat([means, means_upper, means_lower], axis=0)
        return ret

    # Get population errors for plotting. 
    # For Train and Test, average all MSEs within each bootstrap, and use the standard deviation of the bootstraps as the error bar. 
    # For Train (Bootstrapped) and Test (Bootstrapped) the set was created by averaging bootstraps into a single prediction.
    # In this case, remake the error bars using the standard deviation of the Train or Test set bootstraps are the error bars.
    mses_by_disease = (
        mse_df.drop(columns="sample_id")
        .groupby(["Bootstrap", "Model", "Set"])
        .mean()
        .reset_index()
    )
    set_df = mses_by_disease[mses_by_disease["Set"] == set_label]
    if set_label == "Train (Bootstrapped)":
        plot_df = remake_errorbars(
            set_df, mses_by_disease, "Train", groupby=["Model"]
        )
    elif set_label == "Test (Bootstrapped)":
        plot_df = remake_errorbars(
            set_df, mses_by_disease, "Test", groupby=["Model"]
        )
    else:
        plot_df = set_df

    if 'Disease-specific' not in mse_df['Model'] and 'Disease-specific' in order:  # Remove Disease-specific when we don't have it
        order.remove('Disease-specific')
    palette = [palette_map[model] for model in order]
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(
        plot_df,
        x="Model",
        y="MSE",
        order=order,
        palette=palette,
        errorbar="sd",
        capsize=0.05,
        ax=ax,
    )
    plt.xlim(-1, 4)
    ax.plot([-1, 5], [1, 1], linestyle="dashed", color="lightgrey")
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=ticksize)
    if ymax is not None:
        plt.ylim(0, ymax)
    plt.yticks(fontsize=ticksize)
    plt.xlabel("Method", fontsize=labelsize)
    plt.ylabel("MSE", fontsize=labelsize)
    # plt.title(f"{set_label} Errors", fontsize=labelsize)
    plt.tight_layout()
    plt.savefig(f"{savedir}/mse_{set_label.lower()}.pdf", dpi=300)
    plt.clf()

    # Plot the contextualized model errors relative to the other models errors at the population level.
    # So if the Cluster-specific bar is at 0.56, then the Contextualized model average error is 0.56 * Cluster-specific error.
    # All basleines need to be below 1 for the contextualized model to be better than the baselines.
    # Contextualized will always have a relative error of 1, since contextualized is the reference
    plot_df["Relative MSE"] = (
        plot_df["MSE"] / plot_df[plot_df["Model"] == relative_to]["MSE"].mean()
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(
        plot_df,
        x="Model",
        y="Relative MSE",
        order=order,
        palette=palette,
        errorbar="sd",
        capsize=0.05,
        ax=ax,
    )
    plt.xlim(-1, 4)
    ax.plot([-1, 5], [1, 1], linestyle="dashed", color="lightgrey")
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=ticksize)
    if ymax is not None:
        plt.ylim(0, ymax)
    plt.yticks(fontsize=ticksize)
    plt.xlabel("Method", fontsize=labelsize)
    plt.ylabel("Relative MSE", fontsize=labelsize)
    # plt.title(f"{set_label} Errors", fontsize=labelsize)
    plt.tight_layout()
    plt.savefig(f"{savedir}/relative_mse_to_{relative_to}_{set_label.lower()}.pdf", dpi=300)
    plt.clf()

    # Save the relative error values to quickly check the exact relative errors
    relative_df = (
        plot_df.groupby(["Model"])
        .agg({"Relative MSE": ["mean", "std"]})
        .reset_index()[["Model", "Relative MSE"]]
    )
    relative_df.columns = [" ".join(col).strip(" ") for col in relative_df.columns]
    relative_df.to_csv(
        f"{savedir}/relative_mses_{set_label.lower()}.csv", index=False
    )

    # Same as the population plot above, but for each disease type
    mses_by_disease = (
        mse_df.drop(columns="sample_id")
        .groupby(["Bootstrap", "Model", "Set", "Disease"])
        .mean()
        .reset_index()
    )
    set_df = mses_by_disease[mses_by_disease["Set"] == set_label]
    if set_label == "Train (Bootstrapped)":
        plot_df = remake_errorbars(
            set_df, mses_by_disease, "Train", groupby=["Model", "Disease"]
        )
    elif set_label == "Test (Bootstrapped)":
        plot_df = remake_errorbars(
            set_df, mses_by_disease, "Test", groupby=["Model", "Disease"]
        )
    else:
        plot_df = set_df
    n_diseases = len(plot_df["Disease"].unique())
    fig, ax = plt.subplots(figsize=(n_diseases + 5, 5))
    sns.barplot(
        plot_df,
        x="Disease",
        y="MSE",
        hue="Model",
        hue_order=order,
        palette=palette,
        errorbar="sd",
        capsize=0.05,
        ax=ax,
    )
    plt.xlim(-1, n_diseases)
    if ymax is not None:
        plt.ylim(0, ymax)
    plt.yticks(fontsize=ticksize)
    ax.plot(
        list(range(-1, n_diseases + 1)),
        [1] * (n_diseases + 2),
        linestyle="dashed",
        color="lightgrey",
    )

    # Show the number of training samples for each disease on the x-axis by remaking the labels with the disease training sample counts
    test_datapoints = (
        mse_df[
            (mse_df["Set"] == "Test")
            & (mse_df["Model"] == "Contextualized")
            & (mse_df["Bootstrap"] == '0')
        ]["Disease"]
        .value_counts()
        .sort_index()
    )
    if nocount:
        x_labels = [label for label, count in test_datapoints.items()]
    else:
        train_datapoints = (
            mse_df[
                (mse_df["Set"] == "Train")
                & (mse_df["Model"] == "Contextualized")
                & (mse_df["Bootstrap"] == '0')
            ]["Disease"]
            .value_counts()
            .sort_index()
        )
        train_disease_counts = {
            label: count for label, count in train_datapoints.iteritems()
        }
        if "Train" in set_label:
            x_labels = [
                f"{label} ({count})" for label, count in train_datapoints.iteritems()
            ]
        else:
            x_labels = [
                f"{label} ({train_disease_counts[label] if label in train_disease_counts else 0})"
                for label, count in test_datapoints.iteritems()
            ]
    ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=ticksize)
    if ymax is not None:
        plt.ylim(0, ymax)
    plt.yticks(fontsize=ticksize)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=ticksize)
    plt.xlabel("Disease Type (# Training Samples)", fontsize=labelsize)
    plt.ylabel("MSE", fontsize=labelsize)
    # plt.title(f"{set_label} Errors by Disease Type", fontsize=labelsize)
    plt.tight_layout()
    plt.savefig(f"{savedir}/mse_by_disease_{set_label.lower()}.pdf", dpi=300)
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mses', type=str, nargs='+', required=True)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--set', type=str, default='Test (Bootstrapped)')
    # parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--relto', type=str, default='Contextualized')
    parser.add_argument('--ymax', type=float, default=None)
    parser.add_argument('--ticksize', type=int, default=18)
    parser.add_argument('--labelsize', type=int, default=24)
    parser.add_argument('--order', type=str, nargs='+', default=['Population', 'Cluster-specific', 'Disease-specific', 'Contextualized'])
    args = parser.parse_args()
    mse_dfs = []
    for mse_df_path in args.mses:
        df = pd.read_csv(mse_df_path, header=0)
        df['Bootstrap'] = df['Bootstrap'].astype(str)
        # df = df[(df['Set'] == 'Test') | (df['Set'] == 'Test (Bootstrapped)')]
        mse_dfs.append(df)
    mse_df = pd.concat(mse_dfs, axis=0)
    mse_df['Bootstrap'] = mse_df['Bootstrap'].astype(str)
    plot_mses(mse_df, args.set, args.savedir, args.relto, ymax=args.ymax, ticksize=args.ticksize, labelsize=args.labelsize, order=args.order, nocount=len(mse_dfs) > 1)
