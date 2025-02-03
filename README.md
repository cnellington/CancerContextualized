# CancerContextualized

This repository contains code and tutorials to reproduce and extend the results from "Learning to Estimate Sample-specific Transcriptional Networks for 7000 Tumors".

## Workflow
First, install dependencies
```
pip install -r requirements.txt
conda install bioconda::gdc-client
```

1. [Download](./01_download.ipynb): Download and clean data from TCGA.
2. [Compile](./02_compile.ipynb): Compile downloads into a single dataset and preprocess.
3. [Fit](./03_fit.ipynb): Learn to estimate contextualized transcriptional networks, as well as population, disease-specific, and cluster-specific transcriptional network baselines.
4. [Plot](./04_plot.ipynb): Use interactive data analysis tools to explore TCGA data with tumor-specific transcriptional networks, zeroing in on new tumor biology and defining new molecular subtypes.

