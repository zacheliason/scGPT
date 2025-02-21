import scanpy as sc

h5ad_file = "/Users/zach/Documents/School/wei/zscGPT/data/adamson/perturb_processed.h5ad"
h5ad_file = "/Users/zach/Documents/School/wei/pegasus/data/D18_diabetes_merged.h5ad"

adata = sc.read_h5ad(h5ad_file)
adata.obs.rename(columns={'genotype': 'condition'}, inplace=True)

# Filter out rows where 'condition' is missing
adata = adata[~adata.obs['condition'].isna(), :]

# Replace 'WT' with 'ctrl' in the 'condition' column
adata.obs['condition'] = adata.obs['condition'].replace('WT', 'ctrl')

# Check the result
print(adata.obs['condition'].value_counts())

look = adata.obs['condition']




print()
