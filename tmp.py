import scanpy as sc

from scgpt.gears_utils import PertData

# Load your custom dataset
h5ad_file = "/Users/zach/Documents/School/wei/pegasus/data/D18_diabetes_merged.h5ad"
# h5ad_file = (
#     "/Users/zach/Documents/School/wei/zscGPT/data/adamson/perturb_processed.h5ad"
# )
adata = sc.read_h5ad(h5ad_file)

# Rename columns if necessary
adata.obs.rename(
    columns={"genotype": "condition", "celltype": "cell_type"}, inplace=True
)

# Filter out rows where 'condition' is missing
adata = adata[~adata.obs["condition"].isna(), :]

# Replace 'WT' with 'ctrl' in the 'condition' column
adata.obs["condition"] = adata.obs["condition"].apply(lambda x: x + "+ctrl")
adata.obs["condition"] = adata.obs["condition"].replace("WT+ctrl", "ctrl")
adata.var["gene_name"] = adata.var.index

condition_counts = adata.obs["condition"].value_counts()

# Identify conditions with fewer than 2 samples
invalid_conditions = condition_counts[condition_counts < 2].index.tolist()

# Filter out these conditions
adata = adata[~adata.obs["condition"].isin(invalid_conditions), :]

adata.obs["group"] = (
    adata.obs["cell_type"].astype(str) + "_" + adata.obs["condition"].astype(str)
)

group_counts = adata.obs["group"].value_counts()
invalid_groups = group_counts[group_counts < 2].index.tolist()
adata = adata[~adata.obs["group"].isin(invalid_groups), :]

# Verify
print(f"Removed conditions with fewer than 2 samples: {invalid_conditions}")
print(adata.obs["condition"].value_counts())

# look = adata.obs['celltype']

# Ensure necessary columns are present
assert "condition" in adata.obs.columns, "Please specify condition"
assert "gene_name" in adata.var.columns, "Please specify gene name"
assert "cell_type" in adata.obs.columns, "Please specify cell type"

look = adata.obs.sort_values("condition")

if "GATA4" in adata.obs["condition"].unique():
    look2 = adata.obs[adata.obs["condition"] == "SC-EC_GATA4_1"]
    look2 = look2.sort_values("condition")
else:
    look2 = None

data_path = "data/diabetes"  # Specify the path where you want to save/load the data
pert_data = PertData(data_path)

dataset_name = "diabetes"  # Give your dataset a name
pert_data.new_data_process(dataset_name, adata=adata, skip_calc_de=False)

print()
