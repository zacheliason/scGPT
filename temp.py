import re

import scanpy as sc
from gears import PertData
from matplotlib import pyplot as plt

# Set the directory where data will be stored or loaded
data_directory = "/Users/zach/Documents/School/wei/pegasus/data/diabetes"

# Path to your .h5ad file
diabetes_filepath = (
    "/Users/zach/Documents/School/wei/pegasus/data/D18_diabetes_merged.h5ad"
)

adata = sc.read_h5ad(diabetes_filepath)

# Explore the 'Diabetes_status' column in the adata object
diabetes_status = adata.obs["Diabetes_status"]
gene = adata.obs["gene"]
print("Diabetes Status Distribution:")
print(diabetes_status.value_counts())


# Initialize the PertData object with the directory path
pert_data = PertData(data_path=data_directory)
pert_data.new_data_process(
    dataset_name="diabetes", adata=sc.read_h5ad(diabetes_filepath)
)

# Load the .h5ad file
pert_data.load(data_path=diabetes_filepath)

with open("/Users/zach/Downloads/new.txt", "r") as f:
    text = f.read()

pattern = r"epoch\s+(\d+)\s+\|\s+\d+/(\d+)\s+batches.*ms\/batch\s+(\d+\.*\d*).*loss\s+(\d+\.\d+)"
# "scGPT - INFO - | epoch   1 | 1500/1698 batches | lr 0.0001 | ms/batch 507.93 | loss  0.08 | mse  0.16 |"


# Find all matches in the text
matches = re.findall(pattern, text)

# Extract epoch, batch, and loss values
epochs = []
batches = []
losses = []
for match in matches:
    epoch = int(match[0])
    batch = int(match[1])
    ms_batch = float(match[2])
    loss = float(match[3])
    epochs.append(epoch)
    batches.append(batch)
    losses.append(ms_batch)

# Plot loss over epochs
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(losses)), losses, label="Loss", color="#f44f00")
plt.xticks(range(len(losses)), epochs)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

# Plot loss over batches
plt.subplot(1, 2, 2)
plt.plot(batches, losses, "o-", label="Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Loss over Batches")
plt.legend()

plt.tight_layout()
plt.show()
