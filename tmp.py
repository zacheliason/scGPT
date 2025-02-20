import scanpy as sc
import torch
import torch.nn as nn

h5ad_file = (
    "/Users/zach/Documents/School/wei/zscGPT/data/adamson/perturb_processed.h5ad"
)

adata = sc.read_h5ad(h5ad_file)

print()


class PertFusion(nn.Module):
    def __init__(self, hidden_size, d_model):
        super(PertFusion, self).__init__()
        self.pert_fuse = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, d_model),  # Output d_model per position
        )

    def forward(self, x, current_seq_length):
        # Assume x has shape (batch_size, current_seq_length, hidden_size)
        x = self.pert_fuse(x)  # Shape: (batch_size, current_seq_length, d_model)
        return x


# Example usage for testing
if __name__ == "__main__":
    hidden_size = 5
    d_model = 10
    batch_size = 1
    current_seq_length = 10

    # Initialize the model
    model = PertFusion(hidden_size, d_model)

    # Create a sample input tensor
    x = torch.rand(batch_size, hidden_size)
    small_size = x.shape
    x_expanded = x.unsqueeze(1)  # (batch_size, 1, hidden_size)
    unsqueezed_shape = x_expanded.shape
    x_expanded = x_expanded.expand(-1, current_seq_length, -1)
    large_size = x_expanded.shape
    x2 = torch.rand(batch_size, 5000, hidden_size)

    l1 = x.shape
    l3 = x2.shape

    # Forward pass
    output = model.forward(x, current_seq_length)
    output2 = model.forward(x2, current_seq_length)
    l2 = output.shape
    l4 = output2.shape

    # Check the output shape
    assert output.shape == (batch_size, current_seq_length, d_model)
    print("Output shape is correct.")
