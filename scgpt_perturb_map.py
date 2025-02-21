import argparse
import contextlib
import gc
import json
import os
import time
import warnings
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
if not torch.cuda.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

from gears.inference import deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction
from torch import nn
from torch_geometric.loader import DataLoader
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from torchtext.vocab import Vocab

import scgpt as scg
from scgpt.gears_utils import PertData
from scgpt.loss import (
    masked_mse_loss,
)
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import (
    compute_perturbation_metrics,
    set_seed,
)


def prepare_data(
    batch_size,
    test_batch_size,
    data_dir,
    data_name="adamson",
    split="simulation",
):
    pert_data = PertData(data_dir)
    pert_data.load(data_name=data_name)
    pert_data.prepare_split(split=split, seed=1)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=test_batch_size)
    # pert_data = clear_pert_data_memory(pert_data)

    genes = pert_data.adata.var["gene_name"].tolist()
    n_genes = len(genes)
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])

    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )

    return pert_data, vocab, gene_ids, n_genes


def setup_model(
    model_dir, vocab, genes, default_config, max_seq_len, include_zero_gene, gene_ids
):
    print(f"initial vocab length {len(vocab)}")

    # Unpack default model settings
    embsize = default_config["embsize"]
    d_hid = default_config["d_hid"]
    nlayers = default_config["nlayers"]
    nhead = default_config["nhead"]
    n_layers_cls = default_config["n_layers_cls"]
    dropout = default_config["dropout"]
    use_fast_transformer = default_config["use_fast_transformer"]
    pretrained_dict = None

    print()
    print(model_dir)
    print()
    # Update model settings from checkpoint, if provided
    if model_dir is not None and os.path.exists(model_dir):
        model_config_file = os.path.join(model_dir, "args.json")
        model_file = os.path.join(model_dir, "best_model.pt")
        vocab_file = os.path.join(model_dir, "vocab.json")

        # Update vocab if loading from existing model
        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        # Check gene vocabulary matching
        gene_ids_in_vocab = np.array([1 if gene in vocab else -1 for gene in genes])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )

        # Load model configurations
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)

        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )

        # Extract model parameters
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]

        # Load pretrained weights
        pretrained_dict = torch.load(model_file, map_location=torch.device(device))
        pretrained_dict = convert_wqkv_to_in_proj(pretrained_dict)
    else:
        genes = pert_data.adata.var["gene_name"].tolist()
        vocab = Vocab(
            VocabPybind(genes + special_tokens, None)
        )  # bidirectional lookup [gene <-> int]

    ntokens = len(vocab)  # size of vocabulary
    seq_len = min(n_genes, max_seq_len)
    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        pert_pad_id=pert_pad_id,
        use_fast_transformer=use_fast_transformer,
        device=device,
        pert_data=pert_data,
        seq_len=seq_len,
        include_zero_gene=include_zero_gene,
        max_seq_len=max_seq_len,
        gene_ids=gene_ids,
    )

    # Load pretrained weights
    if (
        load_param_prefixs is not None
        and model_dir is not None
        and pretrained_dict is not None
    ):
        # only load params that start with the prefix
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if any([k.startswith(prefix) for prefix in load_param_prefixs])
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    elif model_dir is not None:
        try:
            model.load_state_dict(torch.load(model_file))
            logger.info(f"Loading all model params from {model_file}")
        except RuntimeError:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    model.to(device)
    enable_gradient_checkpointing(model)

    return model


def print_memory_usage():
    # RAM usage
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024 / 1024  # in MB

    # GPU memory if available
    gpu_usage = None
    print(f"RAM Usage: {ram_usage:.2f} MB")
    if gpu_usage is not None:
        print(f"GPU Memory Usage: {gpu_usage:.2f} MB")


def clear_pert_data_memory(pert_data):
    # List of attributes that can be safely cleared
    attributes_to_clear = [
        "processed_data",
        "raw_data",
        "train_control_data",
        "val_control_data",
        "test_control_data",
    ]

    # Clear attributes
    for attr in attributes_to_clear:
        if hasattr(pert_data, attr):
            delattr(pert_data, attr)

    # Force garbage collection
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nMemory usage after clearing:")
    print_memory_usage()

    return pert_data


def convert_wqkv_to_in_proj(pretrained_state_dict):
    converted_state_dict = {}

    # for key, value in pretrained_state_dict.items():
    for key, value in pretrained_state_dict.items():
        # if "self_attn.Wqkv.weight" in key:
        # print(key)
        if "self_attn.in_proj_weight" in key or "self_attn.Wqkv.weight" in key:
            # print(f"{value.shape}\t{key}")
            converted_state_dict[
                key.replace("self_attn.Wqkv.weight", "self_attn.in_proj_weight")
            ] = value
        elif "self_attn.in_proj_bias" in key or "self_attn.Wqkv.bias" in key:
            # print(f"{value.shape}\t{key}")
            converted_state_dict[
                key.replace("self_attn.Wqkv.bias", "self_attn.in_proj_bias")
            ] = value
        else:
            converted_state_dict[key] = value

    return converted_state_dict


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    n_genes: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: any,
) -> float:
    """
    Train the model for one epoch with gradient accumulation.
    Returns the average loss for the epoch.
    """
    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    optimizer.zero_grad()  # Zero gradients at the start

    for batch, batch_data in enumerate(train_loader):
        actual_batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)

        context_manager = (
            torch.cuda.amp.autocast()
            if torch.cuda.is_available()
            else contextlib.nullcontext()
        )

        with context_manager:
            output_dict = model(
                batch_data,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
            )
            output_values = output_dict["mlm_output"]
            target_values = output_dict["target_values"]

            masked_positions = torch.ones_like(
                target_values, dtype=torch.bool
            )  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

            # Normalize loss by accumulation steps
            loss = loss / accumulation_steps

        # Accumulate gradients
        scaler.scale(loss).backward()

        # Update weights only after accumulating gradients for specified steps
        if (batch + 1) % accumulation_steps == 0 or (batch + 1) == num_batches:
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )

            # Step optimizer and scaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()  # Reset gradients

        # Multiply loss by accumulation_steps to get the equivalent loss for logging
        total_loss += loss.item() * accumulation_steps
        total_mse += loss_mse.item() * accumulation_steps

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()

    return total_loss / num_batches


def eval_perturb(
    loader: DataLoader, model: TransformerGenerator, device: torch.device
) -> Dict:
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            p = model.pred_perturb(
                batch,
                include_zero_gene=include_zero_gene,
                # gene_ids=gene_ids,
            )
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(np.float64)
    results["truth"] = truth.detach().cpu().numpy().astype(np.float64)

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(np.float64)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(np.float64)

    return results


def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing for a model."""
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    else:
        # Manual implementation for models without built-in support
        for module in model.modules():
            if hasattr(module, "checkpoint") and not module.checkpoint:
                module.checkpoint = True


def predict(
    model: TransformerGenerator, pert_list: List[str], pool_size: Optional[int] = None
) -> Dict:
    """
    Predict the gene expression values for the given perturbations.

    Args:
        model (:class:`torch.nn.Module`): The model to use for prediction.
        pert_list (:obj:`List[str]`): The list of perturbations to predict.
        pool_size (:obj:`int`, optional): For each perturbation, use this number
            of cells in the control and predict their perturbation results. Report
            the stats of these predictions. If `None`, use all control cells.
    """
    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)
    gene_list = pert_data.gene_names.values.tolist()
    for pert in pert_list:
        for i in pert:
            if i not in gene_list:
                raise ValueError(
                    "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
                )

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        results_pred = {}
        for pert in pert_list:
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )
            loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
            preds = []
            for batch_data in loader:
                pred_gene_values = model.pred_perturb(
                    batch_data,
                    include_zero_gene,
                    amp=amp,  # gene_ids=gene_ids,
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

    return results_pred


def plot_perturbation(
    model: nn.Module, query: str, save_file: str = None, pool_size: int = None
) -> matplotlib.figure.Figure:
    sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

    adata = pert_data.adata
    gene2idx = pert_data.node_map
    cond2name = dict(adata.obs[["condition", "condition_name"]].values)
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

    de_idx = [
        gene2idx[gene_raw2id[i]]
        for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    genes = [
        gene_raw2id[i] for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
    if query.split("+")[1] == "ctrl":
        pred = predict(model, [[query.split("+")[0]]], pool_size=pool_size)
        pred = pred[query.split("+")[0]][de_idx]
    else:
        pred = predict(model, [query.split("+")], pool_size=pool_size)
        pred = pred["_".join(query.split("+"))][de_idx]
    ctrl_means = adata[adata.obs["condition"] == "ctrl"].to_df().mean()[de_idx].values

    pred = pred - ctrl_means
    truth = truth - ctrl_means

    fig, ax = plt.subplots(figsize=[16.5, 4.5])
    plt.title(query)
    plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))

    for i in range(pred.shape[0]):
        _ = plt.scatter(i + 1, pred[i], color="red")

    plt.axhline(0, linestyle="dashed", color="green")

    ax.xaxis.set_ticklabels(genes, rotation=90)

    plt.ylabel("Change in Gene Expression over Control", labelpad=10)
    plt.tick_params(axis="x", which="major", pad=5)
    plt.tick_params(axis="y", which="major", pad=5)
    sns.despine()

    if save_file:
        fig.savefig(save_file, bbox_inches="tight", transparent=False)

    return fig


# CONFIG
# ---------------------------------------------------------------------------------------
print("Beginning scGPT-perturbation configuration...")

if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available memory

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")
set_seed(42)

# settings for data prcocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 0
include_zero_gene = "all"
max_seq_len = 1536

# settings for training
MLM = True  # whether to use masked language modeling, currently it is always on.
CLS = False  # celltype classification objective
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
amp = True

# settings for optimizer
lr = 1e-4  # or 1e-4
batch_size = 32
eval_batch_size = 32
accumulation_steps = 2
epochs = 1  # 15
schedule_interval = 1
early_stop = 10

# settings for the model
default_model_settings = {
    "embsize": 512,  # embedding dimension
    "d_hid": 512,  # dimension of the feedforward network model in nn.TransformerEncoder
    "nlayers": 12,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    "nhead": 8,  # number of heads in nn.MultiheadAttention
    "n_layers_cls": 3,
    "dropout": 0,  # dropout probability
    "use_fast_transformer": True,  # whether to use fast transformer
}

# logging
log_interval = 100


# dataset and evaluation choices
data_name = "adamson"
split = "simulation"
if data_name == "norman":
    perts_to_plot = ["SAMD1+ZBTB1"]
elif data_name == "adamson":
    perts_to_plot = ["KCTD16+ctrl"]


# SETUP
# ---------------------------------------------------------------------------------------
print("Setting up scGPT-perturbation...")
parser = argparse.ArgumentParser(description="scGPT Perturb")
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    required=True,
    help="Directory containing the data",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    required=True,
    help="Directory to save the output",
)
args = parser.parse_args()

print("Data directory: ", args.data_dir)
print("output directory: ", args.output_dir)

data_dir = args.data_dir
output_dir = args.output_dir
save_dir = os.path.join(
    data_dir, "save", f"dev_perturb_{data_name}-{time.strftime('%b%d-%H-%M')}"
)

os.makedirs(save_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

load_model = os.path.join(data_dir, "scGPT_human")
load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]


logger = scg.logger
scg.utils.add_file_handler(logger, os.path.join(save_dir, "run.log"))
logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")


# RUN
# ---------------------------------------------------------------------------------------
print("Running scGPT-perturbation...")
pert_data, vocab, gene_ids, n_genes = prepare_data(
    batch_size, eval_batch_size, data_dir, data_name, split
)

gene_names = pert_data.gene_names.values.tolist()

model = setup_model(
    model_dir=load_model,
    vocab=vocab,
    genes=pert_data.adata.var["gene_name"].tolist(),
    default_config=default_model_settings,
    max_seq_len=max_seq_len,
    include_zero_gene=include_zero_gene,
    gene_ids=gene_ids,
)

# criterion_cls = nn.CrossEntropyLoss()
criterion = masked_mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)

scaler = torch.amp.GradScaler(enabled=amp)

best_val_loss = float("inf")
best_val_corr = 0
sest_model = None
patience = 0
best_model = None


train_losses = []
val_losses = []

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_loader = pert_data.dataloader["train_loader"]
    valid_loader = pert_data.dataloader["val_loader"]

    print(f"\rEpoch {epoch}/{epochs} | ", end="", flush=True)

    # Train the model and get the average loss for the epoch
    train_loss = train(
        model=model,
        train_loader=train_loader,
        n_genes=n_genes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    train_losses.append(train_loss)

    # Evaluate on the validation set
    val_res = eval_perturb(valid_loader, model, device)
    val_metrics = compute_perturbation_metrics(
        val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    )

    val_res["pred"] = torch.from_numpy(val_res["pred"])
    val_res["truth"] = torch.from_numpy(val_res["truth"])

    # Create the mask
    masked_positions = torch.ones_like(val_res["truth"], dtype=torch.bool)

    # Compute the loss
    loss = loss_mse = criterion(val_res["pred"], val_res["truth"], masked_positions)
    val_losses.append(loss)  # Assuming val_metrics contains a "loss" key

    print(
        f"Val Pearson: {val_metrics['pearson']:5.4f} | Time: {time.time() - epoch_start_time:5.2f}s",
        end="\r",
        flush=True,
    )

    logger.info(f"val_metrics at epoch {epoch}: ")
    logger.info(val_metrics)

    elapsed = time.time() - epoch_start_time
    logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")

    val_score = val_metrics["pearson"]
    if val_score > best_val_corr:
        best_val_corr = val_score
        best_model = copy.deepcopy(model)
        logger.info(f"Best model with score {val_score:5.4f}")
        patience = 0
    else:
        patience += 1
        if patience >= early_stop:
            logger.info(f"Early stop at epoch {epoch}")
            break

    torch.save(
        model.state_dict(),
        os.path.join(save_dir, f"model_{epoch}.pt"),
    )

    scheduler.step()

# Plot the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

# Save the plot to the output directory
loss_plot_path = os.path.join(output_dir, "loss_plot.png")
plt.savefig(loss_plot_path)
plt.close()

logger.info(f"Loss plot saved to {loss_plot_path}")

""" ## Evaluations"""

if best_model is None:
    best_model = model

for p in perts_to_plot:
    plot_perturbation(
        best_model, p, pool_size=300, save_file=os.path.join(output_dir, f"{p}.png")
    )

test_loader = pert_data.dataloader["test_loader"]
test_res = eval_perturb(test_loader, best_model, device)

# test_metrics, test_pert_res = compute_metrics(test_res)
test_metrics = compute_perturbation_metrics(
    test_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
)
print(test_metrics)

# save the dicts in json
with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
    json.dump(test_metrics, f)
# with open(f"{save_dir}/test_pert_res.json", "w") as f:
#     json.dump(test_pert_res, f)

deeper_res = deeper_analysis(pert_data.adata, test_res)
non_dropout_res = non_dropout_analysis(pert_data.adata, test_res)

metrics = ["pearson_delta", "pearson_delta_de"]
metrics_non_dropout = [
    "pearson_delta_top20_de_non_dropout",
    "pearson_top20_de_non_dropout",
]
subgroup_analysis = {}
for name in pert_data.subgroup["test_subgroup"].keys():
    subgroup_analysis[name] = {}
    for m in metrics:
        subgroup_analysis[name][m] = []

    for m in metrics_non_dropout:
        subgroup_analysis[name][m] = []

for name, pert_list in pert_data.subgroup["test_subgroup"].items():
    for pert in pert_list:
        for m in metrics:
            subgroup_analysis[name][m].append(deeper_res[pert][m])

        for m in metrics_non_dropout:
            subgroup_analysis[name][m].append(non_dropout_res[pert][m])

for name, result in subgroup_analysis.items():
    for m in result.keys():
        mean_value = np.mean(subgroup_analysis[name][m])
        logger.info("test_" + name + "_" + m + ": " + str(mean_value))
