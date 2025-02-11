import gc
import json
import os
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import psutil
import torch

# Third-party library imports
from gears import PertData
from torchtext._torchtext import VocabPybind
from torchtext.vocab import Vocab

# Local imports
import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import add_file_handler, set_seed


def configure_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.cuda.set_per_process_memory_fraction(0.95)
    matplotlib.rcParams["savefig.transparent"] = False
    warnings.filterwarnings("ignore")


def setup_logging(data_name):
    # Create save directory
    timestamp = time.strftime("%b%d-%H-%M")
    save_dir = Path(f"./save/dev_perturb_{data_name}-{timestamp}/")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = scg.logger
    add_file_handler(logger, save_dir / "run.log")
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return logger, save_dir


def print_memory_usage():
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024 / 1024  # in MB
    gpu_usage = (
        torch.cuda.memory_allocated() / 1024 / 1024
        if torch.cuda.is_available()
        else None
    )

    print(f"RAM Usage: {ram_usage:.2f} MB")
    if gpu_usage is not None:
        print(f"GPU Memory Usage: {gpu_usage:.2f} MB")


def clear_pert_data_memory(pert_data):
    essential_data = {
        "adata": pert_data.adata if hasattr(pert_data, "adata") else None,
        "gene_names": pert_data.gene_names
        if hasattr(pert_data, "gene_names")
        else None,
    }

    attributes_to_clear = [
        "processed_data",
        "raw_data",
        "train_control_data",
        "val_control_data",
        "test_control_data",
    ]

    for attr in attributes_to_clear:
        if hasattr(pert_data, attr):
            delattr(pert_data, attr)

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nMemory usage after clearing:")
    print_memory_usage()

    return pert_data


def prepare_data(data_name, split, batch_size, eval_batch_size):
    print_memory_usage()

    data_dir = "data"
    pert_data = PertData(data_dir)
    pert_data.load(data_name=data_name)
    pert_data.prepare_split(split=split, seed=1)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

    print("\nAfter loading:")
    print_memory_usage()

    pert_data = clear_pert_data_memory(pert_data)
    print("\nAfter clearing:")
    print_memory_usage()

    return pert_data


def setup_model(
    pert_data,
    load_model,
    special_tokens,
    pad_token,
    pad_value,
    pert_pad_id,
    use_fast_transformer,
):
    # Prepare vocabulary
    if load_model:
        model_dir = Path(load_model)
        vocab_file = model_dir / "vocab.json"
        model_file = model_dir / "best_model.pt"
        model_config_file = model_dir / "args.json"

        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        # Load model configurations
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)

        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]
    else:
        genes = pert_data.adata.var["gene_name"].tolist()
        vocab = Vocab(VocabPybind(genes + special_tokens, None))

    vocab.set_default_index(vocab[pad_token])
    genes = pert_data.adata.var["gene_name"].tolist()
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab[pad_token] for gene in genes],
        dtype=int,
    )
    n_genes = len(genes)

    # Create model
    ntokens = len(vocab)
    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=0,
        pad_token=pad_token,
        pad_value=pad_value,
        pert_pad_id=pert_pad_id,
        use_fast_transformer=use_fast_transformer,
    )

    # Load model weights
    if load_model:
        pretrained_dict = torch.load(model_file)
        pretrained_dict = _convert_wqkv_to_in_proj(pretrained_dict)

        load_param_prefixs = [
            "encoder",
            "value_encoder",
            "transformer_encoder",
        ]

        if load_param_prefixs:
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if any([k.startswith(prefix) for prefix in load_param_prefixs])
            }

        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, vocab, gene_ids, n_genes


def _convert_wqkv_to_in_proj(pretrained_state_dict):
    converted_state_dict = {}
    for key, value in pretrained_state_dict.items():
        if "self_attn.in_proj_weight" in key or "self_attn.Wqkv.weight" in key:
            converted_state_dict[
                key.replace("self_attn.Wqkv.weight", "self_attn.in_proj_weight")
            ] = value
        elif "self_attn.in_proj_bias" in key or "self_attn.Wqkv.bias" in key:
            converted_state_dict[
                key.replace("self_attn.Wqkv.bias", "self_attn.in_proj_bias")
            ] = value
        else:
            converted_state_dict[key] = value
    return converted_state_dict


def train(
    model,
    train_loader,
    trainer_utils,
    config,
    log_interval,
):
    """
    Train the model for one epoch with gradient accumulation.

    Parameters:
    - model: The model to train.
    - train_loader: DataLoader for training data.
    - trainer_utils: Dictionary containing utilities (optimizer, scaler, scheduler, device).
    - config: Dictionary containing model-specific configurations (e.g., n_genes, max_seq_len).
    - log_interval: Interval at which to log training metrics.
    """
    model.train()
    optimizer = trainer_utils["optimizer"]
    scaler = trainer_utils["scaler"]
    scheduler = trainer_utils["scheduler"]
    device = trainer_utils["device"]

    n_genes = config["n_genes"]
    max_seq_len = config["max_seq_len"]
    include_zero_gene = config["include_zero_gene"]
    criterion = trainer_utils["criterion"]

    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    optimizer.zero_grad()  # Zero gradients at the start

    for batch, batch_data in enumerate(train_loader):
        actual_batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(actual_batch_size, n_genes)
        pert_flags = x[:, 1].long().view(actual_batch_size, n_genes)
        target_gene_values = batch_data.y  # (batch_size, n_genes)

        # Handle gene IDs and inputs
        (
            input_gene_ids,
            input_values,
            input_pert_flags,
            target_values,
            src_key_padding_mask,
        ) = prepare_inputs(
            ori_gene_values,
            pert_flags,
            target_gene_values,
            include_zero_gene,
            max_seq_len,
            n_genes,
            device,
        )

        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=trainer_utils["amp"]):
            output_dict = model(
                input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                **config["model_params"],
            )
            output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(input_values, dtype=torch.bool)
            loss = loss_mse = criterion(output_values, target_values, masked_positions)
            loss = loss / config["accumulation_steps"]

        # Backward pass and optimization
        scaler.scale(loss).backward()
        if (batch + 1) % config["accumulation_steps"] == 0 or (
            batch + 1
        ) == num_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Logging
        total_loss += loss.item() * config["accumulation_steps"]
        total_mse += loss_mse.item() * config["accumulation_steps"]

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            trainer_utils["logger"].info(
                f"| epoch {config['epoch']:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()


def prepare_inputs(
    ori_gene_values,
    pert_flags,
    target_gene_values,
    include_zero_gene,
    max_seq_len,
    n_genes,
    device,
):
    if include_zero_gene == "all":
        input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
    else:
        input_gene_ids = ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]

    if len(input_gene_ids) > max_seq_len:
        input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
            :max_seq_len
        ]

    input_values = ori_gene_values[:, input_gene_ids]
    input_pert_flags = pert_flags[:, input_gene_ids]
    target_values = target_gene_values[:, input_gene_ids]

    src_key_padding_mask = torch.zeros_like(
        input_values, dtype=torch.bool, device=device
    )

    return (
        input_gene_ids,
        input_values,
        input_pert_flags,
        target_values,
        src_key_padding_mask,
    )


def main():
    configure_environment()
    set_seed(42)

    # Configuration settings
    data_name = "adamson"
    split = "simulation"
    batch_size = 32
    eval_batch_size = 32
    lr = 1e-4
    accumulation_steps = 2
    epochs = 15
    schedule_interval = 1
    early_stop = 10
    amp = True
    log_interval = 100
    load_model = "../save/scGPT_human"
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    pad_token = "<pad>"
    pad_value = 0
    pert_pad_id = 0
    use_fast_transformer = True

    # Setup logging
    logger, save_dir = setup_logging(data_name)

    # Prepare data
    pert_data = prepare_data(data_name, split, batch_size, eval_batch_size)

    # Setup model
    model, vocab, gene_ids, n_genes = setup_model(
        pert_data,
        load_model,
        special_tokens,
        pad_token,
        pad_value,
        pert_pad_id,
        use_fast_transformer,
    )

    # Train model
    train_loader = pert_data.get_dataloader(batch_size=batch_size)
    val_loader = pert_data.get_dataloader(batch_size=eval_batch_size)

        
def train(
    model,
    train_loader,
    trainer_utils,
    config,
    log_interval,
):

    trainer_utils = {
        "optimizer": optimizer,
        "scaler": scaler,
        "scheduler": scheduler,
        "device": device,
        "criterion": criterion,
    }

    config = {
        "n_genes": n_genes,
        "max_seq_len": max_seq_len,
        "include_zero_gene": include_zero_gene,
        "epoch": 0,
        "accumulation_steps": 1,
    }

    train(model, train_loader, trainer_utils, config, log_interval)

    train(
        model,
        train_loader,
        val_loader,
        epochs,
        lr,
        accumulation_steps,
        schedule_interval,
        early_stop,
        amp,
        log_interval,
    )


if __name__ == "__main__":
    main()
