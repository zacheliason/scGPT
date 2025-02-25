import contextlib
import math
from typing import Any, Mapping, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Bernoulli
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import SGConv
from tqdm import trange

from .. import logger
from ..gears_utils import (
    GeneSimNetwork,
    get_similarity_network,
)
from ..utils import map_raw_id_to_vocab_id
from .model import (
    ContinuousValueEncoder,
    ExprDecoder,
    FastTransformerEncoderWrapper,
    FlashTransformerEncoderLayer,
    MVCDecoder,
)


class MLP(torch.nn.Module):
    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        Multi-layer perceptron
        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer

        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1
                else None,
                torch.nn.ReLU(),
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.network(x)


class TransformerGenerator(nn.Module):
    def model_initialize(
        self,
        hidden_size=64,
        num_go_gnn_layers=1,
        num_gene_gnn_layers=1,
        decoder_hidden_size=16,
        num_similar_genes_go_graph=20,
        num_similar_genes_co_express_graph=20,
        coexpress_threshold=0.4,
        uncertainty=False,
        uncertainty_reg=1,
        direction_lambda=1e-1,
        G_go=None,
        G_go_weight=None,
        G_coexpress=None,
        G_coexpress_weight=None,
        no_perturb=False,
        **kwargs,
    ):
        """
        Initialize the model

        Parameters
        ----------
        hidden_size: int
            hidden dimension, default 64
        num_go_gnn_layers: int
            number of GNN layers for GO graph, default 1
        num_gene_gnn_layers: int
            number of GNN layers for co-expression gene graph, default 1
        decoder_hidden_size: int
            hidden dimension for gene-specific decoder, default 16
        num_similar_genes_go_graph: int
            number of maximum similar K genes in the GO graph, default 20
        num_similar_genes_co_express_graph: int
            number of maximum similar K genes in the co expression graph, default 20
        coexpress_threshold: float
            pearson correlation threshold when constructing coexpression graph, default 0.4
        uncertainty: bool
            whether or not to turn on uncertainty mode, default False
        uncertainty_reg: float
            regularization term to balance uncertainty loss and prediction loss, default 1
        direction_lambda: float
            regularization term to balance direction loss and prediction loss, default 1
        G_go: scipy.sparse.csr_matrix
            GO graph, default None
        G_go_weight: scipy.sparse.csr_matrix
            GO graph edge weights, default None
        G_coexpress: scipy.sparse.csr_matrix
            co-expression graph, default None
        G_coexpress_weight: scipy.sparse.csr_matrix
            co-expression graph edge weights, default None
        no_perturb: bool
            predict no perturbation condition, default False

        Returns
        -------
        None
        """

        self.config = {
            "hidden_size": hidden_size,
            "num_go_gnn_layers": num_go_gnn_layers,
            "num_gene_gnn_layers": num_gene_gnn_layers,
            "decoder_hidden_size": decoder_hidden_size,
            "num_similar_genes_go_graph": num_similar_genes_go_graph,
            "num_similar_genes_co_express_graph": num_similar_genes_co_express_graph,
            "coexpress_threshold": coexpress_threshold,
            "uncertainty": uncertainty,
            "uncertainty_reg": uncertainty_reg,
            "direction_lambda": direction_lambda,
            "G_go": G_go,
            "G_go_weight": G_go_weight,
            "G_coexpress": G_coexpress,
            "G_coexpress_weight": G_coexpress_weight,
            "device": self.device,
            "num_genes": self.num_genes,
            "num_perts": self.num_perts,
            "no_perturb": no_perturb,
        }

        # if self.wandb:
        #     self.wandb.config.update(self.config)

        if self.config["G_coexpress"] is None:
            ## calculating co expression similarity graph
            edge_list = get_similarity_network(
                network_type="co-express",
                adata=self.adata,
                threshold=coexpress_threshold,
                k=num_similar_genes_co_express_graph,
                data_path=self.data_path,
                data_name=self.dataset_name,
                split=self.split,
                seed=self.seed,
                train_gene_set_size=self.train_gene_set_size,
                set2conditions=self.set2conditions,
            )

            sim_network = GeneSimNetwork(
                edge_list, self.gene_list, node_map=self.node_map
            )
            self.G_coexpress = sim_network.edge_index
            self.G_coexpress_weight = sim_network.edge_weight

        if self.config["G_go"] is None:
            ## calculating gene ontology similarity graph
            edge_list = get_similarity_network(
                network_type="go",
                adata=self.adata,
                threshold=coexpress_threshold,
                k=num_similar_genes_go_graph,
                pert_list=self.pert_list,
                data_path=self.data_path,
                data_name=self.dataset_name,
                split=self.split,
                seed=self.seed,
                train_gene_set_size=self.train_gene_set_size,
                set2conditions=self.set2conditions,
                default_pert_graph=self.default_pert_graph,
            )

            sim_network = GeneSimNetwork(
                edge_list, self.pert_list, node_map=self.node_map_pert
            )
            self.G_go = sim_network.edge_index
            self.G_go_weight = sim_network.edge_weight

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int,
        n_cls: int,
        vocab: Any,
        pert_data: Any,
        seq_len: int = 1536,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        pert_pad_id: int = 2,
        do_mvc: bool = False,
        domain_spec_batchnorm: Union[bool, str] = False,
        n_input_bins: Optional[int] = 0,
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        decoder_activation: Optional[str] = None,
        decoder_adaptive_bias: bool = False,
        ecs_threshold: float = 0.3,
        explicit_zero_prob: bool = False,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
        # GEARS stuff
        device: Optional[torch.device] = None,
        hidden_size: int = 64,
        num_go_gnn_layers: int = 1,
        num_gene_gnn_layers: int = 1,
        decoder_hidden_size: int = 16,
        num_similar_genes_go_graph: int = 20,
        num_similar_genes_co_express_graph: int = 20,
        include_zero_gene: str = "all",
        max_seq_len: int = 1536,
        gene_ids: Any = None,
        # G_go: Optional[torch.Tensor] = None,
        # G_go_weight: Optional[torch.Tensor] = None,
        # G_coexpress: Optional[torch.Tensor] = None,
        # G_coexpress_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.pad_token_id = vocab[pad_token]
        self.pad_value = pad_value
        self.pert_pad_id = pert_pad_id
        self.ecs_threshold = ecs_threshold
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.n_input_bins = n_input_bins
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.norm_scheme = "pre" if pre_norm else "post"
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")
        if use_fast_transformer:
            try:
                from flash_attn.flash_attention import FlashMHA
            except ImportError:
                import warnings

                warnings.warn(
                    "flash-attn is not installed, using pytorch transformer instead. "
                    "Set use_fast_transformer=False to avoid this warning. "
                    "Installing flash-attn is highly recommended."
                )
                use_fast_transformer = False
        self.use_fast_transformer = use_fast_transformer

        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        self.pert_encoder = nn.Embedding(3, d_model, padding_idx=pert_pad_id)

        # print("Using simple batchnorm instead of domain specific batchnorm")
        # self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        if use_fast_transformer:
            if fast_transformer_backend == "linear":
                self.transformer_encoder = FastTransformerEncoderWrapper(
                    d_model, nhead, d_hid, nlayers, dropout
                )
            elif fast_transformer_backend == "flash":
                encoder_layers = FlashTransformerEncoderLayer(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                )
                self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        else:
            encoder_layers = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout, batch_first=True
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # self.decoder = nn.Linear(d_model, 1)
        self.decoder = AffineExprDecoder(
            d_model,
            explicit_zero_prob=explicit_zero_prob,
            activation=decoder_activation,
            adaptive_bias=decoder_adaptive_bias,
        )
        self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)
        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
            )

        self.dataloader = pert_data.dataloader
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.default_pert_graph = pert_data.default_pert_graph
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model_initialize()

        hidden_size = self.config["hidden_size"]
        self.gene_ids = gene_ids

        self.index_map = self._create_pert_index_map(
            pert_data.pert_names.tolist(), pert_data.gene_names.values.tolist()
        )

        self.max_seq_len = max_seq_len
        self.include_zero_gene = include_zero_gene

        ### perturbation gene ontology GNN
        self.G_sim = self.G_go.to(device)
        self.G_sim_weight = self.G_go_weight.to(device)

        self.sim_layers = torch.nn.ModuleList()

        self.num_layers = num_go_gnn_layers
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))

        # self.pert_fuse = MLP(
        #     [hidden_size, hidden_size, hidden_size], last_layer_act="ReLU"
        # )

        self.seq_length = seq_len
        self.d_model = d_model
        self.pert_fuse = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.d_model),
        )

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        pert_idx,
        src_key_padding_mask: Tensor,
        seq_len: int,
    ) -> Tensor:
        src = self.encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src
        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        # perts = self.pert_encoder(input_pert_flags)  # (batch, seq_len, embsize)
        pert_embedding = self.perturb_encode(pert_idx=pert_idx, seq_len=seq_len)

        if pert_embedding.shape != values.shape:
            raise ValueError(
                f"Shape of pert_embedding {pert_embedding.shape} does not match the shape of values {values.shape}."
            )

        # print(f"Pert Embedding Shape: {pert_embedding.shape}")
        # print(f"Source Shape: {src.shape}")
        # print(f"Values Shape: {values.shape}")

        total_embs = src + values + pert_embedding

        # total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output  # (batch, seq_len, embsize)

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

    def perturb_encode(self, pert_idx, seq_len):
        pert_index = []
        for idx, i in enumerate(pert_idx):
            for j in i:
                if j != -1:
                    pert_index.append([idx, j])
        pert_index = torch.tensor(pert_index).T

        pert_global_emb = self.pert_emb(
            torch.LongTensor(list(range(self.num_perts))).to(self.device)
        )

        ## augment global perturbation embedding with GNN
        for idx, layer in enumerate(self.sim_layers):
            pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
            if idx < self.num_layers - 1:
                pert_global_emb = pert_global_emb.relu()

        if pert_index.shape[0] != 0:
            ### in case all samples in the batch are controls, then there is no indexing for pert_index.
            pert_track = {}
            for i, j in enumerate(pert_index[0]):
                if j.item() in pert_track:
                    pert_track[j.item()] = (
                        pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    )
                else:
                    pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

            if len(list(pert_track.values())) > 0:
                # sample_row = list(pert_track.values())[0]

                # for i in range(max(pert_track.keys()) + 1):
                #     if i not in pert_track:
                #         pert_track[i] = torch.zeros_like(sample_row)

                if len(list(pert_track.values())) == 1:
                    # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                    emb_total = torch.stack(list(pert_track.values()) * 2)
                    emb_total = emb_total.unsqueeze(1)
                    emb_total = emb_total.expand(-1, seq_len, -1)
                    emb_total = self.pert_fuse(emb_total)
                else:
                    emb_total = torch.stack(list(pert_track.values()))
                    emb_total = emb_total.unsqueeze(1)
                    emb_total = emb_total.expand(-1, seq_len, -1)
                    emb_total = self.pert_fuse(emb_total)

                # emb_total = emb_total.view(-1, seq_len, self.d_model)

        pert_track_index_to_emb_index = {k: i for i, k in enumerate(pert_track.keys())}

        batch_size = len(pert_idx)
        new_tensor = torch.zeros(
            (batch_size, seq_len, self.d_model), device=emb_total.device
        )

        for batch_idx in range(batch_size):
            if batch_idx in pert_track:
                new_tensor[batch_idx] = emb_total[
                    pert_track_index_to_emb_index[batch_idx]
                ]

        # print(f"seq_len: {seq_len}, new_tensor: {new_tensor.shape}")
        return new_tensor

    def _create_pert_index_map(self, pert_names, gene_names):
        """
        Creates a mapping from indices in pert_names to matching indices in gene_names.
        Caches results to avoid recomputation.

        Parameters:
        pert_names (np.ndarray): Array of gene names to map from
        gene_names (list): List of gene names to map to
        cache_file (str): Path to save/load the index mapping

        Returns:
        dict: Mapping of indices {pert_index: gene_names_index}
        """

        # Create new mapping
        index_map = {}

        # Convert gene_names to dictionary for O(1) lookup
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}

        # Create mapping
        for idx, pert in enumerate(pert_names):
            if pert in gene_to_idx:
                # Store the mapping using string keys (JSON requirement)
                index_map[str(idx)] = str(gene_to_idx[pert])

        return index_map

    def process_batch(self, batch_data, sample=True, predict=False):
        # if "pert_idx" not in batch_data:
        #     pert_idx = batcH_data.edge_stores["pert_idx"]

        x, pert_idx = batch_data.x, batch_data.pert_idx
        # pert = batch_data.pert

        actual_batch_size = batch_data.batch_size
        # if predict:
        if x.size(0) % self.num_genes == 0:
            ori_gene_values = x[:, 0].view(actual_batch_size, self.num_genes)
        else:
            ori_gene_values = x[:, 0].view(actual_batch_size, 1)

        pert_flags = torch.zeros_like(ori_gene_values)

        if len(pert_idx) % actual_batch_size == 0:
            pert_idx = pert_idx.view(actual_batch_size, -1)
        else:
            print("PERT ERROR")

        for batch_idx, ps in enumerate(pert_idx):
            for p in ps:
                gene_idx = int(p)
                pert_flags[batch_idx, gene_idx] = torch.tensor(
                    1, dtype=torch.int64, device=pert_flags.device
                )

            # if str(p) in self.index_map:
            #     gene_idx = int(self.index_map[str(p)])
            #     pert_flags[batch_idx, gene_idx] = torch.tensor(
            #         1, dtype=torch.int64, device=pert_flags.device
            #     )

            # if str(p) not in self.index_map and p != -1:
            #     print(f"Missing perturbation {p} in index_map")

        # pert_flags = x[:, 1].long().view(actual_batch_size, n_genes)
        target_gene_values = batch_data.y  # (batch_size, n_genes)

        if self.include_zero_gene in ["all", "batch-wise"]:
            if self.include_zero_gene == "all":
                input_gene_ids = torch.arange(
                    self.num_genes, device=self.device, dtype=torch.long
                )
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            # # sample input_gene_id
            # if len(input_gene_ids) > max_seq_len:
            #     input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
            #         :max_seq_len
            #     ]

            if sample:
                # Get indices of perturbed genes (where pert_flags != 0)
                perturbed_gene_mask = (pert_flags != 0).any(dim=0)  # Shape: (n_genes,)
                perturbed_gene_ids = torch.where(perturbed_gene_mask)[0]

                # Get indices of non-perturbed genes
                non_perturbed_gene_ids = torch.where(~perturbed_gene_mask)[0]

                # Prioritize perturbed genes in truncation
                if len(perturbed_gene_ids) >= self.max_seq_len:
                    # Case 1: More perturbed genes than `max_seq_len` → truncate perturbed genes
                    input_gene_ids = perturbed_gene_ids[: self.max_seq_len]
                else:
                    # Case 2: Include all perturbed genes + sample non-perturbed genes
                    num_non_perturbed = self.max_seq_len - len(perturbed_gene_ids)
                    non_perturbed_sampled = non_perturbed_gene_ids[
                        torch.randperm(len(non_perturbed_gene_ids))[:num_non_perturbed]
                    ]
                    input_gene_ids = torch.cat(
                        [perturbed_gene_ids, non_perturbed_sampled]
                    )

            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids].to(
                device=self.device, dtype=torch.long
            )
            target_values = (
                target_gene_values[:, input_gene_ids] if not predict else None
            )

            src = map_raw_id_to_vocab_id(input_gene_ids, self.gene_ids)
            src = src.repeat(actual_batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=self.device
            )

            return (
                src,
                input_values,
                input_pert_flags,
                target_values,
                src_key_padding_mask,
                input_gene_ids,
                pert_idx,
            )

    def forward(
        self,
        batch_data,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False,
        sample_batch: bool = True,
        predict: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.

        Returns:
            dict of output Tensors.
        """

        (
            src,
            input_values,
            input_pert_flags,
            target_values,
            src_key_padding_mask,
            input_gene_ids,
            pert_idx,
        ) = self.process_batch(batch_data, sample=sample_batch, predict=predict)

        if self.explicit_zero_prob and not do_sample and not self.training:
            do_sample = True
            logger.warning("Auto set do_sample to True when model is in eval mode.")

        # binning input gene values
        if self.n_input_bins > 0:
            from ..preprocess import binning

            processed_values = torch.stack(
                [binning(row, n_bins=self.n_input_bins) for row in input_values], dim=0
            ).to(input_values.device)
        else:
            processed_values = input_values

        seq_len = processed_values.size(1)

        transformer_output = self._encode(
            src, processed_values, pert_idx, src_key_padding_mask, seq_len=seq_len
        )
        output = {}
        mlm_output = self.decoder(transformer_output, input_values)
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        cell_emb = self._get_cell_emb_from_layer(transformer_output, input_values)
        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        if MVC:
            mvc_output = self.mvc_decoder(
                cell_emb,
                self.cur_gene_token_embs,
            )  # (batch, seq_len)
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        if ECS:
            # Here using customized cosine similarity instead of F.cosine_similarity
            # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
            # normalize the embedding
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)

            # mask out diagnal elements
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            # only optimize positive similarities
            cos_sim = F.relu(cos_sim)

            output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        output["target_values"] = target_values
        output["input_gene_ids"] = input_gene_ids

        return output

    def encode_batch(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        output_to_cpu: bool = True,
    ) -> Tensor:
        """
        Args:
            src: Tensor, shape [N, seq_len]
            values: Tensor, shape [N, seq_len]
            src_key_padding_mask: Tensor, shape [N, seq_len]

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        outputs = []
        N = src.size(0)
        device = next(self.parameters()).device
        for i in trange(0, N, batch_size):
            output = self._encode(
                src[i : i + batch_size].to(device),
                values[i : i + batch_size].to(device),
                src_key_padding_mask[i : i + batch_size].to(device),
            )
            if output_to_cpu:
                output = output.cpu()
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    def pred_perturb(
        self,
        batch_data,
        include_zero_gene="batch-wise",
        # gene_ids=None,
        amp=True,
        predict=False,
    ) -> Tensor:
        """
        Args:
            batch_data: a dictionary of input data with keys.

        Returns:
            output Tensor of shape [N, seq_len]
        """
        self.eval()
        device = next(self.parameters()).device
        batch_data.to(device)

        if self.include_zero_gene in ["all", "batch-wise"]:
            context_manager = (
                torch.cuda.amp.autocast(enabled=amp)
                if torch.cuda.is_available()
                else contextlib.nullcontext()
            )

            with context_manager:
                output_dict = self(
                    batch_data,
                    CLS=False,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=True,
                    sample_batch=False,
                    predict=predict,
                )
            output_values = output_dict["mlm_output"].float()
            input_gene_ids = output_dict["input_gene_ids"]
            pred_gene_values = torch.zeros_like(
                batch_data.x[:, 0].view(len(batch_data.pert), -1)
            )
            pred_gene_values[:, input_gene_ids] = output_values
        return pred_gene_values


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class AffineExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        activation: Optional[str] = None,
        tanh_coeff: bool = False,
        adaptive_bias: bool = False,
    ):
        """
        Predict the expression value of each gene in an affine like form of Ax + b.
        This decoder takes two ExprDecoder intrinsically to genrate the coefficient A and bias b.

        Args:
            d_model: The embedding dimension.
            explicit_zero_prob: If True, predict the probability of each gene being
                zero.
            activation: The activation function for the coefficient A and bias b.
            tanh_coeff: If True, use tanh activation for the coefficient A.
            adaptive_bias: If True, use a learnable bias for the bias b.
        """
        super().__init__()
        self.explicit_zero_prob = explicit_zero_prob
        self.tanh_coeff = tanh_coeff
        self.adaptive_bias = adaptive_bias
        self.coeff_decoder = ExprDecoder(d_model, explicit_zero_prob=explicit_zero_prob)
        self.bias_decoder = ExprDecoder(d_model, explicit_zero_prob=explicit_zero_prob)

        self.activation = activation
        if activation is not None:
            assert hasattr(nn, activation), f"Unknown activation: {activation}"
            self.activation = getattr(nn, activation)()

    def forward(self, x: Tensor, values: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embsize]
            values: Tensor, shape [batch_size, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len]
        """
        coeff = self.coeff_decoder(x)
        bias = self.bias_decoder(x)

        if self.activation is not None:
            coeff["pred"] = self.activation(coeff["pred"])
            bias["pred"] = self.activation(bias["pred"])

        # if self.tanh_coeff:
        #     coeff["pred"] = 1 + torch.tanh(coeff["pred"])

        if self.adaptive_bias:
            # bias["pred"] = bias["pred"] * values.mean(dim=1, keepdim=True)
            non_zero_value_mean = values.sum(dim=1, keepdim=True) / (values != 0).sum(
                dim=1, keepdim=True
            )
            bias["pred"] = bias["pred"] * non_zero_value_mean

        if self.explicit_zero_prob:
            return {
                "pred": coeff["pred"] * values + bias["pred"],
                "zero_probs": coeff["zero_probs"],
            }

        return dict(pred=coeff["pred"] * values + bias["pred"])


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        zero_out_idx: Optional[int] = None,
    ):
        """
        Generic token embedding module.

        Args:
            num_embeddings: The number of tokens.
            embedding_dim: The embedding dimension.
            padding_idx: The index of the padding token.
            zero_out_idx: Indicate if any idx embedding should be zero vector.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

        self.zero_out_idx = zero_out_idx
        if zero_out_idx is not None:
            self._fill_idx_with_zero(zero_out_idx)
            zero_vector = self(zero_out_idx)
            assert torch.all(zero_vector == 0.0)
            assert not zero_vector.requires_grad

    def _fill_idx_with_zero(self, idx) -> None:
        with torch.no_grad():
            self.embedding.weight[idx].fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)
