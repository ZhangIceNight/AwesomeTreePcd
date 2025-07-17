from typing import Optional, Tuple
import math
import numpy as np
import scipy.sparse as sparse
import cvxpy as cp
from cvxpy.error import SolverError
from functools import partial
from sklearn.metrics import pairwise_distances
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from timm.models.layers import DropPath, trunc_normal_
# from logger import get_missing_parameters_message, get_unexpected_parameters_message

from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from .pointnet2_utils import PointNetFeaturePropagation

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Mamba Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Mamba block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        # drop path 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None, ):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)

        # first conv
        feature = self.first_conv(point_groups.transpose(2, 1))  # bs*g, 256, n
        
        # global feature
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # bs*g, 256, 1

        # feature expansion
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # bs*g, 512, n

        # second conv
        feature = self.second_conv(feature)  # bs*g, encoder_channel, n

        # global feature
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # bs*g, encoder_channel

        
        return feature_global.reshape(bs, g, self.encoder_channel)


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            init.constant_(self.bias.data, 0.1)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj.float(), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class HGCN_layer(nn.Module):
    def __init__(self, img_len, in_c):
        super(HGCN_layer, self).__init__()
        self.gc1 = GraphConvolution(in_c, in_c)
        self.bn1 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)

        self.gc2 = GraphConvolution(in_c, in_c)
        self.bn2 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)

        self.gc3 = GraphConvolution(in_c, in_c)
        self.bn3 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.Softplus()

    def forward(self, feature, H):
        gc1 = self.gc1(feature, H)
        gc1 = self.bn1(gc1)
        gc1 = self.relu(feature + gc1)

        gc2 = self.gc2(gc1, H)
        gc2 = self.bn2(gc2)
        gc2 = self.relu(feature + gc2)

        gc3 = self.gc3(gc2, H)
        gc3 = self.bn3(gc3)
        gc3 = self.relu(feature + gc3)
        return gc3


class HGCNNet_deep(nn.Module):
    def __init__(self, img_len):
        super(HGCNNet, self).__init__()
        self.gc1 = GraphConvolution(384, 384)
        self.bn1 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.HGCN_layer1 = HGCN_layer(img_len, 384)

        self.gc2 = GraphConvolution(384, 384)
        self.bn2 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.HGCN_layer2 = HGCN_layer(img_len, 384)

        self.gc3 = GraphConvolution(384, 384)
        self.bn3 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.HGCN_layer3 = HGCN_layer(img_len, 384)

        self.gc4 = GraphConvolution(384, 384)
        self.bn4 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.HGCN_layer4 = HGCN_layer(img_len, 384)

        self.gc5 = GraphConvolution(384, 384)
        self.relu = nn.Softplus()

    def forward(self, feature, H):
        gc1 = self.gc1(feature, H)
        gc1 = self.bn1(gc1)
        gc1 = self.relu(gc1)
        gc1 = self.HGCN_layer1(gc1, H)

        gc2 = self.gc2(gc1, H)
        gc2 = self.bn2(gc2)
        gc2 = self.relu(gc2)
        gc2 = self.HGCN_layer2(gc2, H)

        gc3 = self.gc3(gc2, H)
        gc3 = self.bn3(gc3)
        gc3 = self.relu(gc3)
        gc3 = self.HGCN_layer3(gc3, H)

        gc4 = self.gc4(gc3, H)
        gc4 = self.bn4(gc4)
        gc4 = self.relu(gc4)
        gc4 = self.HGCN_layer4(gc4, H)

        gc5 = self.gc5(gc4, H)
        gc5 = self.relu(gc5)
        return gc5


class HGCNNet(nn.Module):
    def __init__(self, img_len):
        super(HGCNNet, self).__init__()
        # 第一层
        self.gc1 = GraphConvolution(384, 384)
        self.bn1 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.HGCN_layer1 = HGCN_layer(img_len, 384)

        # 第二层
        self.gc2 = GraphConvolution(384, 384)
        self.bn2 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.HGCN_layer2 = HGCN_layer(img_len, 384)

        # 输出层
        self.gc3 = GraphConvolution(384, 384)
        self.relu = nn.Softplus()

    def forward(self, feature, H):
        # 第一层
        gc1 = self.gc1(feature, H)
        gc1 = self.bn1(gc1)
        gc1 = self.relu(gc1)
        gc1 = self.HGCN_layer1(gc1, H)

        # 第二层
        gc2 = self.gc2(gc1, H)
        gc2 = self.bn2(gc2)
        gc2 = self.relu(gc2)
        gc2 = self.HGCN_layer2(gc2, H)

        # 输出层
        gc3 = self.gc3(gc2, H)
        gc3 = self.relu(gc3)
        
        return gc3

class MixerModelForSegmentation(MixerModel):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_path: int = 0.1,
            fetch_idx: Tuple[int] = [3, 7, 11],
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MixerModel, self).__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.fetch_idx = fetch_idx

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        feature_list = []
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            if idx in self.fetch_idx:
                if not self.fused_add_norm:
                    residual_output = (hidden_states + residual) if residual is not None else hidden_states
                    hidden_states_output = self.norm_f(residual_output.to(dtype=self.norm_f.weight.dtype))
                else:
                    # Set prenorm=False here since we don't need the residual
                    fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                    hidden_states_output = fused_add_norm_fn(
                        hidden_states,
                        self.norm_f.weight,
                        self.norm_f.bias,
                        eps=self.norm_f.eps,
                        residual=residual,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32,
                    )
                feature_list.append(hidden_states_output)
        return feature_list


class get_model(nn.Module):
    def __init__(self, cls_dim):
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.cls_dim = cls_dim

        self.group_size = 32
        # 1024==64 2048==128 4096==256
        self.num_group = 128
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # Weight for hypergraph merging
        self.W = Parameter(torch.ones(self.num_group * 1))
        # define the encoder
        self.encoder_dims = 384
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.HGCN = HGCNNet(img_len=self.num_group)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.blocks = MixerModelForSegmentation(d_model=self.trans_dim,
                                                n_layer=self.depth,
                                                rms_norm=False,
                                                drop_path=0.2,
                                                fetch_idx=[3, 7, 11])

        self.drop_out = nn.Dropout(0)
        self.drop_path_rate = 0.1
        self.drop_path_block = DropPath(self.drop_path_rate) if self.drop_path_rate > 0. else nn.Identity()

        self.norm = nn.LayerNorm(self.trans_dim)

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(0.2))

        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3, mlp=[self.trans_dim * 4, 1024])
        
        # 更新卷积层的输入维度
        self.convs1 = nn.Conv1d(3328, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if incompatible.missing_keys:
                print('missing_keys')
                # print(get_missing_parameters_message(incompatible.missing_keys))
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                # print(get_unexpected_parameters_message(incompatible.unexpected_keys))
            print(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}')
        else:
            print(f'[Mamba] No ckpt is loaded, training from scratch!')

    def KNN(self, X, n_neighbors, is_prob=True):
        n_nodes = X.shape[0]
        n_edges = n_nodes

        # try:
        #     m_dist = pairwise_distances(X)
        # except:
        #     print(X.shape)
        #     print(X)
        #     print(filename)

        m_dist = pairwise_distances(X)
        # if np.all(m_dist == 0):
        #     print(filename)
        #     print(m_dist)
        #     exit(0)
        # top n_neighbors+1
        m_neighbors = np.argpartition(m_dist, kth=n_neighbors + 1, axis=1)
        m_neighbors_val = np.take_along_axis(m_dist, m_neighbors, axis=1)

        m_neighbors = m_neighbors[:, :n_neighbors + 1]
        m_neighbors_val = m_neighbors_val[:, :n_neighbors + 1]

        # check
        for i in range(n_nodes):
            if not np.any(m_neighbors[i, :] == i):
                m_neighbors[i, -1] = i
                m_neighbors_val[i, -1] = 0.

        node_idx = m_neighbors.reshape(-1)
        edge_idx = np.tile(np.arange(n_edges).reshape(-1, 1), (1, n_neighbors + 1)).reshape(-1)

        if not is_prob:
            values = np.ones(node_idx.shape[0])
        else:
            avg_dist = np.mean(m_dist)
            m_neighbors_val = m_neighbors_val.reshape(-1)
            values = np.exp(-np.power(m_neighbors_val, 2.) / np.power(avg_dist, 2.))

        knn = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges)).toarray()
        return knn

    def similarity(self, X, n_neighbors):
        n_nodes = X.shape[0]
        n_edges = n_nodes
        sim = np.zeros((n_nodes, n_edges))

        for i in range(n_nodes):
            dist = []
            for j in range(n_edges):
                s = X[i].dot(X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]))
                dist.append(s)
            m_neighbors = sorted(dist, reverse=True)[0:n_neighbors + 1]
            for n in m_neighbors:
                ind = dist.index(n)
                sim[i][ind] = 1.0
        return sim

    def hyperG(self, knn, l1, sim, W):
        H = np.concatenate((knn, l1, sim), axis=1)
        # H = knn
        # the degree of the node
        DV = np.sum(H, axis=1)
        # the degree of the hyperedge
        DE = np.sum(H, axis=0)
        invDE = np.mat(np.diag(np.power(DE, -1)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))

        HT = H.T
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        DV2_H = torch.as_tensor(DV2_H).cuda().float()
        invDE_HT_DV2 = torch.as_tensor(invDE_HT_DV2).cuda().float()

        w = torch.diag(W)
        G = torch.mm(w, invDE_HT_DV2)
        G = torch.mm(DV2_H, G)
        return G
    def abhyperG(self, hyp, W):
        H = hyp
        # H = knn
        # the degree of the node
        DV = np.sum(H, axis=1)
        # the degree of the hyperedge
        DE = np.sum(H, axis=0)
        invDE = np.mat(np.diag(np.power(DE, -1)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))

        HT = H.T
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        DV2_H = torch.as_tensor(DV2_H).cuda().float()
        invDE_HT_DV2 = torch.as_tensor(invDE_HT_DV2).cuda().float()

        w = torch.diag(W)
        G = torch.mm(w, invDE_HT_DV2)
        G = torch.mm(DV2_H, G)
        return G
    def l1_representation(self, X, n_neighbors, gamma=1):
        n_nodes = X.shape[0]
        n_edges = n_nodes
        m_dist = pairwise_distances(X)
        m_neighbors = np.argsort(m_dist)[:, 0:n_neighbors + 1]

        edge_idx = np.tile(np.arange(n_edges).reshape(-1, 1), (1, n_neighbors + 1)).reshape(-1)
        node_idx = []
        values = []

        for i_edge in range(n_edges):
            neighbors = m_neighbors[i_edge].tolist()
            if i_edge in neighbors:
                neighbors.remove(i_edge)
            else:
                neighbors = neighbors[:-1]

            P = X[neighbors, :]  # k neighbor
            v = X[i_edge, :]  # ceneroid sample

            # cvxpy
            x = cp.Variable(P.shape[0], nonneg=True)
            objective = cp.Minimize(cp.norm((P.T @ x).T - v, 2) + gamma * cp.norm(x, 1))
            prob = cp.Problem(objective)
            try:
                prob.solve()
            except SolverError:
                prob.solve(solver='SCS', verbose=False)

            node_idx.extend([i_edge] + neighbors)
            values.extend([1.] + x.value.tolist())

        node_idx = np.array(node_idx)
        values = np.array(values)
        l1 = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges)).toarray()
        return l1

    def forward(self, pts):
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2).contiguous() # [B, N, 3]
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)    # neighborhood: [B, G, M, 3] (G: num_group=128, M: group_size=32)
                                                        # center: [B, G, 3]
        group_input_tokens = self.encoder(neighborhood)  # [B, G, encoder_dims=384]

        pos = self.pos_embed(center) # [B, G, trans_dim=384]

        # hypergraph serailization
        # X = group_input_tokens.cpu().detach().numpy()
        # H = []
        # n_neighbors = 2
        # for j in range(B):
        #     # knn = self.KNN(X[j, :, :], n_neighbors)
        #     l1 = self.l1_representation(X[j, :, :], n_neighbors)
        #     # sim = self.similarity(X[j, :, :], n_neighbors)

        #     # G = self.hyperG(knn, l1, sim, self.W)
        #     G = self.abhyperG(l1, self.W)
        #     H.append(torch.as_tensor(G).unsqueeze(0))

        # H = torch.cat(H, dim=0) # [B, 3G, 3G]

        # group_input_tokens = self.HGCN(group_input_tokens, H) # [B, G, 384]


        # final input
        x = group_input_tokens # [B, G, 384]

        feature_list = self.blocks(x, pos) # List of 3 tensors, each [B, G, 384]

        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list] # List of 3 tensors, each [B, 384, G]
        x = torch.cat((feature_list), dim=1)  # 1152 # [B, 1152, G]  (384*3 = 1152)
        x_max = torch.max(x, 2)[0] # [B, 1152]
        x_avg = torch.mean(x, 2) # [B, 1152]
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N) # [B, 1152, N]
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N) # [B, 1152, N]
        # cls_label_one_hot = cls_label.view(B, 16, 1)
        # cls_label_featue = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        # x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1)
        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1)

        f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x) # [B, 3328, N]

        x = torch.cat((f_level_0, x_global_feature), 1)  # [B, 3328, N]
        x = self.relu(self.bns1(self.convs1(x)))  # [B, 512, N]
        x = self.dp1(x)  # [B, 512, N]
        x = self.relu(self.bns2(self.convs2(x)))  # [B, 256, N]
        x = self.convs3(x)  # [B, cls_dim, N]
        x = F.log_softmax(x, dim=1)  # [B, cls_dim, N]
        x = x.permute(0, 2, 1)  # [B, N, cls_dim]
        return x


# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()
#         self.num_classes = 2
#         self.epsilon = 0.1

#     def forward(self, pred, target):
#         # total_loss = F.nll_loss(pred, target)
#         #torch.Size([65536, 2]) torch.Size([65536]) torch.Size([65536, 2])
#         #print('\033[31m' + str(pred.shape), str(target.shape), str(label_smooth(target).shape) + '\033[0m')
#         log_probs = F.log_softmax(pred, dim=-1)
#         target = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1).to(torch.int64), 1)
#         target = (1 - self.epsilon) * target + self.epsilon / self.num_classes
#         loss = (-target * log_probs).mean(0).sum()
#         return loss
    
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss
