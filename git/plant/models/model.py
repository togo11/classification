from timm.models.maxxvit import MbConvBlock, ConvNeXtBlock, PartitionAttention2d, PartitionAttentionCl
from timm.models.maxxvit import Stem, _init_transformer, _init_conv
from dataclasses import dataclass, replace
from functools import partial
from typing import Optional, Union, Tuple, List
import torch.nn.functional as F

import torch
from torch import nn
from timm.models.helpers import build_model_with_cfg, checkpoint_seq, named_apply
from timm.models.layers import ConvMlp,  ClassifierHead, trunc_normal_tf_, LayerNorm
from timm.models.layers import get_norm_layer
from timm.models.layers import to_2tuple, extend_tuple, _assert
from timm.models.registry import register_model

__all__ = ['MaxVit']

def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.95, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'stem.conv1', 'classifier': 'head.fc',
        'fixed_input_size': True,
        **kwargs
    }


default_cfgs = {
    # Trying to be like the MaxViT paper configs
    'maxvit_tiny_rework_224': _cfg(url=''),
    'maxvit_tiny_rework_224_bg': _cfg(url=''),
    'maxvit_tiny_rework_224_gb': _cfg(url=''),
    'maxvit_tiny_rework_256_22': _cfg(
        url='',
        input_size=(3, 256, 256)),
    'maxvit_tiny_rework_256_28': _cfg(
        url='',
        input_size=(3, 256, 256)),
    'maxvit_tiny_rework_256_82': _cfg(
        url='',
        input_size=(3, 256, 256)),
    'maxvit_tiny_rework_256_88': _cfg(
        url='',
        input_size=(3, 256, 256)),
    'maxvit_base_rework_256': _cfg(
        url='',
        input_size=(3, 256, 256)),
    'maxvit_base_rework_256_2M': _cfg(
        url='',
        input_size=(3, 256, 256)),
    'maxvit_large_rework_256': _cfg(
        url='',
        input_size=(3, 256, 256))

}


@dataclass
class MaxxVitTransformerCfg:
    dim_head: int = 32
    expand_ratio: float = 4.0
    expand_first: bool = True   # rw false
    shortcut_bias: bool = True
    attn_bias: bool = True
    attn_drop: float = 0.
    proj_drop: float = 0.
    pool_type: str = 'avg2'
    rel_pos_type: str = 'bias'
    rel_pos_dim: int = 512  # for relative position types w/ MLP
    partition_ratio_b: int = 32
    partition_ratio_g: int = 32
    window_size: Optional[Tuple[int, int]] = None
    grid_size: Optional[Tuple[int, int]] = None
    init_values: Optional[float] = None
    act_layer: str = 'gelu'
    norm_layer: str = 'layernorm2d'
    norm_layer_cl: str = 'layernorm'
    norm_eps: float = 1e-6

    def __post_init__(self):
        if self.grid_size is not None:
            self.grid_size = to_2tuple(self.grid_size)
        if self.window_size is not None:
            self.window_size = to_2tuple(self.window_size)
            if self.grid_size is None:
                self.grid_size = self.window_size


@dataclass
class MaxxVitConvCfg:
    block_type: str = 'mbconv'
    expand_ratio: float = 4.0
    expand_output: bool = True  # calculate expansion channels from output (vs input chs), rw:false
    kernel_size: int = 3
    group_size: int = 1  # 1 == depthwise
    pre_norm_act: bool = False  # activation after pre-norm
    output_bias: bool = True  # bias for shortcut + final 1x1 projection conv
    stride_mode: str = 'dw'  # stride done via one of 'pool', '1x1', 'dw'
    pool_type: str = 'avg2'
    downsample_pool_type: str = 'avg2'
    attn_early: bool = False  # apply attn between conv2 and norm2, instead of after norm2
    attn_layer: str = 'se'
    attn_act_layer: str = 'silu'
    attn_ratio: float = 0.25    # rw:1/16
    init_values: Optional[float] = 1e-6  # for ConvNeXt block, ignored by MBConv
    act_layer: str = 'gelu'     # rw:'silu'
    norm_layer: str = ''
    norm_layer_cl: str = ''
    norm_eps: Optional[float] = None

    def __post_init__(self):
        # mbconv vs convnext blocks have different defaults, set in post_init to avoid explicit config args
        assert self.block_type in ('mbconv', 'convnext')
        use_mbconv = self.block_type == 'mbconv'
        if not self.norm_layer:
            self.norm_layer = 'batchnorm2d' if use_mbconv else 'layernorm2d'
        if not self.norm_layer_cl and not use_mbconv:
            self.norm_layer_cl = 'layernorm'
        if self.norm_eps is None:
            self.norm_eps = 1e-5 if use_mbconv else 1e-6
        self.downsample_pool_type = self.downsample_pool_type or self.pool_type


@dataclass
class MaxxVitCfg:
    embed_dim: Tuple[int, ...] = (96, 192, 384, 768)
    depths: Tuple[int, ...] = (2, 3, 5, 2)
    block_type: Tuple[Union[str, Tuple[str, ...]], ...] = ('M', 'M', 'M', 'M')
    stem_width: Union[int, Tuple[int, int]] = 64
    stem_bias: bool = True
    conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg()
    transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg()
    weight_init: str = 'vit_eff'


def _rw_max_cfg(
        stride_mode='dw',
        pool_type='avg2',
        conv_output_bias=False,
        conv_attn_ratio=1 / 16,
        conv_norm_layer='',
        transformer_norm_layer='layernorm2d',
        transformer_norm_layer_cl='layernorm',
        window_size=None,
        dim_head=32,
        init_values=None,
        rel_pos_type='bias',
        rel_pos_dim=512,
):
    # 'RW' timm variant models were created and trained before seeing https://github.com/google-research/maxvit
    # Differences of initial timm models:
    # - mbconv expansion calculated from input instead of output chs
    # - mbconv shortcut and final 1x1 conv did not have a bias
    # - mbconv uses silu in timm, not gelu
    # - expansion in attention block done via output proj, not input proj
    return dict(
        conv_cfg=MaxxVitConvCfg(
            stride_mode=stride_mode,
            pool_type=pool_type,
            expand_output=False,
            output_bias=conv_output_bias,
            attn_ratio=conv_attn_ratio,
            act_layer='silu',
            norm_layer=conv_norm_layer,
        ),
        transformer_cfg=MaxxVitTransformerCfg(
            expand_first=False,
            pool_type=pool_type,
            dim_head=dim_head,
            window_size=window_size,
            init_values=init_values,
            norm_layer=transformer_norm_layer,
            norm_layer_cl=transformer_norm_layer_cl,
            rel_pos_type=rel_pos_type,
            rel_pos_dim=rel_pos_dim,
        ),
    )


model_cfgs = dict(
    maxvit_tiny_rework_224=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=64,
    ),
    maxvit_tiny_rework_224_bg=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=64,
        transformer_cfg=replace(MaxxVitTransformerCfg(), partition_ratio_b=32, partition_ratio_g=224)
    ),
    maxvit_tiny_rework_224_gb=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=64,
        transformer_cfg=replace(MaxxVitTransformerCfg(), partition_ratio_b=224, partition_ratio_g=32)
    ),

    maxvit_tiny_rework_256_22=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=64,
        transformer_cfg=replace(MaxxVitTransformerCfg(), partition_ratio_b=128, partition_ratio_g=128)
    ),
    maxvit_tiny_rework_256_28=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=64,
        transformer_cfg=replace(MaxxVitTransformerCfg(), partition_ratio_b=128, partition_ratio_g=32)
    ),
    maxvit_tiny_rework_256_82=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=64,
        transformer_cfg=replace(MaxxVitTransformerCfg(), partition_ratio_b=32, partition_ratio_g=128)
    ),

    maxvit_tiny_rework_256_88=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=64,
        transformer_cfg=replace(MaxxVitTransformerCfg(), partition_ratio_b=32, partition_ratio_g=32)
    ),

    maxvit_base_rework_256=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 6, 14, 2),
        block_type=('M',) * 4,
        stem_width=64,
        transformer_cfg=replace(MaxxVitTransformerCfg(), partition_ratio_b=32, partition_ratio_g=32)
    ),
    maxvit_base_rework_256_2M=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(3, 4, 6, 3),
        block_type=('M',) * 4,
        stem_width=64,
        transformer_cfg=replace(MaxxVitTransformerCfg(), partition_ratio_b=32, partition_ratio_g=32)
    ),
    maxvit_large_rework_256=MaxxVitCfg(
        embed_dim=(192, 384, 768, 1536),
        depths=(2, 6, 14, 2),
        block_type=('M',) * 4,
        stem_width=64,
        transformer_cfg=replace(MaxxVitTransformerCfg(), partition_ratio_b=32, partition_ratio_g=32)
    ),
)


class MaxxVitBlock(nn.Module):
    """ MaxVit conv, window partition + FFN , grid partition + FFN
    """

    def __init__(
            self,
            dim: int,
            dim_out: int,
            stride: int = 1,
            conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            use_nchw_attn: bool = False,  # FIXME move to cfg? True is5 ~20-30% faster on TPU, 5-10% slower on GPU
            drop_path: float = 0.,
            type: str = 'block',
    ):
        super().__init__()

        conv_cls = ConvNeXtBlock if conv_cfg.block_type == 'convnext' else MbConvBlock
        self.conv = conv_cls(dim, dim_out, stride=stride, cfg=conv_cfg, drop_path=drop_path)

        attn_kwargs = dict(dim=dim_out, cfg=transformer_cfg, drop_path=drop_path)
        partition_layer = PartitionAttention2d if use_nchw_attn else PartitionAttentionCl
        self.nchw_attn = use_nchw_attn
        self.attn_block = partition_layer(**attn_kwargs)
        self.attn_grid = partition_layer(partition_type='grid', **attn_kwargs)
        self.type = type
    def init_weights(self, scheme=''):
        named_apply(partial(_init_transformer, scheme=scheme), self.attn_block)
        named_apply(partial(_init_transformer, scheme=scheme), self.attn_grid)
        named_apply(partial(_init_conv, scheme=scheme), self.conv)

    def forward(self, x):
        # NCHW format
        x = self.conv(x)

        if not self.nchw_attn:
            x = x.permute(0, 2, 3, 1)  # to NHWC (channels-last)

        if self.type == 'block':
            x = self.attn_block(x)
        elif self.type == 'grid':
            x = self.attn_grid(x)
        if not self.nchw_attn:
            x = x.permute(0, 3, 1, 2)  # back to NCHW
        return x


class MaxVitStage(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 2,
            depth: int = 4,
            feat_size: Tuple[int, int] = (14, 14),
            block_types: Union[str, Tuple[str]] = 'C',
            transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),#
            conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            drop_path: Union[float, List[float]] = 0.,
            type: str = 'block'
    ):
        super().__init__()
        self.grad_checkpointing = False

        block_types = extend_tuple(block_types, depth)
        blocks = []

        for i, t in enumerate(block_types):
            block_stride = stride if i == 0 else 1
            assert t in ('C', 'M')
            if t == 'C':
                conv_cls = ConvNeXtBlock if conv_cfg.block_type == 'convnext' else MbConvBlock
                blocks += [conv_cls(
                    in_chs,
                    out_chs,
                    stride=block_stride,
                    cfg=conv_cfg,
                    drop_path=drop_path[i],
                )]
            elif t == 'M':
                blocks += [MaxxVitBlock(
                    in_chs,
                    out_chs,
                    stride=block_stride,
                    conv_cfg=conv_cfg,
                    transformer_cfg=transformer_cfg,
                    drop_path=drop_path[i],
                    type=type,
                )]
            in_chs = out_chs
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


def cfg_window_size(cfg: MaxxVitTransformerCfg, img_size: Tuple[int, int]):
    if cfg.window_size is not None:
        assert cfg.grid_size
        return cfg
    partition_size_b = img_size[0] // cfg.partition_ratio_b, img_size[1] // cfg.partition_ratio_b
    partition_size_g = img_size[0] // cfg.partition_ratio_g, img_size[1] // cfg.partition_ratio_g
    cfg = replace(cfg, window_size=partition_size_b, grid_size=partition_size_g)
    return cfg


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        # x: (B, L, 1, 1, 3C)
        # b, c, h, w
        batch = x.size(0)
        # cav_num = x.size(1)

        if self.radix > 1:
            # x: b c 1, 2
            x = x.view(batch,
                       -1,
                       self.cardinality, self.radix)
            x = F.softmax(x, dim=2)
            # B, 3LC
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    def __init__(self, input_dim):
        super(SplitAttn, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.bn1 = nn.LayerNorm(input_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim * 2, bias=False)

        self.rsoftmax = RadixSoftmax(2, 1)

    def forward(self, window_list):
        # window list: [(B, L, H, W, C) * 3]
        # x list: [(B, C, H, W) * 2]
        assert len(window_list) == 2, 'only 2 windows are supported'

        x_1, x_2 = window_list[0], window_list[1]
        # B, L = x_1.shape[0], x_2.shape[1]
        B = x_1.shape[0]

        # global average pooling, B, L, H, W, C
        x_gap = x_1 + x_2
        # B, L, 1, 1, C
        # B, C, 1, 1
        # x_gap = x_gap.mean((2, 3), keepdim=True)
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_gap = self.act1(self.bn1(self.fc1(x_gap)))
        # B, L, 1, 1, 2C
        x_attn = self.fc2(x_gap)
        # B L 1 1 2C
        # x_attn = self.rsoftmax(x_attn).view(B, L, 1, 1, -1)
        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)

        out = x_1 * x_attn[:, 0:self.input_dim, :, :] + \
              x_2 * x_attn[:, self.input_dim:, :, :]

        return out


class MaxVit(nn.Module):
    def __init__(
            self,
            cfg: MaxxVitCfg,
            img_size: Union[int, Tuple[int, int]] = 224,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            drop_rate: float = 0.,
            drop_path_rate: float = 0.
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        transformer_cfg = cfg_window_size(cfg.transformer_cfg, img_size)
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = cfg.embed_dim[-1]
        self.embed_dim = cfg.embed_dim
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.split_attention = SplitAttn(7)

        self.stem = Stem(
            in_chs=in_chans,
            out_chs=cfg.stem_width,
            act_layer=cfg.conv_cfg.act_layer,
            norm_layer=cfg.conv_cfg.norm_layer,
            norm_eps=cfg.conv_cfg.norm_eps,
        )

        stride = self.stem.stride
        feat_size = tuple([i // s for i, s in zip(img_size, to_2tuple(stride))])

        num_stages = len(cfg.embed_dim)
        assert len(cfg.depths) == num_stages
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.depths)).split(cfg.depths)]

        in_chs = self.stem.out_chs
        stages_1 = []
        for i in range(num_stages):
            stage_stride = 2
            out_chs = cfg.embed_dim[i]
            feat_size = tuple([(r - 1) // stage_stride + 1 for r in feat_size])
            stages_1 += [MaxVitStage(
                in_chs,
                out_chs,
                depth=cfg.depths[i],
                block_types=cfg.block_type[i],
                conv_cfg=cfg.conv_cfg,
                transformer_cfg=transformer_cfg,
                # feat_size=feat_size,
                drop_path=dpr[i],
                type='block'
            )]
            stride *= stage_stride
            in_chs = out_chs
        self.stages_1 = nn.Sequential(*stages_1)

        in_chs = self.stem.out_chs
        stages_2 = []
        for i in range(num_stages):
            stage_stride = 2
            out_chs = cfg.embed_dim[i]
            feat_size = tuple([(r - 1) // stage_stride + 1 for r in feat_size])
            stages_2 += [MaxVitStage(
                in_chs,
                out_chs,
                depth=cfg.depths[i],
                block_types=cfg.block_type[i],
                conv_cfg=cfg.conv_cfg,
                transformer_cfg=transformer_cfg,
                # feat_size=feat_size,
                drop_path=dpr[i],
                type='grid'
            )]
            stride *= stage_stride
            in_chs = out_chs
        self.stages_2 = nn.Sequential(*stages_2)

        final_norm_layer = get_norm_layer(cfg.transformer_cfg.norm_layer)
        self.norm = final_norm_layer(self.num_features, eps=cfg.transformer_cfg.norm_eps)

        # Classifier head
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)

        # Weight init (default PyTorch init works well for AdamW if scheme not set)
        assert cfg.weight_init in ('', 'normal', 'trunc_normal', 'xavier_normal', 'vit_eff')
        if cfg.weight_init:
            named_apply(partial(self._init_weights, scheme=cfg.weight_init), self)

    def _init_weights(self, module, name, scheme=''):
        if hasattr(module, 'init_weights'):
            try:
                module.init_weights(scheme=scheme)
            except TypeError:
                module.init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            k for k, _ in self.named_parameters()
            if any(n in k for n in ["relative_position_bias_table", "rel_pos.mlp"])}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',  # stem and embed
            blocks=[(r'^stages\.(\d+)', None), (r'^norm', (99999,))]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is None:
            global_pool = self.head.global_pool.pool_type
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        input = self.stem(x)
        x_1 = self.stages_1(input)
        x_2 = self.stages_2(input)

        x = x_1 + x_2
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

def _create_maxxvit(variant, cfg_variant=None, pretrained=False, **kwargs):
    return build_model_with_cfg(
        MaxVit, variant, pretrained,
        model_cfg=model_cfgs[variant] if not cfg_variant else model_cfgs[cfg_variant],
        feature_cfg=dict(flatten_sequential=True),
        **kwargs)

@register_model
def maxvit_tiny_rework_224(pretrained=False, **kwargs):
    return _create_maxxvit('maxvit_tiny_rework_224', pretrained=pretrained, **kwargs)

@register_model
def maxvit_tiny_rework_224_bg(pretrained=False, **kwargs):
    return _create_maxxvit('maxvit_tiny_rework_224_bg', pretrained=pretrained, **kwargs)

@register_model
def maxvit_tiny_rework_224_gb(pretrained=False, **kwargs):
    return _create_maxxvit('maxvit_tiny_rework_224_gb', pretrained=pretrained, **kwargs)

@register_model
def maxvit_tiny_rework_256_22(pretrained=False, **kwargs):
    return _create_maxxvit('maxvit_tiny_rework_256_22', pretrained=pretrained, **kwargs)

@register_model
def maxvit_tiny_rework_256_28(pretrained=False, **kwargs):
    return _create_maxxvit('maxvit_tiny_rework_256_28', pretrained=pretrained, **kwargs)

@register_model
def maxvit_tiny_rework_256_82(pretrained=False, **kwargs):
    return _create_maxxvit('maxvit_tiny_rework_256_82', pretrained=pretrained, **kwargs)

@register_model
def maxvit_tiny_rework_256_88(pretrained=False, **kwargs):
    return _create_maxxvit('maxvit_tiny_rework_256_88', pretrained=pretrained, **kwargs)

@register_model
def maxvit_base_rework_256(pretrained=False, **kwargs):
    return _create_maxxvit('maxvit_base_rework_256', pretrained=pretrained, **kwargs)

@register_model
def maxvit_base_rework_256_2M(pretrained=False, **kwargs):
    return _create_maxxvit('maxvit_base_rework_256_2M', pretrained=pretrained, **kwargs)

@register_model
def maxvit_large_rework_256(pretrained=False, **kwargs):
    return _create_maxxvit('maxvit_large_rework_256', pretrained=pretrained, **kwargs)
