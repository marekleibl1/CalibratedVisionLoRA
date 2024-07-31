# Based on the following implementations: 
# https://raw.githubusercontent.com/JamesQFreeman/LoRA-ViT
# https://github.com/lukemelas/PyTorch-Pretrained-ViT

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from torch.nn.parameter import Parameter
from torchvision import transforms


import numpy as np
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import torch
from torch.utils import model_zoo
from typing import Optional


def get_base_config():
    """Base ViT config ViT"""
    return dict(
        dim=768,
        ff_dim=3072,
        num_heads=12,
        num_layers=12,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        representation_size=768,
        classifier='token'
    )


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = get_base_config()
    config.update(dict(patches=(16, 16)))
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.update(dict(patches=(32, 32)))
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = get_base_config()
    config.update(dict(
        patches=(16, 16),
        dim=1024,
        ff_dim=4096,
        num_heads=16,
        num_layers=24,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        representation_size=1024
    ))
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.update(dict(patches=(32, 32)))
    return config


def drop_head_variant(config):
    config.update(dict(representation_size=None))
    return config


PRETRAINED_MODELS = {
    'B_16': {
        'config': get_b16_config(),
        'num_classes': 21843,
        'image_size': (224, 224),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16.pth"
    },
    'B_32': {
        'config': get_b32_config(),
        'num_classes': 21843,
        'image_size': (224, 224),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32.pth"
    },
    'L_16': {
        'config': get_l16_config(),
        'num_classes': 21843,
        'image_size': (224, 224),
        'url': None
    },
    'L_32': {
        'config': get_l32_config(),
        'num_classes': 21843,
        'image_size': (224, 224),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32.pth"
    },
    'B_16_imagenet1k': {
        'config': drop_head_variant(get_b16_config()),
        'num_classes': 1000,
        'image_size': (384, 384),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth"
    },
    'B_32_imagenet1k': {
        'config': drop_head_variant(get_b32_config()),
        'num_classes': 1000,
        'image_size': (384, 384),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32_imagenet1k.pth"
    },
    'L_16_imagenet1k': {
        'config': drop_head_variant(get_l16_config()),
        'num_classes': 1000,
        'image_size': (384, 384),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_16_imagenet1k.pth"
    },
    'L_32_imagenet1k': {
        'config': drop_head_variant(get_l32_config()),
        'num_classes': 1000,
        'image_size': (384, 384),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32_imagenet1k.pth"
    },
}


def load_pretrained_weights(
    model,
    model_name=None,
    weights_path=None,
    load_first_conv=True,
    load_fc=True,
    load_repr_layer=False,
    resize_positional_embedding=False,
    verbose=True,
    strict=True,
):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): Full model (a nn.Module)
        model_name (str): Model name (e.g. B_16)
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_first_conv (bool): Whether to load patch embedding.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        resize_positional_embedding=False,
        verbose (bool): Whether to print on completion
    """
    assert bool(model_name) ^ bool(
        weights_path), 'Expected exactly one of model_name or weights_path'

    # Load or download weights
    if weights_path is None:
        url = PRETRAINED_MODELS[model_name]['url']
        if url:
            state_dict = model_zoo.load_url(url)
        else:
            raise ValueError(
                f'Pretrained model for {model_name} has not yet been released')
    else:
        state_dict = torch.load(weights_path)

    # Modifications to load partial state dict
    expected_missing_keys = []
    if not load_first_conv and 'patch_embedding.weight' in state_dict:
        expected_missing_keys += ['patch_embedding.weight',
                                  'patch_embedding.bias']
    if not load_fc and 'fc.weight' in state_dict:
        expected_missing_keys += ['fc.weight', 'fc.bias']
    if not load_repr_layer and 'pre_logits.weight' in state_dict:
        expected_missing_keys += ['pre_logits.weight', 'pre_logits.bias']
    for key in expected_missing_keys:
        state_dict.pop(key)

    # Change size of positional embeddings
    if resize_positional_embedding:
        posemb = state_dict['positional_embedding.pos_embedding']
        posemb_new = model.state_dict()['positional_embedding.pos_embedding']
        state_dict['positional_embedding.pos_embedding'] = \
            resize_positional_embedding_(posemb=posemb, posemb_new=posemb_new,
                                         has_class_token=hasattr(model, 'class_token'))
        maybe_print('Resized positional embeddings from {} to {}'.format(
                    posemb.shape, posemb_new.shape), verbose)

    # Load state dict
    ret = model.load_state_dict(state_dict, strict=False)
    if strict:
        assert set(ret.missing_keys) == set(expected_missing_keys), \
            'Missing keys when loading pretrained weights: {}'.format(
                ret.missing_keys)
        assert not ret.unexpected_keys, \
            'Missing keys when loading pretrained weights: {}'.format(
                ret.unexpected_keys)
        maybe_print('Loaded pretrained weights.', verbose)
    else:
        maybe_print('Missing keys when loading pretrained weights: {}'.format(
            ret.missing_keys), verbose)
        maybe_print('Unexpected keys when loading pretrained weights: {}'.format(
            ret.unexpected_keys), verbose)
        return ret


def maybe_print(s: str, flag: bool):
    if flag:
        print(s)


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def resize_positional_embedding_(posemb, posemb_new, has_class_token=True):
    """Rescale the grid of position embeddings in a sensible manner"""
    from scipy.ndimage import zoom

    # Deal with class token
    ntok_new = posemb_new.shape[1]
    if has_class_token:  # this means classifier == 'token'
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

    # Get old and new grid sizes
    gs_old = int(np.sqrt(len(posemb_grid)))
    gs_new = int(np.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

    # Rescale grid
    zoom_factor = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = zoom(posemb_grid, zoom_factor, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)

    # Deal with class token and return
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None  # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""

    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


class ViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000
    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
        self,
        name: Optional[str] = None,
        pretrained: bool = False,
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        positional_embedding: str = '1d',
        in_channels: int = 3,
        image_size: Optional[int] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()

        # Configuration
        if name is None:
            check_msg = 'must specify name of pretrained model'
            assert not pretrained, check_msg
            assert not resize_positional_embedding, check_msg
            if num_classes is None:
                num_classes = 1000
            if image_size is None:
                image_size = 384
        else:  # load pretrained model
            assert name in PRETRAINED_MODELS.keys(), \
                'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            config = PRETRAINED_MODELS[name]['config']
            patches = config['patches']
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            attention_dropout_rate = config['attention_dropout_rate']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']
            classifier = config['classifier']
            if image_size is None:
                image_size = PRETRAINED_MODELS[name]['image_size']
            if num_classes is None:
                num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.image_size = image_size

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1

        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()

        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads,
                                       ff_dim=ff_dim, dropout=dropout_rate)

        # Representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim

        # Classifier head
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        self.fc = nn.Linear(pre_logits_size, num_classes)

        # Initialize weights
        self.init_weights()

        # Load pretrained model
        if pretrained:
            pretrained_num_channels = 3
            pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
            pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
            load_pretrained_weights(
                self, name,
                load_first_conv=(in_channels == pretrained_num_channels),
                load_fc=(num_classes == pretrained_num_classes),
                load_repr_layer=load_repr_layer,
                resize_positional_embedding=(
                    image_size != pretrained_image_size),
            )

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    # nn.init.constant(m.bias, 0)
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.
        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(
                b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)  # b,gh*gw+1,d
        x = self.transformer(x)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            x = self.norm(x)[:, 0]  # b,d
            x = self.fc(x)  # b,num_classes
        return x



def random_matrix_and_inverse(n, m):
    """
    Genera a random orthogonal matrix and it's pseudo inverse.

    Update: There's no need to have the two matrixes (pseudo)inverse to each other 
    as they operate on different feature space. We can just generate them independentely.
    """
    # Generate a random matrix
    A = np.random.randn(n, m)
    # Perform QR decomposition to get a radnom orthogonal matrix
    Q, R = np.linalg.qr(A)
    # Adjust Q to ensure it is uniformly distributed
    Q *= np.sign(np.diag(R))
    # Compute pseuoinverse
    Q_inv = np.linalg.pinv(Q)
    return Q, Q_inv


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module, r: int, alpha: int, 
                 random_projections=False, trainable_projections=False):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b
        self.r = r
        self.alpha = alpha
        self.random_projections=random_projections

        if random_projections: 
            n = w.in_features
            m = w_a.in_features
            
            C, D = random_matrix_and_inverse(n, m)

            self.C = nn.Linear(in_features=n, out_features=m, bias=False)
            self.D = nn.Linear(in_features=m, out_features=n, bias=False)

            self.C.weight.requires_grad = trainable_projections
            self.D.weight.requires_grad = trainable_projections
        
            # Set the weight matrix
            with torch.no_grad():
                self.C.weight.copy_(torch.tensor(5 * C.T))
                self.D.weight.copy_(torch.tensor(5 * D.T)) 
 

    

    def forward(self, x):
        if self.random_projections:
            # print('x shape', x.shape, 'C shape', self.C.shape)
            # return self.w(x) + (self.alpha // self.r) * (self.w_b( self.w_a(self.C * x) * self.D))
            return self.w(x) + (self.alpha // self.r) * self.D(self.w_b( self.w_a(self.C (x))))
            
        return self.w(x) + (self.alpha // self.r) * self.w_b(self.w_a(x))


class ProjectedLastLayer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, trainable_projections=False):
        super(ProjectedLastLayer, self).__init__()

        self.linear1 = nn.Linear(in_features, hidden_features, bias=False)
        self.linear2 = nn.Linear(hidden_features, out_features)

        self.linear1.weight.requires_grad = trainable_projections

        C, _ = random_matrix_and_inverse(in_features, hidden_features)

        with torch.no_grad():
            self.linear1.weight.copy_(torch.tensor(C.T))

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
 

class LoRA_ViT(nn.Module):
    """Applies low-rank adaptation to a vision transformer.

    Args:
        vit_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, vit_model: ViT, **config):
        super(LoRA_ViT, self).__init__()

        r = config['lora_rank']
        alpha = config.get('alpha', 4)
        num_classes = config['n_classes']
        lora_layer = config.get('lora_layers')
        random_projections = config.get('random_projections', False)
        random_projections_dim =config.get('random_projections_dim', False)
        trainable_projections =  config.get('trainable_projections', False)
        
        assert r > 0
        assert alpha > 0
        base_vit_dim = vit_model.transformer.blocks[0].attn.proj_q.in_features
        dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.transformer.blocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        np.random.seed(42) 

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.transformer.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_q_linear = blk.attn.proj_q
            w_v_linear = blk.attn.proj_v

            if random_projections: 
                dim = random_projections_dim
            
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.proj_q = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q, r, alpha, random_projections, trainable_projections)
            blk.attn.proj_v = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v, r, alpha, random_projections, trainable_projections)

        self.reset_parameters()
        self.lora_vit = vit_model

        if random_projections:
            self.lora_vit.fc = ProjectedLastLayer(vit_model.fc.in_features, random_projections_dim ,num_classes, trainable_projections)
        else: 
            self.lora_vit.fc = nn.Linear(vit_model.fc.in_features, num_classes)

    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.fc.in_features
        _out = self.lora_vit.fc.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.fc.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.lora_vit.fc.in_features
        _out = self.lora_vit.fc.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.fc.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        _in = self.lora_vit.fc.in_features
        _out = self.lora_vit.fc.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.fc.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                
            _in = self.lora_vit.fc.in_features
            _out = self.lora_vit.fc.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.fc.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)


def create_lora_vit_model(**config):

    # Get a pretrained model
    model = ViT('B_16_imagenet1k', pretrained=True)
    model.eval()
    
    transform =  transforms.Compose([
            transforms.Resize((384, 384)), 
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])
    
    
    # def test_inference():
    #     img = Image.fromarray((np.random.rand(100, 100, 3) * 255).astype(np.uint8)) # Image.open('img.jpg')
    #     img_transformed = transform(img).unsqueeze(0)
        
    #     with torch.no_grad():
    #         outputs = model(img_transformed)
    
    #     print('Transformed image shape', img_transformed.shape) # torch.Size([1, 3, 384, 384])
    #     print(outputs.shape)  # (1, 1000)
    
    # # Test that we can generate a prediction on a random image
    # test_inference()
    
    lora_model = LoRA_ViT(model, **config)
    return lora_model, transform

