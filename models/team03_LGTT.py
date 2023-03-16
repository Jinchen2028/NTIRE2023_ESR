import math

import torch
import torch.nn as nn
import torch.nn.functional as f
from einops import rearrange


class MeanShift(nn.Module):
    r"""MeanShift for NTIRE 2023 Challenge on Efficient Super-Resolution.

    This implementation avoids counting the non-optimized parameters
        into the model parameters.

    Args:
        rgb_range (int):
        sign (int):
        data_type (str):

    Note:
        May slow down the inference of the model!

    """

    def __init__(self, rgb_range: int, sign: int = -1, data_type: str = 'DIV2K') -> None:
        super(MeanShift, self).__init__()

        self.sign = sign

        self.rgb_range = rgb_range
        self.rgb_std = (1.0, 1.0, 1.0)
        if data_type == 'DIV2K':
            # RGB mean for DIV2K 1-800
            self.rgb_mean = (0.4488, 0.4371, 0.4040)
        elif data_type == 'DF2K':
            # RGB mean for DF2K 1-3450
            self.rgb_mean = (0.4690, 0.4490, 0.4036)
        else:
            raise NotImplementedError(f'Unknown data type for MeanShift: {data_type}.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.Tensor(self.rgb_std)
        weight = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        bias = self.sign * self.rgb_range * torch.Tensor(self.rgb_mean) / std
        return f.conv2d(input=x, weight=weight.type_as(x), bias=bias.type_as(x))


class Conv2d1x1(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                 **kwargs) -> None:
        super(Conv2d1x1, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                        dilation=dilation, groups=groups, bias=bias, **kwargs)


class Conv2d3x3(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                 **kwargs) -> None:
        super(Conv2d3x3, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(3, 3), stride=stride, padding=(1, 1),
                                        dilation=dilation, groups=groups, bias=bias, **kwargs)


class ShiftConv2d1x1(nn.Conv2d):
    r"""ShiftConv2d1x1 for NTIRE 2023 Challenge on Efficient Super-Resolution.

    This implementation avoids counting the non-optimized parameters
        into the model parameters.

    Args:
        in_channels (int):
        out_channels (int):
        stride (tuple):
        dilation (tuple):
        bias (bool):
        shift_mode (str):
        val (float):

    Note:
        May slow down the inference of the model!

    """

    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), bias: bool = True, shift_mode: str = '+', val: float = 1.,
                 **kwargs) -> None:
        super(ShiftConv2d1x1, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                             dilation=dilation, groups=1, bias=bias, **kwargs)

        assert in_channels % 5 == 0, f'{in_channels} % 5 != 0.'
        self.in_channels = in_channels
        self.channel_per_group = in_channels // 5
        self.shift_mode = shift_mode
        self.val = val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cgp = self.channel_per_group
        mask = torch.zeros(self.in_channels, 1, 3, 3)
        if self.shift_mode == '+':
            mask[0 * cgp:1 * cgp, 0, 1, 2] = self.val
            mask[1 * cgp:2 * cgp, 0, 1, 0] = self.val
            mask[2 * cgp:3 * cgp, 0, 2, 1] = self.val
            mask[3 * cgp:4 * cgp, 0, 0, 1] = self.val
            mask[4 * cgp:, 0, 1, 1] = self.val
        elif self.shift_mode == 'x':
            mask[0 * cgp:1 * cgp, 0, 0, 0] = self.val
            mask[1 * cgp:2 * cgp, 0, 0, 2] = self.val
            mask[2 * cgp:3 * cgp, 0, 2, 0] = self.val
            mask[3 * cgp:4 * cgp, 0, 2, 2] = self.val
            mask[4 * cgp:, 0, 1, 1] = self.val
        else:
            raise NotImplementedError(f'Unknown shift mode for ShiftConv2d1x1: {self.shift_mode}.')

        x = f.conv2d(input=x, weight=mask.type_as(x), bias=None,
                     stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=self.in_channels)
        x = f.conv2d(input=x, weight=self.weight, bias=self.bias,
                     stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        return x


class TransformerGroup(nn.Module):
    r"""

    Args:
        sa_list:
        mlp_list:
        conv_list:

    """

    def __init__(self, sa_list: list, mlp_list: list, conv_list: list = None) -> None:
        super(TransformerGroup, self).__init__()

        assert len(sa_list) == len(mlp_list)

        self.sa_list = nn.ModuleList(sa_list)
        self.mlp_list = nn.ModuleList(mlp_list)
        self.conv = nn.Sequential(*conv_list if conv_list is not None else [nn.Identity()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w
        """
        for sa, mlp in zip(self.sa_list, self.mlp_list):
            x = x + sa(x)
            x = x + mlp(x)
        return self.conv(x)


class Upsampler(nn.Sequential):
    r"""Tail of the image restoration network.

    Args:
        upscale (int):
        in_channels (int):
        out_channels (int):
        upsample_mode (str):

    """

    def __init__(self, upscale: int, in_channels: int,
                 out_channels: int, upsample_mode: str = 'csr') -> None:

        layer_list = list()
        if upsample_mode == 'csr':  # classical
            if (upscale & (upscale - 1)) == 0:  # 2^n?
                for _ in range(int(math.log(upscale, 2))):
                    layer_list.append(Conv2d3x3(in_channels, 4 * in_channels))
                    layer_list.append(nn.PixelShuffle(2))
            elif upscale == 3:
                layer_list.append(Conv2d3x3(in_channels, 9 * in_channels))
                layer_list.append(nn.PixelShuffle(3))
            else:
                raise ValueError(f'Upscale {upscale} is not supported.')
            layer_list.append(Conv2d3x3(in_channels, out_channels))
        elif upsample_mode == 'lsr':  # lightweight
            layer_list.append(Conv2d3x3(in_channels, out_channels * (upscale ** 2)))
            layer_list.append(nn.PixelShuffle(upscale))
        elif upsample_mode == 'denoising' or upsample_mode == 'deblurring' or upsample_mode == 'deraining':
            layer_list.append(Conv2d3x3(in_channels, out_channels))
        else:
            raise ValueError(f'Upscale mode {upscale} is not supported.')

        super(Upsampler, self).__init__(*layer_list)


class _Swish(torch.autograd.Function):  # noqa
    @staticmethod
    def forward(ctx, i):  # noqa
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    r"""A memory-efficient implementation of Swish. The original code is from
        https://github.com/zudi-lin/rcan-it/blob/main/ptsr/model/_utils.py.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa
        return _Swish.apply(x)


class GTModeler(nn.Module):
    r"""Self attention for 4D input.

    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        attn_layer (list): layers used to calculate attn
        proj_layer (list): layers used to proj output
        window_list (tuple): list of window sizes. Input will be equally divided
            by channel to use different windows sizes
        shift_list (tuple): list of shift sizes

    Returns:
        b c h w -> b c h w
    """

    def __init__(self, dim: int,
                 num_heads: int,
                 attn_layer: list = None,
                 proj_layer: list = None,
                 window_list: tuple = ((8, 8),),
                 shift_list: tuple = None,
                 ) -> None:
        super(GTModeler, self).__init__()

        self.dim = dim
        self.num_heads = num_heads

        self.window_list = window_list
        if shift_list is not None:
            assert len(shift_list) == len(window_list)
            self.shift_list = shift_list
        else:
            self.shift_list = ((0, 0),) * len(window_list)

        self.attn = nn.Sequential(*attn_layer if attn_layer is not None else [nn.Identity()])
        self.proj = nn.Sequential(*proj_layer if proj_layer is not None else [nn.Identity()])

    @staticmethod
    def check_image_size(x: torch.Tensor, window_size: tuple) -> torch.Tensor:
        _, _, h, w = x.size()
        windows_num_h = math.ceil(h / window_size[0])
        windows_num_w = math.ceil(w / window_size[1])
        mod_pad_h = windows_num_h * window_size[0] - h
        mod_pad_w = windows_num_w * window_size[1] - w
        return f.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w

        Returns:
            b c h w -> b c h w
        """
        # calculate qkv
        qkv = self.attn(x)
        _, C, _, _ = qkv.size()

        # split channels
        qkv_list = torch.split(qkv, [C // len(self.window_list)] * len(self.window_list), dim=1)

        output_list = list()
        for attn_slice, window_size, shift_size in zip(qkv_list, self.window_list, self.shift_list):
            _, _, h, w = attn_slice.size()
            attn_slice = self.check_image_size(attn_slice, window_size)

            # roooll!
            if shift_size != (0, 0):
                attn_slice = torch.roll(attn_slice, shifts=shift_size, dims=(2, 3))

            # cal attn
            _, _, H, W = attn_slice.size()
            q, v = rearrange(attn_slice, 'b (qv head c) (nh ws1) (nw ws2) -> qv (b head nh nw) (ws1 ws2) c',
                             qv=2, head=self.num_heads,
                             ws1=window_size[0], ws2=window_size[1])
            attn = (q @ q.transpose(-2, -1))
            attn = f.softmax(attn, dim=-1)
            output = rearrange(attn @ v, '(b head nh nw) (ws1 ws2) c -> b (head c) (nh ws1) (nw ws2)',
                               head=self.num_heads,
                               nh=H // window_size[0], nw=W // window_size[1],
                               ws1=window_size[0], ws2=window_size[1])

            # roooll back!
            if shift_size != (0, 0):
                output = torch.roll(output, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))

            output_list.append(output[:, :, :h, :w])

        # proj output
        output = self.proj(torch.cat(output_list, dim=1))
        return output


class PixelMixer(nn.Module):
    r"""Pixel Mixer for NTIRE 2023 Challenge on Efficient Super-Resolution.

    This implementation avoids counting the non-optimized parameters
        into the model parameters.

    Args:
        planes (int):
        mix_margin (int):

    Note:
        May slow down the inference of the model!

    """

    def __init__(self, planes: int, mix_margin: int = 1) -> None:
        super(PixelMixer, self).__init__()

        assert planes % 5 == 0

        self.planes = planes
        self.mix_margin = mix_margin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mix_margin

        mask = torch.zeros(self.planes, 1, m * 2 + 1, m * 2 + 1)
        mask[3::5, 0, 0, m] = 1.
        mask[2::5, 0, -1, m] = 1.
        mask[1::5, 0, m, 0] = 1.
        mask[0::5, 0, m, -1] = 1.
        mask[4::5, 0, m, m] = 1.

        return f.conv2d(input=f.pad(x, pad=(m, m, m, m), mode='circular'),
                        weight=mask.type_as(x), bias=None, stride=(1, 1), padding=(0, 0),
                        dilation=(1, 1), groups=self.planes)


class LTModeler(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.token_mixer = PixelMixer(planes=dim)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.token_mixer(x) - x)


class Mlp(nn.Module):
    r"""Multi-layer perceptron.

    Args:
        in_features: Number of input channels
        hidden_features:
        out_features: Number of output channels
        act_layer:

    """

    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, act_layer: nn.Module = nn.GELU) -> None:
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = ShiftConv2d1x1(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = ShiftConv2d1x1(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class LGTGroup(TransformerGroup):
    def __init__(self, dim: int, num_block: int, num_heads: int, num_gtmodeler: int,
                 window_list: tuple = None, shift_list: tuple = None,
                 mlp_ratio: int = None, act_layer: nn.Module = nn.GELU) -> None:
        token_mixer_list = [LTModeler(dim) if _ > (num_gtmodeler - 1)
                            else GTModeler(dim=dim, num_heads=num_heads,
                                           attn_layer=[Conv2d1x1(dim, dim * 2),
                                                       nn.BatchNorm2d(dim * 2)],
                                           proj_layer=[Conv2d1x1(dim, dim)],
                                           window_list=window_list,
                                           shift_list=shift_list if (_ + 1) % 2 == 0 else None)
                            for _ in range(num_block)]

        mlp_list = [Mlp(dim, dim * mlp_ratio, act_layer=act_layer)
                    for _ in range(num_block)]

        super(LGTGroup, self). \
            __init__(sa_list=token_mixer_list, mlp_list=mlp_list, conv_list=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w
        """
        for sa, mlp in zip(self.sa_list, self.mlp_list):
            x = x + sa(x)
            x = x + mlp(x)

        return self.conv(x)


class LGTT(nn.Module):
    r"""Local-Global Term Transformer
    """

    def __init__(self, upscale: int = 4, num_in_ch: int = 3, num_out_ch: int = 3, task: str = 'lsr',
                 dim: int = 30, n_groups: int = 4, n_blocks: int = 6,
                 num_heads: int = 3, mlp_ratio: int = 2, num_gtmodeler: int = 1,
                 window_list: tuple = [[36, 8], [8, 36]], shift_list: tuple = [[18, 4], [4, 18]]):
        super(LGTT, self).__init__()

        self.sub_mean = MeanShift(255, sign=-1, data_type='DF2K')
        self.add_mean = MeanShift(255, sign=1, data_type='DF2K')

        self.head = Conv2d3x3(num_in_ch, dim)

        self.body = nn.Sequential(*[LGTGroup(dim=dim, num_block=n_blocks, num_heads=num_heads,
                                             num_gtmodeler=num_gtmodeler,
                                             window_list=window_list, shift_list=shift_list,
                                             mlp_ratio=mlp_ratio, act_layer=Swish)
                                    for _ in range(n_groups)])

        self.tail = Upsampler(upscale=upscale, in_channels=dim, out_channels=num_out_ch, upsample_mode=task)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reduce the mean of pixels
        sub_x = self.sub_mean(x)

        # head
        head_x = self.head(sub_x)

        # body
        body_x = self.body(head_x)
        body_x = body_x + head_x

        # tail
        tail_x = self.tail(body_x)

        # add the mean of pixels
        add_x = self.add_mean(tail_x)

        return add_x
