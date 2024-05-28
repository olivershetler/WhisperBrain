import torch
from torch import nn, Tensor
from typing import Dict, Optional
from whisper.model import Whisper, ModelDimensions, LayerNorm, AudioEncoder

from whisper.decoding import DecodingTask
from whisper.decoding import detect_language as detect_language_function
from whisper.transcribe import transcribe as transcribe_function

@torch.no_grad()
def decode_function(model, brain_data, options):
    result = DecodingTask(model, options).run(brain_data)
    return result

class Residual3D(nn.Module):
    """This block is used to apply a 3D convolution to the input tensor with residual connections."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.gelu = nn.GELU()

    def forward(self, x: Tensor):
        residual = x
        x = self.conv(x)
        x = self.gelu(x)
        x += residual
        return x

class BrainRegionIntakeBlock(nn.Module):
    """This block is used to intake the brain region data and convert it into a tensor that can be used by the model.
    There are four brain regions and each has a tensor of shape (N, T, 3, 8, 8) where N is the batch size and T is the number of time steps; 3 is the number of channels, 8 is the height and 8 is the width of the tensor.

    This intake block will apply convolutions to the tensor and then convert it into a tensor of shape (N, T, F) where F is the number of features for each brain region's embedding dimension.

    The output tensor will be of shape (N, T, F). These tensors will be combined within the BrainPreNet block to create a tensor of shape (N, T, D) where D is the total number of features for all brain regions.
    """
    def __init__(self):
        super().__init__() #TODO revise this block, since these are just random guesses

        def conv_block(depth, in_channels, out_channels, kernel_size, stride, padding):
            block_list = []
            for i in range(depth):
                if i == 0:
                    block_list.append(nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding))
                else:
                    block_list.append(nn.Conv3d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding))
                block_list.append(nn.GELU())
            return nn.Sequential(*block_list)

        def res_block (depth, in_channels, out_channels, kernel_size, stride, padding):
            block_list = []
            for i in range(depth):
                if i == 0:
                    block_list.append(Residual3D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding))
                else:
                    block_list.append(Residual3D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding))
            return nn.Sequential(*block_list)

        self.conv_intake = nn.Sequential(
                            nn.Conv3d(
                                in_channels=5,
                                out_channels=5,
                                kernel_size=(3, 3, 1),
                                stride=(1, 1, 1),
                                padding=(1, 1, 0)
                            ),
                            nn.GELU())

        self.spatial_res = res_block(
            depth=3,
            in_channels=5,
            out_channels=5,
            kernel_size=(3, 3, 1),
            stride=(1, 1, 1),
            padding=(1, 1, 0)
        )
        self.temporal_res = res_block(
            depth=3,
            in_channels=5,
            out_channels=5,
            kernel_size=(1, 1, 3),
            stride=(1, 1, 1),
            padding=(0, 0, 1)
        )

        self.dense_block = nn.Sequential(
            nn.Linear(5 * 8 * 8, (5 * 8 * 8) // 2), # 5 * 8 * 8 = 320 -> 160
            nn.GELU(),
            nn.Linear((5 * 8 * 8) // 2, (5 * 8 * 8) // 4), # 160 -> 80
            nn.GELU(),
            nn.Linear((5 * 8 * 8) // 4, (5 * 8 * 8) // 8), # 80 -> 40
            nn.GELU(),
            nn.Linear((5 * 8 * 8) // 8, (5 * 8 * 8) // 16), # 40 -> 20
            nn.GELU(),
        )

        self.ln = LayerNorm(20)
        self.fn = nn.Linear(20, 20)

    def forward(self, x: Tensor):
        # x is of shape (N, 5, 8, 8, T)
        x = self.conv_intake(x)
        x = self.spatial_res(x)
        x = self.temporal_res(x)
        # reshape the tensor to (N, T, 5, 8, 8)
        x = x.permute(0, 4, 1, 2, 3)
        # reshape the tensor to (N, T, 5 * 8 * 8)
        x = x.view(x.shape[0], x.shape[1], -1)
        # reshape the tensor to (N*T, F)
        x = x.view(-1, x.shape[-1])
        x = self.dense_block(x)
        x = self.ln(x)
        x = self.fn(x)
        # reshape the tensor to (N, T, F)
        x = x.view(-1, x.shape[0], x.shape[1])
        return x

class BrainPreNet(nn.Module):
    def __init__(self, n_mels: int = 80):
        # n_mels must be same as n_mels in AudioEncoder
        super().__init__()
        self.inferior_6v_intake = BrainRegionIntakeBlock()
        self.superior_6v_intake = BrainRegionIntakeBlock()
        self.inferior_44_intake = BrainRegionIntakeBlock()
        self.superior_44_intake = BrainRegionIntakeBlock()
        # next, we define the layers that will be used to combine the features from the 4 brain regions
        # the output is a tensor of shape (N, T, D) where D is the total number of features for all brain regions
        # which is to say n_mels = D so that the output can be passed to the AudioEncode
        # r
        self.combination_layers = nn.Sequential(
            nn.Linear(80, n_mels),
            nn.GELU(),
            nn.Linear(n_mels, n_mels),
            nn.GELU(),
        )
        self.fn = nn.Linear(n_mels, n_mels)


    def forward(self, x: Tensor):
        N = x.shape[0]
        T = x.shape[1]
        # Divide the tensor (N, T, 2, 2, 5, 8, 8) into 4 tensors of shape (N, T, 1, 1, 5, 8, 8) and then squeeze the extra dimensions into (N, T, 5, 8, 8) then permute to (N, 5, 8, 8, T)
        inferior_6v = x[:, 0, 1].squeeze((1, 2))
        superior_6v = x[:, 0, 0].squeeze((1, 2))
        inferior_44 = x[:, 1, 1].squeeze((1, 2))
        superior_44 = x[:, 1, 0].squeeze((1, 2))

        # Apply the intake blocks to convert the tensors into tensors of shape (N, T, F)
        inferior_6v = self.inferior_6v_intake(inferior_6v)
        superior_6v = self.superior_6v_intake(superior_6v)
        inferior_44 = self.inferior_44_intake(inferior_44)
        superior_44 = self.superior_44_intake(superior_44)

        # Combine the features from the 4 brain regions
        x = torch.cat([inferior_6v, superior_6v, inferior_44, superior_44], dim=-1)
        x = self.combination_layers(x)
        x = self.fn(x)
        x = x.permute(0, 2, 1)
        return x

class BrainEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.prenet = BrainPreNet()
        self.encoder = AudioEncoder(
            n_mels=n_mels,
            n_ctx=n_ctx,
            n_state=n_state,
            n_head=n_head,
            n_layer=n_layer,
        )

    def forward(self, x: Tensor):
        x = self.prenet(x)
        x = self.encoder(x)
        return x


class WhisperBrain(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.whisper = Whisper(self.dims)
        self.brain_encoder = BrainEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.mode = 'audio'

    def forward(self, signals: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.mode == 'audio':
            return self.whisper(signals, tokens)
        elif self.mode == 'brain':
            return self.whisper.logits(tokens, self.brain_encoder(signals))
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def embed_brain(self, signal: Tensor):
        return self.brain_encoder(signal)

    def embed_audio(self, mel: Tensor):
        return self.whisper.embed_audio(mel)

    def logits(self, tokens, embeddings):
        return self.whisper.logits(tokens, embeddings)

    @property
    def encoder(self):
        if self.mode == 'audio':
            return self.whisper.encoder
        elif self.mode == 'brain':
            return self.brain_encoder
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @property
    def decoder(self):
        return self.whisper.decoder

    @property
    def is_multilingual(self):
        return self.whisper.is_multilingual

    @property
    def num_languages(self):
        return self.whisper.num_languages

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        return self.whisper.install_kv_cache_hooks(cache)

    def toggle_mode(self, mode: str):
        if mode not in ['audio', 'brain']:
            raise ValueError(f"Invalid mode: {mode}")
        else:
            self.mode = mode

    def toggle_freeze(self, part: str, unfreeze: str=False):
        options = {
            'whisper': self.whisper,
            'text': self.whisper.decoder,
            'audio': self.whisper.encoder,
            'brain': self.brain_encoder
        }
        if unfreeze not in [True, False]:
            raise ValueError("unfreeze must be a boolean value")
        else:
            try:
                for param in options[part].parameters():
                    param.requires_grad = unfreeze # True for unfreeze, False for freeze
            except KeyError:
                raise ValueError(f"Invalid mode: {part}. Valid modes are {options.keys()}")

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function

