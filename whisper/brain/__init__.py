import io
import os
import torch
from typing import List, Optional, Union

from whisper import load_model, available_models, ModelDimensions
from whisper.brain.model import WhisperBrain
from whisper.model import Whisper
from whisper.brain.dataset import WillettDataset
from whisper.brain.losses import mse_adjusted_temporal_gaussian_infonce_loss

__all__ = [
    "load_checkpoint",
    "initialize_from_whisper",
    "WillettDataset",
    "mse_adjusted_temporal_gaussian_infonce_loss"
]

def initialize_from_whisper(name: str, device: Optional[Union[str, torch.device]] = None, download_root: str = None, in_memory: bool = False) -> Whisper:
    whisper = load_model(name, device, download_root, in_memory)
    whisper_brain = WhisperBrain(whisper.dims)
    whisper_brain.whisper = whisper
    whisper_brain.to(device)
    return whisper_brain

def load_checkpoint(name: str, device:  Optional[Union[str, torch.device]] = None, in_memory: bool = False) -> WhisperBrain:
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name : str path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    in_memory: bool
        whether to preload the model weights into host memory

    Returns
    -------
    model : WhisperBrain
        The WhisperBrain BrainToText model instance
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location=device)
    del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])
    model = WhisperBrain(dims)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model.to(device)
