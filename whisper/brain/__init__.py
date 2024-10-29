import io
import os
import torch
from typing import List, Optional, Union
from pathlib import Path

from whisper import load_model, available_models, ModelDimensions
from whisper.brain.model import WhisperBrain, EmbeddingDiscriminator, LogitsDiscriminator
from whisper.model import Whisper
from whisper.brain.dataset import WillettDataset
from whisper.brain.losses import ClipLoss
from whisper.brain.trainers import *

def initialize_from_whisper(name: str, device: Optional[Union[str, torch.device]] = None, download_root: str = None, in_memory: bool = False) -> Whisper:
    whisper = load_model(name, device, download_root, in_memory)
    whisper_brain = WhisperBrain(whisper.dims)
    whisper_brain.whisper = whisper
    # copy the weights from the whisper.encoder to the whisper_brain.encoder.encoder
    whisper_brain.brain_encoder.encoder.load_state_dict(whisper.encoder.state_dict().copy())
    whisper_brain.to(device)
    return whisper_brain

def init_discriminator(dims: ModelDimensions, target: str = "embedding", device: Optional[Union[str, torch.device]] = None) -> List[Union[EmbeddingDiscriminator, LogitsDiscriminator]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target_options = {
        "embedding": lambda dims: EmbeddingDiscriminator(dims).to(device),
        "logits": lambda dims: LogitsDiscriminator(dims).to(device)
    }
    if target not in target_options:
        raise ValueError(f"target must be one of {target_options}")
    return target_options[target](dims)

def load_checkpoint(name: str,
                    model_type: str = "WhisperBrain",
                    device:  Optional[Union[str, torch.device]] = None,
                    in_memory: bool = False,
                    alt_dims=None) -> WhisperBrain:
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
        if in_memory:
            with open(name, "rb") as fp:
                checkpoint_file = fp.read()
        else:
            checkpoint_file = name
    else:
        name_dir = Path(name).parent
        if not os.path.isdir(name_dir):
            raise RuntimeError(
                f"Model {name} not found; the directory {name_dir} does not exist and the file {name} does not exist."
            )
        else:
            files = list(name_dir.glob("*.pt"))
            if len(files) == 0:
                raise RuntimeError(
                    f"Model {name} not found; no .pt files found in {name_dir}."
                )
            elif len(files) > 1:
                raise RuntimeError(
                    f"Model {name} not found; the directory {name_dir} contains multiple .pt files:\n\n{files}"
                )

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location=device)
    del checkpoint_file

    try:
        dims = checkpoint["dims"]
    except Exception as e:
        if alt_dims is not None:
            dims = alt_dims
        else:
            message = f"Could not load model dimensions from {name}; provide alt_dims to override. Traceback: {e}"
            raise RuntimeError(message)
    type_options = {
        "WhisperBrain": WhisperBrain,
        "EmbeddingDiscriminator": EmbeddingDiscriminator,
        "LogitsDiscriminator": LogitsDiscriminator
    }
    model = type_options[model_type](dims)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError:
        message = f"Model {name} may not be not a {model_type} model; available models = {type_options.keys()}"
    return model.to(device)

