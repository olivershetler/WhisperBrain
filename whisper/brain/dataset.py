import h5py
import json
import torch
import numpy as np

from whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES
import torchaudio
import torchaudio.transforms as at
from scipy.signal import lfiltic, lfilter


def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    """Taken from
    https://colab.research.google.com/drive/1P4ClLkPmfsaKn2tBbRp0nVjGMRKR-EWz?usp=sharing#scrollTo=DM_LjZQZFb1b
    """
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

class WillettDataset:
    """A class for representing a preprocessed version of the Willett et al. 2023 dataset.
    The dataset consists of implant electrode recordings from 4 brain regions: inferior 6v, superior 6v, inferior 44, and superior 44. Each of the brain regions has an 8x8 electrode array, and the recordings are bucketed at 20ms intervals. The dataset is preprocessed to include the following features for each sample.
    - neural_data: a tensor of shape (T, 4, 4, 8, 8) representing the raw electrode recordings from the 4 brain regions with 4 filters applied to each electrode on each 8x8 grid.
    - audio: a tensor of shape (T, F) representing a mel spectrogram of the audio signal.
    - text: a string representing the text transcription of the audio signal.
    - training_mode: a boolean indicating whether the sample is in training mode or not. In training mode, the brain signals are lightly augmented.

    The brain signals were loaded from matlab files provided for a competition. They were reshaped using a function that was written to reflect the electrode array layout in the readme files accompanying the data.

    The dataset has a training mode where the brain signals are lightly augmented by adding random noise, bias and small transpositions (1 or 2 electrodes) to the signals.

    The audio signals were generated from an openai speech-to-text model in a preprocessing script. They were modified to have the same length as the brain signals using a simple padding algorithm.

    The text are taken from the matlab files and some cleaning was done using some functions borrowed from the codebase that came with the original Willett et al. 2023 paper.

    The initialization of the Willett2023Dataset class requires the following parameters
    """
    def __init__(self, hf_path, partition, device=None):
        self.hf_path = hf_path
        self.partition = partition
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        with h5py.File(hf_path, 'r') as hf:
            hfp = hf[partition]
            self.n_days = int(hfp.attrs['n_days'])
            self.n_sentences = int(hfp.attrs['n_sentences'])
            self.n_time_steps = int(hfp.attrs['n_time_steps'])
            self.max_sentence_length = int(hfp.attrs['max_sentence_length'])
            self.sentence_start_timestep_idx = json.loads(hfp.attrs['sentence_start_timestep_idx'])
            self.block_start_timestep_idx = json.loads(hfp.attrs['block_start_timestep_idx'])
            self.day_start_timestep_idx = json.loads(hfp.attrs['day_start_timestep_idx'])
            self.bad_days = json.loads(hfp.attrs['bad_days'])
            # data are stored using hf[partition][day_n][block_n][sentence_n]
            # we need to make an index that maps sentences to the (day_n, block_n, sentence_n) tuple
            self.sentence_idx = {}
            i = 0
            for day_n in hf[partition].keys():
                for block_n in hf[partition][day_n].keys():
                    for sentence_n in hf[partition][day_n][block_n].keys():
                        self.sentence_idx[i] = (day_n, block_n, sentence_n)
                        i += 1

    def __len__(self):
        return len(self.sentence_idx)

    def __getitem__(self, idx):
        with h5py.File(self.hf_path, 'r') as hf:
            day_n, block_n, sentence_n = self.sentence_idx[idx]
            group = hf[self.partition][day_n][block_n][sentence_n]
            raw_neural_data = np.array(group['neural_data'], dtype=np.float32)
            signal_length = raw_neural_data.shape[0]
            neural_data = self.preprocess_neural_data(raw_neural_data)
            if self.partition == 'holdout':
                return (neural_data, None, None, signal_length)

            raw_text = group.attrs['sentence_text']
            text = self.preprocess_text(raw_text)

            raw_audio = group['audio']

            mel = self.preprocess_audio(raw_audio)

            assert mel.shape[-1] == neural_data.shape[-1], f"Mel signal length {mel.shape[-1]} should be half of the brain signal length {neural_data.shape[-1]} with overall shape {neural_data.shape}"

            return (neural_data.to(self.device), mel.to(self.device), text, signal_length)

    def preprocess_neural_data(self, neural_data):
        # neural data is of shape (T, 2, 2, 5, 8, 8)
        neural_data = torch.tensor(neural_data)
        #neural_data = self._smooth_threshold_crossings(neural_data)
        #neural_data = self._add_noise_to_threshold_crossings(neural_data)
        neural_data = self._normalize_neural_data(neural_data)
        # permute T to the last dimension to match the audio signal
        neural_data = neural_data.permute(1, 2, 3, 4, 5, 0).to(self.device)
        neural_data = pad_or_trim(neural_data, length=N_FRAMES, axis=-1).to(self.device)
        return neural_data

    def _normalize_neural_data(self, neural_data):
        for region in range(2):
            for subregion in range(2):
                neural_data[:, region, subregion, 0, :, :] = self.normalize(neural_data[:, region, subregion, 0, :, :])
                neural_data[:, region, subregion, 1:, :, :] = self.clip(neural_data[:, region, subregion, 1:, :, :])
        return neural_data

    def normalize(self, data):
        return (data - data.mean()) / data.std()

    def clip(self, data):
        return torch.clamp(data, min=0, max=1)

    def _smooth_threshold_crossings(self, neural_data):
        """Applies Exponential Weighted Moving Average smoothing to each channel."""
        grid = [(a, sa, c, y, x) for a in range(2) for sa in range(2) for c in range(2,4) for y in range(8) for x in range(8)]
        for (a, sa, c, y, x) in grid:
            neural_data[:,a,sa,c,y,x]= self._smooth_channel(neural_data[:,a,sa,c,y,x])
        return neural_data

    def _smooth_channel(self, channel, window_size=25):
        alpha = 2 /(window_size + 1)
        b = [alpha]
        a = [1, alpha-1]
        zi = lfiltic(b, a, channel[0:1].numpy(), [0])
        return torch.tensor(lfilter(b, a, channel, zi=zi)[0])

    def _add_noise_to_threshold_crossings(self, neural_data):
        """Add a small amount of gaussian noise to the threshold crossings."""
        noise_shape = list(neural_data.shape)
        noise_shape[3] = noise_shape[3] - 1
        noise_shape = tuple(noise_shape)
        neural_data[:, :, :, 1:, :, :] += 0.2 * torch.randn(noise_shape)
        return neural_data

    def preprocess_audio(self, audio):
        audio = torch.tensor(audio)
        # TODO: possibly trim silence before and after the audio signal if training is not going well
        audio = pad_or_trim(audio.flatten()).to(self.device)
        mel = log_mel_spectrogram(audio)
        return mel

    def preprocess_text(self, text):
        # clean text
        # add start and end tokens
        return text











