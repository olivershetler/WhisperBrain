# WhisperBrain

WhisperBrain is an extension to the OpenAI Whisper model designed to for use with Speech Brain Computer Interfaces. 

# Setup

Clone the repository and use `pip install -e .` to create an editable installation.

See [WhisperBrainBenchmarkAudio.ipynb](https://github.com/olivershetler/WhisperBrain/blob/main/notebooks/WhisperBrainBenchmarkAudio.ipynb) for a demonstration on how to use the audio decoder features of the WhisperBrain model.

Go to [WhisperBrainDataPrep.ipynb](https://github.com/olivershetler/WhisperBrain/blob/main/notebooks/WhisperBrainDataPrep.ipynb) for the script that converts the [data](https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq) from Willett et al. (2023) into an HDF5 file that is used by the dataset objects in this repository. Note that you will need to rename some folders and remove some bad .mat files for the script to run from start to finish.

Go to 
