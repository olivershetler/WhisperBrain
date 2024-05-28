from scipy.io import loadmat
from pathlib import Path
import h5py
import numpy as np
import json
import traceback

#from gtts import gTTS
#import pyttsx3
from google.cloud import texttospeech
import librosa
import numpy as np
import tempfile
import os
from pydub import AudioSegment
import traceback
import re

from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram, SAMPLE_RATE, N_SAMPLES
import torch

NEURAL_DATA_SHRINK_FACTOR = 2


CHANNEL_MAPS = {
    "area_44": {
        "superior": [
            [192, 193, 208, 216, 160, 165, 178, 185],
            [194, 195, 209, 217, 162, 167, 180, 184],
            [196, 197, 211, 218, 164, 170, 177, 189],
            [198, 199, 210, 219, 166, 174, 173, 187],
            [200, 201, 213, 220, 168, 176, 183, 186],
            [202, 203, 212, 221, 172, 175, 182, 191],
            [204, 205, 214, 223, 161, 169, 181, 188],
            [206, 207, 215, 222, 163, 171, 179, 190]
        ],
        "inferior": [
            [129, 144, 150, 158, 224, 232, 239, 255],
            [128, 142, 152, 145, 226, 233, 242, 241],
            [130, 135, 148, 149, 225, 234, 244, 243],
            [131, 138, 141, 151, 227, 235, 246, 245],
            [134, 140, 143, 153, 228, 236, 248, 247],
            [132, 146, 147, 155, 229, 237, 250, 249],
            [133, 137, 154, 157, 230, 238, 252, 251],
            [136, 139, 156, 159, 231, 240, 254, 253]
        ]
    },
    "area_6v": {
        "superior": [
            [62, 51, 43, 35, 94, 87, 79, 78],
            [60, 53, 41, 33, 95, 86, 77, 76],
            [63, 54, 47, 44, 93, 84, 75, 74],
            [58, 55, 48, 40, 92, 85, 73, 72],
            [59, 45, 46, 38, 91, 82, 71, 70],
            [61, 49, 42, 36, 90, 83, 69, 68],
            [56, 52, 39, 34, 89, 81, 67, 66],
            [57, 50, 37, 32, 88, 80, 65, 64]
        ],
        "inferior": [
            [125, 126, 112, 103, 31, 28, 11, 8],
            [123, 124, 110, 102, 29, 26, 9, 5],
            [121, 122, 109, 101, 27, 19, 18, 4],
            [119, 120, 108, 100, 25, 15, 12, 6],
            [117, 118, 107, 99, 23, 13, 10, 3],
            [115, 116, 106, 97, 21, 20, 7, 2],
            [113, 114, 105, 98, 17, 24, 14, 0],
            [127, 111, 104, 96, 30, 22, 16, 1]
        ]
    }
}

def map_arrays(arrays, area, subarea):
    CH = len(arrays)
    T = arrays[0].shape[0]
    new_array = np.zeros((T, CH, 8, 8))
    for a, array in enumerate(arrays):
        channel_map = CHANNEL_MAPS[area][subarea]
        for i in range(8):
            for j in range(8):
                new_array[:, a, i, j] = array[:, channel_map[i][j]]
    return new_array

def load_mat_file(path):
    try:
        mat = loadmat(str(path))
        return mat
    except Exception as e:
        print("=====================================")
        print(f"Error loading {path}")
        print(traceback.format_exc())
        print("=====================================")
        return None

def convert_to_kebab_case(text):
    return re.sub(r'\W+', '-', text.lower()).strip('-')


def generate_speech_with_google(text, speed=1.0, save_path=None, credentials_path=None):
    # Load credentials from the JSON key file
    client = texttospeech.TextToSpeechClient.from_service_account_file(credentials_path)

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=speed
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    with open(save_path, 'wb') as out:
        out.write(response.audio_content)

    return save_path

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def change_playback_speed(audio, target_length_ms):
    # Calculate the current length of the audio in milliseconds
    current_length_ms = len(audio)

    # Print debug information
    print(f"Current length (ms): {current_length_ms}")
    print(f"Target length (ms): {target_length_ms}")
    print(f"Original frame rate: {audio.frame_rate}")

    # Calculate the speed change ratio
    speed_change_ratio = current_length_ms / target_length_ms

    # Change the speed of the audio by modifying the frame rate
    new_frame_rate = int(audio.frame_rate * speed_change_ratio)
    new_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": new_frame_rate
    })

    # Print debug information
    print(f"New frame rate: {new_frame_rate}")

    # Set the frame rate back to the original
    new_audio = new_audio.set_frame_rate(audio.frame_rate)

    # Print final length for verification
    print(f"Adjusted length (ms): {len(new_audio)}")

    return new_audio


def text_to_audio(sentence, session_sample_length, audio_dir, credentials_path):
    try:
        sentence_text = sentence.decode('utf-8').strip()
        kebab_case_sentence = convert_to_kebab_case(sentence_text)
        audio_path = os.path.join(audio_dir, f"{kebab_case_sentence}.wav")
        adjusted_audio_path = os.path.join(audio_dir, f"{kebab_case_sentence}_adjusted.wav")

        # Ensure the directory exists
        ensure_directory_exists(audio_dir)

        # Generate speech at normal speed
        generate_speech_with_google(sentence_text, speed=1.0, save_path=audio_path, credentials_path=credentials_path)

        # Load the baseline audio file using pydub and trim silence
        audio = AudioSegment.from_file(audio_path)
        current_length_ms = len(audio)

        # Calculate the desired audio length in milliseconds
        desired_length_ms = session_sample_length * 20 / NEURAL_DATA_SHRINK_FACTOR  # session_sample_length * 20ms in milliseconds

        # Calculate the speed adjustment factor
        speed_adjustment = current_length_ms / desired_length_ms

        # Generate speech with the adjusted speed
        generate_speech_with_google(sentence_text, speed=speed_adjustment, save_path=adjusted_audio_path, credentials_path=credentials_path)

        # Change playback speed to make the audio match more exactly the desired length
        if current_length_ms != desired_length_ms:
            audio = AudioSegment.from_file(adjusted_audio_path)
            audio = change_playback_speed(audio, desired_length_ms)
            audio.export(adjusted_audio_path, format="wav")

        # Load the adjusted audio file using the Whisper load_audio function
        adjusted_audio = load_audio(adjusted_audio_path)

        return adjusted_audio

    except Exception as e:
        print(f"Error processing text-to-audio-mel for sentence: {sentence_text}")
        print(traceback.format_exc())
        raise e

def process_sentence(h5file, partition, day_n, block_n, sentence_n, date, block_number, sentence_number, sentence, arrays, sentence_start_timestep_idx, sentence_end_timestep_idx, LEN, data_dir):
    sample_group = h5file[partition][day_n][block_n][sentence_n]
    session_sample_length = arrays[0].shape[0]
    sentence_start_timestep_idx.append(LEN)
    sentence_end_timestep_idx.append(LEN + session_sample_length)

    if partition != 'holdout' and sentence != 'HELD OUT':
        audio = text_to_audio(sentence, session_sample_length, audio_dir=data_dir / 'audio', credentials_path=data_dir / 'whisperbrain-50afa82736ec.json')
        sample_group.create_dataset('audio', data=audio, dtype=np.float32)
        sample_group.attrs.update({
            'audio_shape': '(N_SAMPLES,)',
            'audio_dims': json.dumps({
                'N_SAMPLES': 'Number of audio samples'
            })
        })

    areas_data = {}
    for area in ['area_44', 'area_6v']:
        area_data = []
        for subarea in ['superior', 'inferior']:
            array = map_arrays(arrays, area, subarea)
            area_data.append(array)
        areas_data[area] = np.stack(area_data, axis=0)  # Shape: (2, T, CH, 8, 8)
    combined_data = np.stack([areas_data['area_44'], areas_data['area_6v']], axis=0)  # Shape: (2, 2, T, CH, 8, 8)
    combined_data = combined_data.transpose(2, 0, 1, 3, 4, 5)  # Shape: (T, 2, 2, CH, 8, 8)

    sample_group.create_dataset('neural_data', data=combined_data, dtype=np.float32)

    sample_group.attrs.update({
        'date': date,
        'block_number': block_number,
        'sentence_number': sentence_number,
        'sentence_text': sentence,
        'neural_data_shape': '(T, A, SA, CH, X, Y)',
        'neural_data_dims': json.dumps({
            'T': 'Time dimension, representing the number of time steps',
            'A': 'Area dimension, representing the different brain areas (e.g., area_44 and area_6v)',
            'SA': 'Subarea dimension, representing the different subareas within each brain area (e.g., superior and inferior)',
            'CH': 'Channel dimension, representing the number of channels (e.g., different types of neural signals)',
            'X': 'Spatial X dimension, representing the X-coordinate in the 8x8 grid of electrodes',
            'Y': 'Spatial Y dimension, representing the Y-coordinate in the 8x8 grid of electrodes'
        })
    })

    return LEN + session_sample_length

def process_block(h5file, partition, day_n, date, mat, block_start_timestep_idx, LEN, sentence_start_timestep_idx, sentence_end_timestep_idx, sentence_count, data_dir):
    prev_block_num = -1
    sentence_number = 0
    for i, sentence in enumerate(mat['sentenceText']):
        sentence = sentence.encode('utf-8')
        block_number = mat['blockIdx'][i, 0]
        block_n = str(block_number)
        if block_n not in h5file[partition][day_n]:
            block_start_timestep_idx.append(LEN)
            h5file[partition][day_n].create_group(block_n)
            h5file[partition][day_n][block_n].attrs.update({'block_number': block_number, 'date': date})
        if block_number != prev_block_num:
            sentence_number = 0
            prev_block_num = block_number
        else:
            sentence_number += 1
        sentence_n = str(sentence_number)
        h5file[partition][day_n][block_n].create_group(sentence_n)
        h5file[partition][day_n][block_n][sentence_n].attrs.update({'sentence_number': sentence_number, 'sentence_text': sentence, 'block_number': block_number, 'date': date})
        spike_pow = mat['spikePow'][0, i]
        tx1 = mat['tx1'][0, i]
        tx2 = mat['tx2'][0, i]
        tx3 = mat['tx3'][0, i]
        tx4 = mat['tx4'][0, i]
        arrays = [spike_pow, tx1, tx2, tx3, tx4]
        LEN = process_sentence(h5file, partition, day_n, block_n, sentence_n, date, block_number, sentence_number, sentence, arrays, sentence_start_timestep_idx, sentence_end_timestep_idx, LEN, data_dir)
        sentence_count += 1
    return LEN, sentence_count

def process_partition(h5file, partition, partition_dir, day_start_timestep_idx, block_start_timestep_idx, LEN, sentence_count, sentence_start_timestep_idx, sentence_end_timestep_idx, data_dir):
    partition_paths = partition_dir.glob('*.mat')
    bad_days = []
    for p, path in enumerate(sorted(partition_paths)):
        print(f"Processing {path}")
        day_n = str(p)
        session = path.stem
        year, month, day = session.split('.')[1:]
        date = f'{year}-{month}-{day}'
        mat = load_mat_file(path)
        if mat is None:
            bad_days.append(date)
            continue
        h5file[partition].create_group(day_n)
        h5file[partition][day_n].attrs['date'] = date
        day_start_timestep_idx.append(LEN)
        LEN, sentence_count = process_block(h5file, partition, day_n, date, mat, block_start_timestep_idx, LEN, sentence_start_timestep_idx, sentence_end_timestep_idx, sentence_count, data_dir)
    return LEN, sentence_count, bad_days

def write_hdf5_metadata(h5file, partition, sentence_count, LEN, sentence_start_timestep_idx, sentence_end_timestep_idx, block_start_timestep_idx, day_start_timestep_idx, bad_days, MAX_SENTENCE_LENGTH):
    partition_group = h5file[partition]
    partition_group.attrs.update({
        'n_days': len(partition_group.keys()),
        'n_sentences': sentence_count,
        'n_time_steps': LEN,
        'max_sentence_length': MAX_SENTENCE_LENGTH,
        'sentence_start_timestep_idx': json.dumps(sentence_start_timestep_idx),
        'block_start_timestep_idx': json.dumps(block_start_timestep_idx),
        'day_start_timestep_idx': json.dumps(day_start_timestep_idx),
        'bad_days': json.dumps(bad_days)
    })

def mat_to_hdf5(data_dir):
    data_dir = Path(data_dir)
    h5_path = data_dir / 'Willett&EtAl2023.h5'
    if h5_path.exists():
        raise FileExistsError(f"{h5_path} already exists. Please delete it before running this script.")
    sentence_start_timestep_idx = []
    sentence_end_timestep_idx = []
    block_start_timestep_idx = []
    day_start_timestep_idx = []
    try:
        with h5py.File(h5_path, 'w') as h5file:
            partitions = {'train': data_dir / 'train', 'test': data_dir / 'test', 'holdout': data_dir / 'holdout'}
            sentence_count = 0
            LEN = 0
            MAX_SENTENCE_LENGTH = 0
            for partition, partition_dir in partitions.items():
                h5file.create_group(partition)
                LEN, sentence_count, bad_days = process_partition(h5file, partition, partition_dir, day_start_timestep_idx, block_start_timestep_idx, LEN, sentence_count, sentence_start_timestep_idx, sentence_end_timestep_idx, data_dir)
                write_hdf5_metadata(h5file, partition, sentence_count, LEN, sentence_start_timestep_idx, sentence_end_timestep_idx, block_start_timestep_idx, day_start_timestep_idx, bad_days, MAX_SENTENCE_LENGTH)
            assert sentence_count == len(sentence_start_timestep_idx) == len(sentence_end_timestep_idx), f"Error: Metadata not of equal length\n------\nsentence_count: {sentence_count},\nlen(sentence_start_timestep_idx): {len(sentence_start_timestep_idx)},\nlen(sentence_end_timestep_idx): {len(sentence_end_timestep_idx)}"
    except Exception as e:
        h5_path.unlink()
        raise e


if __name__ == "__main__":
    mat_to_hdf5("K:\ke\sta\data\Willett&EtAl2023\data")