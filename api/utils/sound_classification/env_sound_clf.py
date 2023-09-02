import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

import os

yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)

labels_dict = {
    0: "dog",
    14: "chirping_birds",
    36: "vacuum_cleaner",
    19: "thunderstorm",
    30: "door_wood_knock",
    34: "can_opening",
    9: "crow",
    22: "clapping",
    48: "fireworks",
    41: "chainsaw",
    47: "airplane",
    31: "mouse_click",
    17: "pouring_water",
    45: "train",
    8: "sheep",
    15: "water_drops",
    46: "church_bells",
    37: "clock_alarm",
    32: "keyboard_typing",
    16: "wind",
    25: "footsteps",
    4: "frog",
    3: "cow",
    27: "brushing_teeth",
    43: "car_horn",
    12: "crackling_fire",
    40: "helicopter",
    29: "drinking_sipping",
    10: "rain",
    7: "insects",
    26: "laughing",
    6: "hen",
    44: "engine",
    23: "breathing",
    20: "crying_baby",
    49: "hand_saw",
    24: "coughing",
    39: "glass_breaking",
    28: "snoring",
    18: "toilet_flush",
    2: "pig",
    35: "washing_machine",
    38: "clock_tick",
    21: "sneezing",
    1: "rooster",
    11: "sea_waves",
    42: "siren",
    5: "cat",
    33: "door_wood_creaks",
    13: "crickets",
}


@tf.function
def load_wav_16k_mono(filename):
    """Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio."""

    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def extract_embedding_inference(wav_data):
    """run YAMNet to extract embedding from the wav data"""

    # Extract embeddings
    _, embeddings, _ = yamnet_model(wav_data)

    return embeddings


def classify_env_sound(audio_path: str) -> str:
    """
    Classifies the environment sound present in the audio denoted by the given audio path

    Args:
        audio_path: Path of the audio file. Must be a path to a '.wav' file.

    Returns:
        Prediction as a string
    """

    # Validate provided audio path
    assert os.path.exists(audio_path) and audio_path.endswith(
        ".wav"
    ), "Provided audio path is not a valid .wav file."

    # Load audio file
    audio = load_wav_16k_mono(audio_path)

    # Extract embeddings for the audio file
    audio = extract_embedding_inference(audio)

    # Load model
    model = tf.keras.models.load_model("api/utils/sound_classification/model/env_sound_clf_model")

    # Take predictions
    preds = model.predict(audio)

    return labels_dict[np.argmax(preds[0])]
