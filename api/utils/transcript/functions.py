import os
from typing import Union
from pathlib import Path

import moviepy.editor as mp
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from resemblyzer import VoiceEncoder
from resemblyzer.audio import preprocess_wav, sampling_rate
from spectralcluster import SpectralClusterer
from scipy.io.wavfile import write, read
import numpy as np
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

TMP_AUDIO_PATH = "preprocessed.wav"


def extract_audio(path, audio_path, save = False):
    # Checking whether paths exist
    assert os.path.exists(path), f"Invalid video path: {path}"
    assert os.path.exists(audio_path), f"Invalid output path: {audio_path}"

    # Extracting audio from video file
    video = mp.VideoFileClip(filename=path)
    audio = video.audio

    # Saving the audio file
    if save:
        audio_fname = os.path.join(
            audio_path, f"{os.path.basename(path).split('.')[-2]}.wav"
        )
        audio.write_audiofile(audio_fname)
        return audio_fname
    else:
        return audio


def preprocess_audio(audio_file_path):
    # Load the audio
    wav_fpath = Path(audio_file_path)
    # Preprocess the audio
    wav = preprocess_wav(wav_fpath)

    # Write preprocessed audio to disk
    write(TMP_AUDIO_PATH, sampling_rate, wav)

    return wav


def label_speakers(labels, wav_splits):
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    labelling = []
    start_time = 0

    for i, time in enumerate(times):
        if i > 0 and labels[i] != labels[i - 1]:
            temp = [str(labels[i - 1]), start_time, time]
            labelling.append(tuple(temp))
            start_time = time
        if i == len(times) - 1:
            temp = [str(labels[i]), start_time, time]
            labelling.append(tuple(temp))

    return labelling


def cluster_speakers(audio):
    """

    Clustering speakers to identify which speaker spoke at specific times in the audio clip
    """

    # Instantiate the VoiceEncoder model and take predictions for embeddings
    encoder = VoiceEncoder("cpu")
    _, cont_embeds, wav_splits = encoder.embed_utterance(
        audio, return_partials=True, rate=16
    )

    # Cluster similar embeddings together using SpectralClusterer
    clusterer = SpectralClusterer(min_clusters=2, max_clusters=100)

    # Generate labels
    labels = clusterer.predict(cont_embeds)

    # Map speaker labels with clip start and end times
    labelling = label_speakers(labels, wav_splits)

    return labelling


def transcribe_video(path, type):
    if type == 'mp4':
        # Extract audio
        audio_path = extract_audio(
            path=path,
            audio_path="..",
            save=True,
        )
    else:
        audio_path = path

    # Preprocess extracted audio
    audio = preprocess_audio(audio_file_path=audio_path)

    # Speaker cluster details
    speaker_clusters = cluster_speakers(audio)

    print(
        f"{len(speaker_clusters)} different speech segments by {len(list(set([i[0] for i in speaker_clusters])))} speakers were detected."
    )

    transcriptions = []

    # Generate transcriptions for each speech segment
    for speaker, st, et in speaker_clusters:
        # Extract sublcip from complete audio file
        ffmpeg_extract_subclip(TMP_AUDIO_PATH, st, et, targetname="segment.wav")

        # Read the audio segment
        rate, segment_audio = read("segment.wav")

        # Instantiate speech-to-text model
        model = Speech2TextForConditionalGeneration.from_pretrained(
            "facebook/s2t-small-librispeech-asr"
        )

        # Instantiate speech-to-text preprocessor
        processor = Speech2TextProcessor.from_pretrained(
            "facebook/s2t-small-librispeech-asr"
        )

        # Preprocess audio for the model
        inputs = processor(segment_audio, sampling_rate=rate, return_tensors="pt")
        generated_ids = model.generate(
            inputs["input_features"], attention_mask=inputs["attention_mask"]
        )

        # Take predictions from speech-to-text model
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)

        transcriptions.append(f"Speaker {speaker}: {transcription[0]}\n")

    # Write transcripts to file
    with open("transcript.txt", "w") as f:
        f.writelines(transcriptions)

    print("Generated Transcript was successfully saved at 'transcript.txt'.")

    # Delete created temporary files
    os.remove(TMP_AUDIO_PATH)
    os.remove("segment.wav")
    os.remove(audio_path)

    return transcriptions
