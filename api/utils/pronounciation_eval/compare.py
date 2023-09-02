from api.utils.pronounciation_eval.sp_audio import diff as diff
from api.utils.pronounciation_eval.sp_audio import diff_base as diff_base
import os


def is_match(audio_label, audio_arr, audio_freq, threshold_100 = 100, threshold_90 = 500, threshold_70 = 1000, threshold_30 = 10000, threshold_0 = 50000):
    """
    
    Check whether the provided audio matches the pronunciation of the word
    :param audio_label: Word
    :param audio_arr: Audio
    :param audio_freq: Frequency of the audio
    :param threshold_90: Threshold for 90% match
    :param threshold_70: Threshold for 70% match
    :param threshold_30: Threshold for 30% match
    :return: True/False - Whether audio matches the pronunciation or not
    """

    # Directory containing base case audio files for words
    base_cases_dir = 'api/utils/pronounciation_eval/voice'

    # List of words in the base cases directory
    base_cases = list(set([word.split('.')[0] for word in os.listdir(base_cases_dir)]))

    # Check whether the word exists in the base cases
    if audio_label in base_cases:
        # Compare audio with male base case
        male_base = diff_base('{}/{}.wav'.format(base_cases_dir, audio_label), '{}/{}.wav'.format(base_cases_dir, audio_label))
        male_use = diff('{}/{}.wav'.format(base_cases_dir, audio_label), audio_arr, audio_freq)

        male_diff = abs(male_use - male_base)

        # Compare audio with female base case
        # female_base = diff_base('{}/{}_f.wav'.format(base_cases_dir, audio_label), '{}/{}_f.wav'.format(base_cases_dir, audio_label))
        # female_use = diff('{}/{}_m.wav'.format(base_cases_dir, audio_label), audio_arr, audio_freq)
        #
        # female_diff = abs(female_use - female_base)

        score = 0

        import random

        if (male_diff <= threshold_100):
            score = 100
        elif ((threshold_100 < male_diff <= threshold_90)):
            score = random.randint(90, 99)
        elif ((threshold_90 < male_diff <= threshold_70)):
            score = random.randint(70, 89)
        elif ((threshold_0 > male_diff > threshold_30)):
            score = random.randint(50, 69)
        elif ((threshold_70 < male_diff <= threshold_30)):
            score = random.randint(30, 49)
        else:
            score = random.randint(0, 29)

        if 1000 > male_diff > 2000:
            score = 0

        return '{}'.format(score)

    else:
        return 'Word does not exist in the audio directory'
    
