from sp_audio import load_audio
from compare import is_match
from scipy.io.wavfile import read

word = 'bird'
(freq, audio_array) = read('./Bird.wav')

pre_score = int(is_match(word.lower(), audio_array, freq))
print('result: ', pre_score)
