from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from scipy.io.wavfile import read

from rest_framework import status
import os

from api.utils.transcript.functions import transcribe_video
from api.utils.sound_classification.env_sound_clf import classify_env_sound
from api.utils.pronounciation_eval.compare import is_match


class TranscriptAPIView(APIView):
    """ Handles the logic for transcript generator """

    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            if 'audio' in request.FILES:
                file_obj = request.FILES['audio']
                temp_file_extension = 'wav'  # Change this to the appropriate audio format
            elif 'video' in request.FILES:
                file_obj = request.FILES['video']
                temp_file_extension = 'mp4'  # Change this to the appropriate video format
            else:
                return Response({'error': 'No audio or video file uploaded'}, status=status.HTTP_400_BAD_REQUEST)
        except :
            return Response({'error': 'something is wrong with the uploaded file'}, status=status.HTTP_400_BAD_REQUEST)
        # Save the uploaded image to a temporary file
        temp_video_file = default_storage.save(f'temp_file.{temp_file_extension}', ContentFile(file_obj.read()))

        video_path = os.path.join(settings.MEDIA_ROOT, temp_video_file)
        print('here')
        transcript = transcribe_video(path=video_path, type=temp_file_extension)

        context = {
            'transcript': transcript
        }

        return Response(context)


class SoundClassificationAPIView(APIView):
    """ Handles the logic for sound classification """

    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            file_obj = request.FILES['audio']
        except Exception as e:
            print(f'ERROR: {e}')
            return Response({'error': 'something is wrong with the uploaded file'}, status=status.HTTP_400_BAD_REQUEST)

        # Save the uploaded image to a temporary file
        temp_audio_file = default_storage.save('temp_audio.wav', ContentFile(file_obj.read()))

        audio_path = os.path.join(settings.MEDIA_ROOT, temp_audio_file)

        prediction = classify_env_sound(audio_path=audio_path)
        
        context = {
            'prediction': prediction
        }
        
        return Response(context)


class PronounciationAPIView(APIView):

    def post(self, request, *args, **kwargs):
        word = request.data['word']
        try:
            file_obj = request.FILES['audio']
        except:
            return Response({'error': 'something is wrong with the uploaded file'}, status=status.HTTP_400_BAD_REQUEST)

        # Save the uploaded image to a temporary file
        temp_audio_file = default_storage.save('temp_audio.wav', ContentFile(file_obj.read()))

        audio_path = os.path.join(settings.MEDIA_ROOT, temp_audio_file)
        (freq, audio_array) = read(audio_path)
        pre_score = int(is_match(word.lower(), audio_array, freq))

        context = {
            'score': pre_score
        }
        return Response(context)

