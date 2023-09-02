from django.urls import path

from .api_views import *

urlpatterns = [
    path('get-transcript', TranscriptAPIView.as_view(), name='transcript'),
    path('classify-sound', SoundClassificationAPIView.as_view(), name='sound_classification'),
    path('pronounciation', PronounciationAPIView.as_view(), name='pronounciation')
]