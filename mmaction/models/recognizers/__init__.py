from .audio_recognizer import AudioRecognizer
from .base import BaseRecognizer
from .recognizer2d import Recognizer2D
from .recognizer3d import Recognizer3D
from .recognizer_co_teaching import RecognizerCo
from .recognizer_self_training import RecognizerSelfTraining
__all__ = ['BaseRecognizer', 'Recognizer2D', 'Recognizer3D', 'AudioRecognizer',"RecognizerCo", 'RecognizerSelfTraining']
