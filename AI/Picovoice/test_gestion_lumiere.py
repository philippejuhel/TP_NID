import pyaudio
from picovoice import Picovoice
import struct

# Switch on/off lights for rooms and change their color.

keyword_paths=['RECOPIEZ_ICI_VOTRE_FICHIER.ppn']  # path to Porcupine wake word file (.PPN)
context_path = 'RECOPIEZ_ICI_VOTRE_FICHIER.rhn'  # path to Rhino context file (.RHN)
porcupine_model_path='porcupine_params_fr.pv' # To recognize French language
rhino_model_path='rhino_params_fr.pv' # To recognize French language
access_key='RECOPIEZ ICI VOTRE ACCESS_KEY'

def wake_word_callback():
    print('Wakeword has been recognized')

def inference_callback(inference):
    print(inference.is_understood)
    if inference.is_understood:
        print(inference.intent)
        for k, v in inference.slots.items():
            print(f"{k} : {v}")

pv = Picovoice(
    access_key=access_key,
    keyword_path=keyword_path,
    porcupine_model_path=porcupine_model_path,
    rhino_model_path=rhino_model_path,
    wake_word_callback=wake_word_callback,
    context_path=context_path,
    inference_callback=inference_callback)

py_audio = pyaudio.PyAudio()
audio_stream = py_audio.open(
    rate=pv.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=pv.frame_length)

while True:
    pcm = audio_stream.read(pv.frame_length)
    audio_frame = struct.unpack_from("h" * pv.frame_length, pcm)
    pv.process(audio_frame)