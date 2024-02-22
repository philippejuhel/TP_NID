import pyaudio
import pvporcupine
import struct

# Recognize YOUR wakeword

access_key='PUT YOUR ACCESS KEY HERE'
keyword_paths=['Salut-machin_fr_raspberry-pi_v3_0_0.ppn']
model_path='porcupine_params_fr.pv' # To recognize French language

# Create an Porcupine instance to process the sound and recognize words
porcupine = pvporcupine.create(
  access_key=access_key,
  keyword_paths=keyword_paths, # The wakeword described by in this file will be recognized
  model_path=model_path
)

# Define the audio interface to record sound
py_audio = pyaudio.PyAudio()
audio_stream = py_audio.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length)

while True:
    pcm = audio_stream.read(porcupine.frame_length)  # Read an audio sample
    audio_frame = struct.unpack_from("h" * porcupine.frame_length, pcm)
    result = porcupine.process(audio_frame)  # and ask Porcupine to try to identify the wakeword
    if result > -1:  # The wakeword has been detected
        print(f"Wakeword detected") # So we print this.
