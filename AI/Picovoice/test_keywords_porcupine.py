import pyaudio
import pvporcupine
import struct

# Recognize two words (see keywords)

access_key='PUT YOUR ACCESS KEY HERE'
keywords=['picovoice', 'bumblebee']

#Allowed keywords : nok google, hey google, hey siri, pico clock, hey barista, computer, porcupine, terminator, grapefruit, blueberry, americano, bumblebee, grasshopper, jarvis, alexa, picovoice

# Create an Porcupine instance to process the sound and recognize words
porcupine = pvporcupine.create(
  access_key=access_key,
  keywords=keywords # This two keyworks will be recognized
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
    result = porcupine.process(audio_frame)  # and ask Porcupine to try to identify a keyword
    if result > -1:  # A keyword has been detected
        print(f"keyword detected : {keywords[result]}") # So we print it.
