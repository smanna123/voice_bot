import pyaudio
import wave
import os

def record_audio(duration, output_filename):
    # Set up the directory
    directory = "user_voice"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Set up parameters
    FORMAT = pyaudio.paInt16  # 16-bit resolution
    CHANNELS = 1              # 1 channel for mono audio
    RATE = 16000              # Sample rate (adjusted from 44100 to 16000)
    CHUNK = 1024              # 1024 samples per chunk
    RECORD_SECONDS = duration # Duration of recording
    WAVE_OUTPUT_FILENAME = os.path.join(directory, output_filename)

    # Initialize pyaudio
    audio = pyaudio.PyAudio()

    # Open a stream
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    # Initialize array to store frames
    frames = []

    # Record data
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a WAV file
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved as {WAVE_OUTPUT_FILENAME}")

# Example usage
record_audio(3, "output.wav")
