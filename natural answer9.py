

import os
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech

# Audio Transcription
def transcribe_audio(audio_file_path):
    client = speech.SpeechClient()

    # Read the audio file
    with open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US"  # Adjust language code as needed
    )

    response = client.recognize(config=config, audio=audio)
    transcription = ""
    for result in response.results:
        transcription += result.alternatives[0].transcript

    return transcription

# Text-to-Speech Conversion
def convert_text_to_speech(text, language_code, output_file_path):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,  # Specify desired language code
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open(output_file_path, "wb") as output_file:
        output_file.write(response.audio_content)
        print(f'Audio content written to file "{output_file_path}"')

# Provide the path to the audio file
audio_file_path = "path_to_audio_file.wav"

# Transcribe the audio file
transcription = transcribe_audio(audio_file_path)

# Convert the transcription to a different language
target_language_code = "es"  # Specify the desired language code for the translation

# Specify the output file path for the translated audio
output_file_path = "translated_audio.mp3"

# Convert the transcription to speech in the target language
convert_text_to_speech(transcription, target_language_code, output_file_path)
