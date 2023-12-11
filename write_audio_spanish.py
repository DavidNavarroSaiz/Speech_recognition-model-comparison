import speech_recognition as sr
from dotenv import load_dotenv
import os
import pandas as pd
import time
import json
import openai


load_dotenv()
AZURE_SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY')
AZURE_SPEECH_REGION = os.getenv('AZURE_SPEECH_REGION')
WIT_AI_KEY = os.getenv('WIT_AI_KEY')
WIT_AI_KEY_SPANISH = os.getenv('WIT_AI_KEY_SPANISH')
HOUNDIFY_CLIENT_ID = os.getenv('HOUNDIFY_CLIENT_ID')
HOUNDIFY_CLIENT_KEY = os.getenv('HOUNDIFY_CLIENT_KEY')
OPENAI_API_KEY =str( os.getenv('OPENAI_API_KEY'))

# obtain audio from the microphone
r = sr.Recognizer()
r.pause_threshold = 2
record_duration = 20
noise_adjust_duration = 5

# record audio
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source, duration=noise_adjust_duration)
    print("Say something!")
    audio = r.listen(source, timeout=10)

print("finished listening")

# Generate dynamic file name based on existing files
output_directory = "./speech_recognition/evaluation_models/audio_files/spanish"
existing_files = os.listdir(output_directory)
file_number = len(existing_files) + 1
file_name = f"result_{file_number}.wav"
file_path = os.path.join(output_directory, file_name)
path_csv = "./speech_recognition/evaluation_models/speech_results_spanish.csv"

with open('./speech_recognition/evaluation_models/text_spanish.json') as f:
   reading_text = json.load(f)['text']
   
results = {
    'real_text': reading_text,
    'Whisper_API': '',
    'Whisper Base': '',
    'Whisper Small': '',
    'Wit': '',
    'Azure': '',
    'Google': '',
    'Audio_Path': file_path  # Store the file path in the 'Audio_Path' column
}

# Save the audio file
with open(file_path, "wb") as f:
    f.write(audio.get_wav_data())
    f.close()

# process the recorded audio
with sr.WavFile(file_path) as wav_file:
    audio = r.record(wav_file)
    
audio_file= open(file_path, "rb")

# Measure time for Google speech recognition
time_google_start = time.time()
try:
    results['Google'] = r.recognize_google(audio, language="es-CO")
except sr.UnknownValueError:
    results['Google'] = 'Google Speech Recognition could not understand audio'
except sr.RequestError as e:
    results['Google'] = 'Could not request results from Google Speech Recognition service: {0}'.format(e)
time_google_finish = time.time()
results['time_google'] = round(time_google_finish - time_google_start, 2)

# Measure time for Wit speech recognition
time_wit_start = time.time()
try:
    results['Wit'] = r.recognize_wit(audio, key=WIT_AI_KEY_SPANISH)
except sr.UnknownValueError:
    results['Wit'] = 'Wit.ai could not understand audio'
except sr.RequestError as e:
    results['Wit'] = 'Could not request results from Wit.ai service: {0}'.format(e)
time_wit_finish = time.time()
results['time_wit'] = round(time_wit_finish - time_wit_start, 2)

# Measure time for Whisper Base speech recognition
time_whisper_base_start = time.time()
try:
    results['Whisper Base'] = r.recognize_whisper(audio, language='spanish')
except sr.UnknownValueError:
    results['Whisper Base'] = 'Whisper could not understand audio'
except sr.RequestError as e:
    results['Whisper Base'] = 'Whisper error: {0}'.format(e)
time_whisper_base_finish = time.time()
results['time_whisper_base'] = round(time_whisper_base_finish - time_whisper_base_start, 2)

# Measure time for Whisper Small speech recognition
time_whisper_small_start = time.time()
try:
    results['Whisper Small'] = r.recognize_whisper(audio, language='spanish', model='small')
except sr.UnknownValueError:
    results['Whisper Small'] = 'Whisper could not understand audio'
except sr.RequestError as e:
    results['Whisper Small'] = 'Whisper error: {0}'.format(e)
time_whisper_small_finish = time.time()
results['time_whisper_small'] = round(time_whisper_small_finish - time_whisper_small_start, 2)

# Measure time for Azure speech recognition
time_azure_start = time.time()
try:
    results['Azure'] = str(r.recognize_azure(audio, key=AZURE_SPEECH_KEY, language='es-ES', location=AZURE_SPEECH_REGION)[0])
except sr.UnknownValueError:
    results['Azure'] = 'Microsoft Azure Speech could not understand audio'
except sr.RequestError as e:
    results['Azure'] = 'Could not request results from Microsoft Azure Speech service: {0}'.format(e)
time_azure_finish = time.time()
results['time_azure'] = round(time_azure_finish - time_azure_start, 2)


# Measure time for Whisper Base speech recognition
time_whisper_base_start = time.time()
try:
    model_id = 'whisper-1'


    response = openai.Audio.transcribe(
        model=model_id,
        file=audio_file,
        language="es"   
    )
    results['Whisper_API'] = response['text']
except sr.UnknownValueError:
    results['Whisper_API'] = 'Whisper could not understand audio'
except sr.RequestError as e:
    results['Whisper_API'] = 'Whisper error: {0}'.format(e)
time_whisper_base_finish = time.time()
results['time_Whisper_API'] = round(time_whisper_base_finish - time_whisper_base_start, 2)

# Measure time for Whisper Base speech recognition
time_whisper_base_start = time.time()
try:
    model_id = 'whisper-1'


    response = openai.Audio.transcribe(
        model=model_id,
        file=audio_file,
        language="es"   
    )
    results['Whisper_API'] = response['text']
except sr.UnknownValueError:
    results['Whisper_API'] = 'Whisper could not understand audio'
except sr.RequestError as e:
    results['Whisper_API'] = 'Whisper error: {0}'.format(e)
time_whisper_base_finish = time.time()
results['time_Whisper_API'] = round(time_whisper_base_finish - time_whisper_base_start, 2)


# Create a DataFrame with the new result
df_new = pd.DataFrame([results])

# Check if the CSV file already exists
if os.path.isfile(path_csv):
    # Read the existing CSV file
    df_existing = pd.read_csv(path_csv)

    # Append the new row to the existing DataFrame
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
else:
    # If the CSV file does not exist, use the new DataFrame as is
    df_combined = df_new

# Save the combined DataFrame to the CSV file, overwriting the existing file
df_combined.to_csv(path_csv, index=False)
print('Saved')
