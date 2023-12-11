# Speech Recognition Model Comparison

## Overview

The **Speech Recognition Model Comparison** repository is dedicated to comparing the performance of various speech recognition models for both English and Spanish languages. The models included in this project are:

### English Models:
- Google
- Houndify
- Sphinx
- Wit
- Azure
- Whisper API
- Whisper Base
- Whisper Small

### Spanish Models:
- Google
- Wit
- Whisper
- Azure
- Whisper API
- Whisper Base
- Whisper Small

## Project Structure

- `speech_recognition/evaluation_models/`:
  - `text_english.json`: Contains the English text to be spoken during evaluation.
  - `text_spanish.json`: Contains the Spanish text for evaluation.
  - `csv_english.py`: Script to perform speech recognition evaluation for English. Generates audio files in the `audio_files/english` directory.
  - `csv_spanish.py`: Script to perform speech recognition evaluation for Spanish. Generates audio files in the `audio_files/spanish` directory.
  - `calculate_error.py`: Script to calculate the error for each model based on the generated CSV files.

## How to Use

1. Set up environment variables for API keys:
   - `AZURE_SPEECH_KEY`
   - `AZURE_SPEECH_REGION`
   - `WIT_AI_KEY`
   - `WIT_AI_KEY_SPANISH`
   - `HOUNDIFY_CLIENT_ID`
   - `HOUNDIFY_CLIENT_KEY`
   - `OPENAI_API_KEY`

2. Run `csv_english.py` or `csv_spanish.py`:
   - This script records audio from the microphone, performs speech recognition using various models, and stores the results in a CSV file.

3. Use the generated CSV files for analysis:
   - The results for each model, along with the time taken for recognition, are stored in the respective CSV files (`speech_results_english.csv` or `speech_results_spanish.csv`).

4. Run `calculate_error.py`:
   - This script calculates the error for each model based on the generated CSV files.

##  Dependencies

SpeechRecognition
python-dotenv
pandas
openai
Levenshtein

Install the dependencies using:


`pip install -r requirements.txt`


## Audio Files
Audio files recorded during evaluation are stored in the following directories:
    
English: audio_files/english/
Spanish: audio_files/spanish/