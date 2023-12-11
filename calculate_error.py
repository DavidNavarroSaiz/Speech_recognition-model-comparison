import pandas as pd
import Levenshtein

# Function to calculate Word Error Rate (WER)
def wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    distance = Levenshtein.distance(reference, hypothesis)
    wer = float(distance) / len(ref_words)

    return wer

# Function to calculate Character Error Rate (CER)
def cer(reference, hypothesis):
    distance = Levenshtein.distance(reference, hypothesis)
    cer = float(distance) / len(reference)

    return cer

# Function to calculate errors and return averages
def calculate_error(df, model_name):
    # Initialize variables for cumulative WER and CER
    total_wer = 0
    total_cer = 0
    total_time = 0

    # Calculate WER, CER, and time for each row
    for index, row in df.iterrows():
        real_text = row['real_text']
        model_text = row[model_name]
        time_value = row[f'time_{model_name}']

        wer_value = wer(real_text, model_text)
        cer_value = cer(real_text, model_text)

        # Accumulate the values for averaging
        total_wer += wer_value
        total_cer += cer_value
        total_time += time_value

    # Calculate averages
    avg_wer = total_wer / len(df)
    avg_cer = total_cer / len(df)
    avg_time = total_time / len(df)

    return avg_wer, avg_cer, avg_time

# Load the CSV files
df_spanish = pd.read_csv('./speech_results_spanish.csv')
df_english = pd.read_csv('./speech_results_english.csv')

# Calculate and print errors for the Spanish models
print("Spanish Results:")
avg_wer_whisper_spanish, avg_cer_whisper_spanish, avg_time_whisper_spanish = calculate_error(df_spanish, 'Whisper_API')
avg_wer_azure_spanish, avg_cer_azure_spanish, avg_time_azure_spanish = calculate_error(df_spanish, 'Azure')

print(f"\nAverage WER Whisper_API (Spanish): {avg_wer_whisper_spanish:.4f}, Average CER Whisper_API (Spanish): {avg_cer_whisper_spanish:.4f}, Average Time Whisper_API (Spanish): {avg_time_whisper_spanish:.4f}")
print(f"Average WER Azure (Spanish): {avg_wer_azure_spanish:.4f}, Average CER Azure (Spanish): {avg_cer_azure_spanish:.4f}, Average Time Azure (Spanish): {avg_time_azure_spanish:.4f}")

# Calculate and print errors for the English models
print("\nEnglish Results:")
avg_wer_whisper_english, avg_cer_whisper_english, avg_time_whisper_english = calculate_error(df_english, 'Whisper_API')
avg_wer_azure_english, avg_cer_azure_english, avg_time_azure_english = calculate_error(df_english, 'Azure')

print(f"\nAverage WER Whisper_API (English): {avg_wer_whisper_english:.4f}, Average CER Whisper_API (English): {avg_cer_whisper_english:.4f}, Average Time Whisper_API (English): {avg_time_whisper_english:.4f}")
print(f"Average WER Azure (English): {avg_wer_azure_english:.4f}, Average CER Azure (English): {avg_cer_azure_english:.4f}, Average Time Azure (English): {avg_time_azure_english:.4f}")

total_average_azure = (avg_wer_azure_english+ avg_cer_azure_english + avg_wer_azure_spanish+ avg_cer_azure_spanish)/4
total_average_whisper = (avg_wer_whisper_english + avg_cer_whisper_english +avg_wer_whisper_spanish + avg_cer_whisper_spanish)/4

total_average_time_azure = (avg_time_azure_english + avg_time_azure_spanish)/2
total_average__time_whisper = (avg_time_whisper_english + avg_time_whisper_spanish)/2
print(f"\nAzure \nFinal Error : {total_average_azure:.4f} , final average time {total_average_time_azure}")
print(f"\nWhisper\n Final Error : {total_average_whisper:.4f}, final average time:{total_average__time_whisper} ")
