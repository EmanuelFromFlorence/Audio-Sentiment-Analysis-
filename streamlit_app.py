import os
import traceback
import speech_recognition as sr
from transformers import pipeline
import streamlit as st
import gradio as gr

def perform_sentiment_analysis(text):
    # Load the sentiment analysis model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_analysis = pipeline("sentiment-analysis", model=model_name)

    # Perform sentiment analysis on the text
    results = sentiment_analysis(text)

    # Extract the sentiment label and score
    sentiment_label = results[0]['label']
    sentiment_score = results[0]['score']

    return sentiment_label, sentiment_score

def transcribe_audio(audio_file):
    r = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)  # Read the entire audio file

    transcribed_text = r.recognize_google(audio)  # Perform speech recognition

    return transcribed_text

def AudioTranscribeSentiment(audio_file):
    try:
        # Perform audio transcription
        transcribed_text = transcribe_audio(audio_file)

        # Perform sentiment analysis
        sentiment_label, sentiment_score = perform_sentiment_analysis(transcribed_text)

        return transcribed_text, sentiment_label, sentiment_score

    except Exception as ex:
        traceback.print_exc()
        return None, None, None  # Return None values if there is an error

def audio_file_processing(audio_file):
    # Replace `audio_file_path` with the path to your audio file
    transcribed_text, sentiment_label, sentiment_score = AudioTranscribeSentiment(audio_file)
    
    if transcribed_text is not None:
        print("Transcribed Text:", transcribed_text)
        print("Sentiment Label:", sentiment_label)
        print("Sentiment Score:", sentiment_score)
    else:
        print("Error occurred during audio transcription and sentiment analysis.")

# Replace `audio_file_path` with the path to your audio file
audio_file_path = "/content/harvard.wav"

audio_file_processing(audio_file_path)





def perform_sentiment_analysis(text):
    # Load the sentiment analysis model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_analysis = pipeline("sentiment-analysis", model=model_name)

    # Perform sentiment analysis on the text
    results = sentiment_analysis(text)

    # Extract the sentiment label and score
    sentiment_label = results[0]['label']
    sentiment_score = results[0]['score']

    return sentiment_label, sentiment_score

def transcribe_audio(audio_file, language):
    r = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)  # Read the entire audio file

    transcribed_text = r.recognize_google(audio, language=language)  # Perform speech recognition

    return transcribed_text

def AudioTranscribeSentiment(audio_file, language):
    try:
        # Perform audio transcription
        transcribed_text = transcribe_audio(audio_file, language)

        # Perform sentiment analysis
        sentiment_label, sentiment_score = perform_sentiment_analysis(transcribed_text)

        return transcribed_text, sentiment_label, sentiment_score

    except Exception as ex:
        traceback.print_exc()
        raise st.StreamlitAPIException("There is some issue with audio transcription and sentiment analysis.")

def main():
    # Set Streamlit app layout to wide
    st.set_page_config(layout="wide")

    # Define Streamlit app sidebar
    st.sidebar.title("Settings")
    st.sidebar.markdown("Upload an audio file for analysis:")
    audio_file = st.sidebar.file_uploader("Upload", type=["wav"])
    language = st.sidebar.selectbox("Select Language", ["en-US", "en-GB"])

    # Perform analysis when audio file is uploaded
    if audio_file:
        # Perform audio transcription and sentiment analysis
        transcribed_text, sentiment_label, sentiment_score = AudioTranscribeSentiment(audio_file, language)

        # Display the results
        st.header("Transcribed Text")
        st.text_area("Transcribed Text", transcribed_text, height=200)

        st.header("Sentiment Analysis")
        st.write("Sentiment Label:", sentiment_label)
        st.write("Sentiment Score:", sentiment_score)

if __name__ == "__main__":
    main()
