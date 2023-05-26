import os
import traceback
import streamlit as st
import speech_recognition as sr
from transformers import pipeline
import streamlit.components.v1 as components

# Set Streamlit app layout to wide
st.set_page_config(layout="wide")

# Define Streamlit app sidebar
st.sidebar.title("Settings")
st.sidebar.markdown("Upload an audio file for analysis:")
audio_file = st.sidebar.file_uploader("Upload", type=["wav"])

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

def main():
    # Perform analysis when audio file is uploaded
    if audio_file:
        try:
            # Perform audio transcription
            transcribed_text = transcribe_audio(audio_file)

            # Perform sentiment analysis
            sentiment_label, sentiment_score = perform_sentiment_analysis(transcribed_text)

            # Display the results
            st.header("Transcribed Text")
            st.text_area("Transcribed Text", transcribed_text, height=200)

            st.header("Sentiment Analysis")

            # Display sentiment labels with icons and scores
            negative_icon = "👎"
            neutral_icon = "😐"
            positive_icon = "👍"

            if sentiment_label == "NEGATIVE":
                st.write(f"{negative_icon} Negative (Score: {sentiment_score})")
            else:
                st.empty()

            if sentiment_label == "NEUTRAL":
                st.write(f"{neutral_icon} Neutral (Score: {sentiment_score})")
            else:
                st.empty()

            if sentiment_label == "POSITIVE":
                st.write(f"{positive_icon} Positive (Score: {sentiment_score})")
            else:
                st.empty()

            # Provide additional information about sentiment score interpretation
            st.info("The sentiment score measures how strongly positive, negative, or neutral the feelings or opinions are. "
                    "A higher score indicates a stronger sentiment, while a lower score indicates a weaker sentiment.")

        except Exception as ex:
            st.error("Error occurred during audio transcription and sentiment analysis.")
            st.error(str(ex))
            traceback.print_exc()

if __name__ == "__main__":
    main()

