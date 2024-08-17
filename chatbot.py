import streamlit as st
from transformers import pipeline
import whisper
import json

# Load the models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
psych_summarizer = pipeline("summarization", model="google/flan-t5-large")

# Initialize Whisper model for transcription
whisper_model = whisper.load_model("medium")

# Helper function to summarize text
def summarize_text(text, max_length=130, min_length=30):
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

# Helper function to transcribe and summarize audio
def transcribe_and_summarize_audio(audio_file):
    result = whisper_model.transcribe(audio_file)
    transcribed_text = result['text'].strip()
    if len(transcribed_text.split()) > 5:
        input_length = len(transcribed_text.split())
        max_length = max(20, int(min(0.5 * input_length, 100)))
        min_length = max(5, int(0.3 * max_length))
        summary = summarize_text(transcribed_text, max_length, min_length)
        return transcribed_text, summary
    else:
        return transcribed_text, "Text is too short to summarize meaningfully."

# Helper function to summarize psychological profile
def summarize_psych_profile(profile_text):
    return psych_summarizer(profile_text, max_length=50, min_length=20, repetition_penalty=2.0, do_sample=True, top_k=50)[0]['summary_text']

# Helper function to summarize map data
def summarize_map_data(map_data):
    locations = map_data["locations"]
    routes = map_data["routes"]

    location_summary = "\n".join([f"- {loc['name']} ({loc['type']}) at coordinates {loc['coordinates']}" for loc in locations])
    route_summary = "\n".join([f"- From {route['start']} to {route['end']}, distance: {route['distance_km']} km" for route in routes])

    summary_text = f"""
    Map Data Summary:
    Locations:
    {location_summary}

    Routes:
    {route_summary}
    """
    return summarize_text(summary_text, max_length=100, min_length=50)

# Streamlit app layout
st.title("Summarization and Transcription Tool")

option = st.sidebar.selectbox(
    "Choose an option",
    ("Text Summarization", "Audio Transcription and Summarization", "Psychological Profile Summarization", "Map Data Summarization")
)

if option == "Text Summarization":
    text_input = st.text_area("Enter text for summarization:")
    if st.button("Summarize"):
        if text_input:
            summary = summarize_text(text_input)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.write("Please enter some text.")

elif option == "Audio Transcription and Summarization":
    audio_file = st.file_uploader("Upload an audio file (mp3 format):", type=["mp3"])
    if st.button("Transcribe and Summarize"):
        if audio_file:
            transcribed_text, summary = transcribe_and_summarize_audio(audio_file)
            st.subheader("Transcribed Text:")
            st.write(transcribed_text)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.write("Please upload a valid audio file.")

elif option == "Psychological Profile Summarization":
    profile_input = st.text_area("Enter psychological profile data:")
    if st.button("Summarize"):
        if profile_input:
            summary = summarize_psych_profile(profile_input)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.write("Please enter psychological profile data.")

elif option == "Map Data Summarization":
    map_data_input = st.text_area("Enter map data in JSON format (e.g., {'locations': [...], 'routes': [...]})")
    if st.button("Summarize"):
        try:
            map_data = json.loads(map_data_input)
            summary = summarize_map_data(map_data)
            st.subheader("Summary:")
            st.write(summary)
        except json.JSONDecodeError:
            st.write("Invalid JSON format. Please check your input.")
        except Exception as e:
            st.write(f"An error occurred: {e}")
