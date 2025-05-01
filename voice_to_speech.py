import streamlit as st
import speech_recognition as sr 

def save_note_to_file(text):
    """Save the recognized text as a note in a markdown file."""
    with open("notes.md", "a") as file:
        file.write(f"### Note:\n{text}\n\n")

def record_note():
    """Record voice, convert it to text and display in the Streamlit interface."""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    try:
        with mic as source:
            st.info("Say something...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

            # Use Google's Web Speech API for recognizing speech
            text = recognizer.recognize_google(audio)
            st.success(f"Recognized: {text}")

            # Display the recognized text in the Streamlit app
            st.text_area("Your Note", value=text, height=150)

            # Save the note to markdown file
            save_note_to_file(text)

    except sr.UnknownValueError:
        st.error("Sorry, I could not understand your speech.")
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")

# Streamlit interface
st.title("Voice Notes App")
st.write("Click the button below and speak to record your note.")

if st.button("Record Note"):
    record_note()
