from bs4 import BeautifulSoup
import requests
from PyPDF2 import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, VideoUnavailable

def load_pdf(file):
    pdf = PdfReader(file)
    return "\n".join([page.extract_text() for page in pdf.pages])

def load_text(file):
    return file.read().decode("utf-8")

def load_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()

def load_yt_transcript(url):
    try:
        video_id = url.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([i["text"] for i in transcript])
    except TranscriptsDisabled:
        return "[Error: Transcripts are disabled for this video]"
    except VideoUnavailable:
        return "[Error: Video is unavailable]"
    except Exception as e:
        return f"[Error loading transcript] {str(e)}"
