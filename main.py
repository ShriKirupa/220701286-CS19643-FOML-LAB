import streamlit as st
import time
from loader import load_pdf, load_text, load_url, load_yt_transcript
from vector_store import VectorStore
from ollama_chat import call_deepseek
import uuid
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound
import re

st.set_page_config("NotebookLM Clone", layout="wide")
st.title("🧠 SmartBuddy")

vs = VectorStore()

# --- Session State Init ---
if "sources" not in st.session_state:
    st.session_state.sources = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar Upload Panel ---
st.sidebar.header("📥 Upload Knowledge")
uploaded_file = st.sidebar.file_uploader("Upload File (PDF/TXT)", type=["pdf", "txt", "docx", "csv"])
url_input = st.sidebar.text_input("Enter a URL or YouTube Link")
text_input = st.sidebar.text_area("Paste raw text here")

if st.sidebar.button("➕ Add to Knowledge Base"):
    text_data, source_name = "", ""

    if uploaded_file:
        text_data = load_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else load_text(uploaded_file)
        source_name = uploaded_file.name

    elif url_input:
        if "youtube.com" in url_input or "youtu.be" in url_input:
            text_data = load_yt_transcript(url_input)
            source_name = f"YouTube: {url_input}"
        else:
            text_data = load_url(url_input)
            source_name = f"URL: {url_input}"

    elif text_input:
        text_data = text_input
        source_name = "Raw Text Input"

    if text_data:
        chunks = [text_data[i:i+500] for i in range(0, len(text_data), 450)]
        source_id = str(uuid.uuid4())
        vs.add_texts(chunks, source_id)

        st.session_state.sources.append({
            "id": source_id,
            "name": source_name,
            "checked": True
        })

        st.sidebar.success("✅ Added to Knowledge Base!")
    else:
        st.sidebar.warning("⚠️ Could not load any content.")

# --- Sidebar: List Existing Sources ---
st.sidebar.markdown("### 📚 Your Knowledge Base")
for source in st.session_state.sources:
    source["checked"] = st.sidebar.checkbox(
        source["name"],
        value=source["checked"],
        key=f"checkbox_{source['name']}_{source.get('id', '')}"
    )

# --- Chat UI Begins Here ---
st.subheader("💬 Chat With Your Knowledge Base")

if st.session_state.sources:
    user_input = st.chat_input("Ask something...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append(("user", user_input))

        allowed_ids = [src["id"] for src in st.session_state.sources if src["checked"]]
        if not allowed_ids:
            st.warning("⚠️ Please select at least one knowledge source.")
        else:
            with st.spinner("🔍 Retrieving relevant information..."):
                retrieved_chunks = vs.query(user_input, k=7, allowed_sources=allowed_ids)

            context = "\n".join(retrieved_chunks[:3])
            prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {user_input}\nAnswer:"

            with st.expander("🔍 Show Retrieved Context"):
                st.markdown(context)

            with st.spinner("🧠 DeepSeek is thinking..."):
                response = call_deepseek(prompt)

            full_response = response.get("full", "").strip()
            code_response = response.get("code")
            code_response = code_response.strip() if code_response else ""

            # Extract <think> content using regex
            think_match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
            think_text = think_match.group(1).strip() if think_match else ""

            # Remove the <think> block from full_response
            full_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()

            # Format <think> content in markdown
            formatted_think = ""
            if think_text:
                formatted_think = f"> 💭 **DeepSeek Thinking:**\n>\n> " + "\n> ".join(think_text.splitlines())

            # Add code block if available
            formatted_code = ""
            if code_response:
                try:
                    lexer = guess_lexer(code_response)
                    language = lexer.name.lower().split()[0]
                except ClassNotFound:
                    language = ""
                formatted_code = f"\n\n### Code:\n```{language}\n{code_response}\n```"

            # Final full message
            final_message = ""
            if formatted_think:
                final_message += formatted_think + "\n\n"
            final_message += full_response if full_response else "⚠️ No answer returned."
            final_message += formatted_code

            # Save to chat history
            st.session_state.chat_history.append(("assistant", final_message))

            # Show assistant message with typing animation
            with st.chat_message("assistant"):
                placeholder = st.empty()
                animated_text = ""
                for char in final_message:
                    animated_text += char
                    placeholder.markdown(animated_text)
                    time.sleep(0.005)  # Adjust typing speed here
else:
    st.info("📥 Upload a file or enter text/URL to start chatting.")

# --- Optional Clear Button ---
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
