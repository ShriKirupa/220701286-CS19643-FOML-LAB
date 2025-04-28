import streamlit as st
import uuid
from loader import load_pdf, load_text, load_url, load_yt_transcript
from vector_store import VectorStore
from ollama_chat import call_deepseek
import re

st.set_page_config("NotebookLM Clone", layout="wide")
st.title("🧠 SmartBuddy")

vs = VectorStore()

# --- Session State Init ---
if "sources" not in st.session_state:
    st.session_state.sources = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

if "quiz_question" not in st.session_state:
    st.session_state.quiz_question = ""

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
        st.session_state["user_input"] = user_input

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    user_input = st.session_state["user_input"]
    allowed_ids = [src["id"] for src in st.session_state.sources if src.get("checked", False)]

    if user_input and user_input.strip():
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append(("user", user_input))

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

            think_match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
            think_text = think_match.group(1).strip() if think_match else ""

            full_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()

            formatted_think = ""
            if think_text:
                formatted_think = f"> 💭 **DeepSeek Thinking:**\n>\n> " + "\n> ".join(think_text.splitlines())

            formatted_code = ""
            if code_response:
                formatted_code = f"\n\n### Code:\n```python\n{code_response}\n```"

            final_message = ""
            if formatted_think:
                final_message += formatted_think + "\n\n"
            final_message += full_response if full_response else "⚠️ No answer returned."
            final_message += formatted_code

            st.session_state.chat_history.append(("assistant", final_message))

            with st.chat_message("assistant"):
                st.markdown(final_message)

else:
    st.info("📥 Upload a file or enter text/URL to start chatting.")

# --- Optional Clear Button ---
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

# --- Tools Panel for Mindmap, Flashcards, Quiz, Summary, Key Points ---
st.sidebar.header("🔧 Smart Tools")

if st.sidebar.button("🧠 Generate Mind Map"):
    with st.spinner("🔍 Analyzing context..."):
        retrieved_chunks = vs.get_all_texts(allowed_sources=allowed_ids)
        context = "\n".join(retrieved_chunks[:3])
        mindmap_prompt = f"Generate a mindmap based on the following context:\n\n{context}"
        mindmap_response = call_deepseek(mindmap_prompt)
        st.sidebar.expander("Mindmap Output", expanded=True).markdown(mindmap_response['full'])

if st.sidebar.button("🎴 Generate Flashcards"):
    with st.spinner("🔍 Analyzing context..."):
        retrieved_chunks = vs.get_all_texts(allowed_sources=allowed_ids)
        context = "\n".join(retrieved_chunks[:3])
        flashcard_prompt = f"Generate flashcards based on the following context:\n\n{context}"
        flashcard_response = call_deepseek(flashcard_prompt)
        st.sidebar.expander("Flashcards Output", expanded=True).markdown(flashcard_response['full'])

# 🚀 Updated Quiz Section: Input-based Quiz Generator
st.sidebar.markdown("### ❓ Generate Quiz")
quiz_question = st.sidebar.text_input("Enter topic/question for quiz", key="quiz_question")

if st.sidebar.button("📋 Create Quiz"):
    if not quiz_question.strip():
        st.sidebar.warning("⚠️ Please enter a topic or question first.")
    elif not allowed_ids:
        st.sidebar.warning("⚠️ Please select at least one knowledge source.")
    else:
        with st.spinner("🔍 Retrieving context for quiz..."):
            # Step 1: Retrieve relevant content from the knowledge base
            retrieved_chunks = vs.query(quiz_question, k=7, allowed_sources=allowed_ids)

            if not retrieved_chunks:
                st.sidebar.warning("⚠️ No relevant content found in the knowledge base.")
            else:
                # Step 2: Combine the relevant content into context for DeepSeek
                context = "\n".join([chunk['chunk'] for chunk in retrieved_chunks[:3]])  # Get first 3 chunks as context
                quiz_prompt = f"""
                Create a quiz based ONLY on the following context:
                {context}
                - Format the questions as multiple choice.
                - Provide 4 possible answers for each question.
                """

                # Step 3: Pass the context and topic to DeepSeek for quiz generation
                quiz_response = call_deepseek(quiz_prompt)

                if 'full' in quiz_response:
                    st.sidebar.expander("Quiz Output", expanded=True).markdown(quiz_response['full'])
                else:
                    st.sidebar.warning("⚠️ No quiz questions returned.")

if st.sidebar.button("📝 Generate Summary"):
    with st.spinner("🔍 Analyzing context..."):
        retrieved_chunks = vs.get_all_texts(allowed_sources=allowed_ids)
        context = "\n".join(retrieved_chunks[:3])
        summary_prompt = f"Summarize the following context:\n\n{context}"
        summary_response = call_deepseek(summary_prompt)
        st.sidebar.expander("Summary Output", expanded=True).markdown(summary_response['full'])

if st.sidebar.button("📌 Generate Key Points"):
    with st.spinner("🔍 Analyzing context..."):
        retrieved_chunks = vs.get_all_texts(allowed_sources=allowed_ids)
        context = "\n".join(retrieved_chunks[:3])
        keypoints_prompt = f"Generate key points from the following context:\n\n{context}"
        keypoints_response = call_deepseek(keypoints_prompt)
        st.sidebar.expander("Key Points Output", expanded=True).markdown(keypoints_response['full'])
