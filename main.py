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

# --- Calculate allowed knowledge sources ---
allowed_ids = [src["id"] for src in st.session_state.sources if src.get("checked", False)]

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

    if user_input and user_input.strip():
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append(("user", user_input))

        if not allowed_ids:
            st.warning("⚠️ Please select at least one knowledge source.")
        else:
            with st.spinner("🔍 Retrieving relevant information..."):
                retrieved_chunks = vs.query(user_input, k=7, allowed_sources=allowed_ids)

            context = "\n".join(chunk['chunk'] for chunk in retrieved_chunks[:3])

            # ✅ Add recent chat history as memory context
            history_limit = 3
            memory_context = ""
            for role, msg in st.session_state.chat_history[-history_limit*2:]:
                if role == "user":
                    memory_context += f"User: {msg}\n"
                else:
                    memory_context += f"Assistant: {msg}\n"

            # Updated prompt with contextual memory
            prompt = f"""You are a helpful assistant.

Here is the recent conversation:
{memory_context}

Use the following knowledge context to answer the user's question:
{context}

Current Question: {user_input}
Answer:"""

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

# --- Flashcard Generation ---
flashcard_topic = st.sidebar.text_input("Enter topic/question for Flashcards", key="flashcard_topic")

if st.sidebar.button("🎴 Generate Flashcards"):
    if not flashcard_topic.strip():
        st.sidebar.warning("⚠️ Please enter a topic or question first.")
    elif not allowed_ids:
        st.sidebar.warning("⚠️ Please select at least one knowledge source.")
    else:
        with st.spinner("🔍 Retrieving context for flashcards..."):
            retrieved_chunks = vs.query(flashcard_topic, k=7, allowed_sources=allowed_ids)

            if not retrieved_chunks:
                st.sidebar.warning("⚠️ No relevant content found in the knowledge base.")
            else:
                context = "\n".join([chunk['chunk'] for chunk in retrieved_chunks[:3]])
                flashcard_prompt = f"""
                Generate 10 flashcards based ONLY on the following context:

                {context}

                Format strictly as:
                Question: ...
                Answer: ...

                Separate each flashcard with a double newline.
                """

                flashcard_response = call_deepseek(flashcard_prompt)
                output = flashcard_response.get("full", "").strip()

                # Clean out <think> sections if present
                output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()

                # Parse flashcards
                flashcards_raw = re.split(r'\n{2,}', output)
                flashcards = []

                for card in flashcards_raw:
                    q_match = re.search(r"Question:\s*(.+)", card, re.IGNORECASE)
                    a_match = re.search(r"Answer:\s*(.+)", card, re.IGNORECASE)
                    if q_match and a_match:
                        flashcards.append({
                            "question": q_match.group(1).strip(),
                            "answer": a_match.group(1).strip()
                        })

                if flashcards:
                    st.sidebar.markdown("### 🎴 Flashcards:")
                    pastel_colors = ["#fef3c7", "#d1fae5", "#e0e7ff", "#fee2e2", "#f3e8ff"]  # soft, vibrant colors

                    for i, card in enumerate(flashcards):
                        bg_color = pastel_colors[i % len(pastel_colors)]
                        flashcard_html = f"""
                        <div style="
                            background-color: {bg_color};
                            border-radius: 12px;
                            padding: 16px;
                            margin-bottom: 15px;
                            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.05);
                            transition: transform 0.2s;
                            font-family: 'Segoe UI', sans-serif;
                        ">
                            <p style="font-weight: bold; margin-top: 0; color: #1f2937;">🧠 Question:</p>
                            <p style="margin-bottom: 10px; color: #111827;">{card['question']}</p>
                            <details style="color: #065f46;">
                                <summary style="cursor: pointer; font-weight: 600;"></summary>
                                <p style="margin-top: 10px;"><strong>Answer:</strong> {card['answer']}</p>
                            </details>
                        </div>
                        """
                        st.markdown(flashcard_html, unsafe_allow_html=True)
                else:
                    st.sidebar.warning("⚠️ Could not parse any flashcards properly.")

# Input-based Quiz Generator (Sidebar)
quiz_question = st.sidebar.text_input("Enter topic/question for quiz", key="quiz_question")
if st.sidebar.button("📋 Generate Quiz"):
    with st.spinner("🔍 Retrieving context for quiz..."):
        # Assuming vs.query() and allowed_ids are defined elsewhere in your code
        quiz_question = "Your predefined topic/question here"  # Replace with the topic you want to generate the quiz for
        retrieved_chunks = vs.query(quiz_question, k=7, allowed_sources=allowed_ids)

        if not retrieved_chunks:
            st.warning("⚠️ No relevant content found in the knowledge base.")
        else:
            context = "\n".join([chunk['chunk'] for chunk in retrieved_chunks[:3]])  # Get first 3 chunks as context
            quiz_prompt = f"""
                Create a quiz consisting of 10 multiple-choice questions based ONLY on the following context:
                {context}
                - For each question, provide 4 possible answers.
                - Clearly indicate the correct answer using '(Correct Answer)'.
                """

            # Call to your DeepSeek function or API
            quiz_response = call_deepseek(quiz_prompt)

            if 'full' in quiz_response:
                # Clean the response by removing "think" tag content
                cleaned_quiz_data = re.sub(r'<think>.*?</think>', '', quiz_response['full'], flags=re.DOTALL)
                
                # Display the quiz in the main chat page
                st.markdown("""<div style="text-align: center;"><h3>📋 Generated Quiz</h3></div>""", unsafe_allow_html=True)

                # Print the cleaned quiz data for debugging
                st.write("###  Quiz Data:", cleaned_quiz_data)

                quiz_data = cleaned_quiz_data.split('\n')

            else:
                st.warning("⚠️ No quiz questions returned.")

# ✨ Enhanced Key Points Generation with Input and Styling (Sidebar)
st.sidebar.header("🔑 Key Point Generator")

key_points_input = st.sidebar.text_area("Enter text to generate key points from:", height=150)

if st.sidebar.button("📌 Generate Key Points"):
    if not key_points_input.strip():
        st.sidebar.warning("⚠️ Please enter some text to generate key points.")
    else:
        with st.spinner("🔍 Analyzing text..."):
            keypoints_prompt = f"""Generate concise and impactful key points from the following text, EXCLUDE any content within <think> tags. Ensure that each key point clearly explains a distinct concept .:\n\n{key_points_input}"""            
            keypoints_response = call_deepseek(keypoints_prompt)

            if 'full' in keypoints_response:
                key_points_output = keypoints_response['full'].strip()
                # Remove any <think> tags and their content from the final output
                key_points_output = re.sub(r"<think>.*?</think>", "", key_points_output, flags=re.DOTALL).strip()
                if key_points_output:
                    # Show the key points result on the main page (chat page)
                    st.markdown("""
                        <div style="text-align: center;">
                            <h3>✨ Key Points</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    st.subheader("📝 Key Points:")
                    key_points_list = [point.strip() for point in key_points_output.splitlines() if point.strip()]
                    for i, point in enumerate(key_points_list):
                        st.markdown(f"- ✨ **Point {i+1}:** {point}")
                else:
                    st.info("No key points were generated after filtering.")
            else:
                st.error("Failed to generate key points.")