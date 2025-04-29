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
                context = "\n".join([chunk['chunk'] for chunk in retrieved_chunks[:3]])  # Get first 3 chunks as context
                flashcard_prompt = f"""
                    Generate flashcard questions and answers based ONLY on the following context:
                    {context}
                    - Provide at least 5 flashcards.
                    - Each flashcard MUST follow this exact format:
                      Question: [Your Question Here]
                      Answer: [The Correct Answer Here]
                    - Ensure there is a clear newline character separating the "Question:" and "Answer:" lines for each flashcard.
                    - Start each new flashcard after a double newline character ('\\n\\n').
                    """

                flashcard_response = call_deepseek(flashcard_prompt)

                if 'full' in flashcard_response:
                    flashcard_output = flashcard_response['full']
                    # Remove the <think> block entirely before processing
                    flashcard_output = re.sub(r"<think>.*?</think>", "", flashcard_output, flags=re.DOTALL).strip()
                    flashcard_lines = [line.strip() for line in flashcard_output.splitlines() if line.strip()]

                    questions = []
                    answers = []
                    current_question = None

                    for line in flashcard_lines:
                        if line.startswith("Question:"):
                            if current_question and answers:
                                questions.append(current_question)
                                if answers:
                                    questions.append(answers.pop(0))
                                current_question = line.replace("Question:", "").strip()
                            elif current_question:
                                questions.append(current_question)
                                if answers:
                                    questions.append(answers.pop(0))
                                current_question = line.replace("Question:", "").strip()
                            else:
                                current_question = line.replace("Question:", "").strip()
                        elif line.startswith("Answer:"):
                            answers.append(line.replace("Answer:", "").strip())
                        elif current_question:
                            current_question += " " + line

                    if current_question:
                        questions.append(current_question)
                        if answers:
                            questions.append(answers.pop(0))

                    if len(questions) >= 2 and len(questions) % 2 == 0:
                        st.sidebar.markdown("### 🎴 Flashcards:")
                        for i in range(0, len(questions), 2):
                            question = questions[i]
                            answer = questions[i+1]
                            flashcard_html = f"""
                                <div style="border: 1px solid #e1e4e8; border-radius: 5px; padding: 15px; margin-bottom: 10px;">
                                    <p style="font-weight: bold; margin-top: 0;">Question:</p>
                                    <p>{question}</p>
                                    <details>
                                        <summary style="cursor: pointer;">Show Answer</summary>
                                        <p style="margin-top: 10px;"><strong>Answer:</strong> {answer}</p>
                                    </details>
                                </div>
                            """
                            st.markdown(flashcard_html, unsafe_allow_html=True)
                    elif questions:
                        st.sidebar.warning("⚠️ Could not parse flashcards into question-answer pairs.")
                    else:
                        st.sidebar.warning("⚠️ No flashcards generated.")
                else:
                    st.sidebar.warning("⚠️ No flashcards generated.")

# 🚀 Updated Quiz Section: Input-based Quiz Generator
quiz_question = st.sidebar.text_input("Enter topic/question for quiz", key="quiz_question")

if st.sidebar.button("📋 Create Quiz"):
    if not quiz_question.strip():
        st.sidebar.warning("⚠️ Please enter a topic or question first.")
    elif not allowed_ids:
        st.sidebar.warning("⚠️ Please select at least one knowledge source.")
    else:
        with st.spinner("🔍 Retrieving context for quiz..."):
            retrieved_chunks = vs.query(quiz_question, k=7, allowed_sources=allowed_ids)

            if not retrieved_chunks:
                st.sidebar.warning("⚠️ No relevant content found in the knowledge base.")
            else:
                context = "\n".join([chunk['chunk'] for chunk in retrieved_chunks[:3]])  # Get first 3 chunks as context
                quiz_prompt = f"""
                    Create a quiz consists of 5 multiple-choice questions based ONLY on the following context:
                    {context}
                    - For each question, provide 4 possible answers.
                    - Clearly indicate the correct answer using '(Correct Answer)'.
                    """

                quiz_response = call_deepseek(quiz_prompt)

                if 'full' in quiz_response:
                    st.sidebar.expander("Quiz Output", expanded=True).markdown(quiz_response['full'])
                else:
                    st.sidebar.warning("⚠️ No quiz questions returned.")


# ✨ Enhanced Key Points Generation with Input and Styling
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
                    st.sidebar.subheader("📝 Key Points:")
                    key_points_list = [point.strip() for point in key_points_output.splitlines() if point.strip()]
                    for i, point in enumerate(key_points_list):
                        st.sidebar.markdown(f"- ✨ **Point {i+1}:** {point}")
                else:
                    st.sidebar.info("No key points were generated after filtering.")
            else:
                st.sidebar.error("Failed to generate key points.")