# Importing Necessary Libraries
import streamlit as st
from moviepy.editor import VideoFileClip
from pytube import YouTube
import whisper
import tempfile
import os
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import yt_dlp
import os

# Loading Whisper model
model = whisper.load_model("base")

# Defining the Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)

# Defining Embeddings and QA chain Creation Function
def create_embeddings(openai_api_key, video_file=None, youtube_url=None):
    os.environ['OPENAI_API_KEY'] = openai_api_key

    try:
        # Extract audio from local video or YouTube URL
        if video_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                temp_video_file.write(video_file.read())
                temp_video_file_path = temp_video_file.name

            video_clip = VideoFileClip(temp_video_file_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                video_clip.audio.write_audiofile(temp_audio_file.name, codec='mp3')
                temp_audio_file_path = temp_audio_file.name
            video_clip.close()
            source = "local_video"
        elif youtube_url is not None:
            ydl_opts = {'format': 'best', 'outtmpl': 'temp_video.mp4'}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            video_clip = VideoFileClip('temp_video.mp4')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                video_clip.audio.write_audiofile(temp_audio_file.name, codec='mp3')
                temp_audio_file_path = temp_audio_file.name
            video_clip.close()
            source = youtube_url
        else:
            return "Please upload a video file or provide a YouTube URL."

        # Transcribe audio to text
        result = model.transcribe(temp_audio_file_path)['text']

        # Create a Document object from the transcribed text with source metadata
        document = Document(page_content=result, metadata={"source": source})

        # Create embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        docs = text_splitter.split_documents([document])
        vector_store = FAISS.from_documents(docs, embeddings)

        # Create QA chain and store it in session state
        qa_chain = RetrievalQAWithSourcesChain.from_llm(
            llm=OpenAI(),
            retriever=vector_store.as_retriever(search_kwargs={"k": 3})
        )
        st.session_state.qa_chain = qa_chain

        return "Embeddings created successfully!"
    except Exception as e:
        return f"Error in creating embeddings: {str(e)}"
    finally:
        if 'temp_audio_file_path' in locals() and os.path.exists(temp_audio_file_path):
            os.remove(temp_audio_file_path)
        if video_file is not None and os.path.exists(temp_video_file_path):
            os.remove(temp_video_file_path)
        if youtube_url is not None and os.path.exists('temp_video.mp4'):
            os.remove('temp_video.mp4')

# Defining Chat with Video Function
def chat_with_video(question):
    if 'qa_chain' not in st.session_state or st.session_state.qa_chain is None:
        return "Please create embeddings first."
    try:
        response = st.session_state.qa_chain({'question': question}, return_only_outputs=True)
        return response['answer']
    except Exception as e:
        return f"Error in chatting with video: {str(e)}"

# Defining File Clearing Function
def clear_files():
    # Clear the QA chain from session state
    st.session_state.qa_chain = None
    return "Session state cleared successfully."

# Creating Streamlit UI
st.title("Video Embedding and Chat Application")
tab1, tab2, tab3 = st.tabs(["Create Embeddings", "Chat with Video", "Clear Files"])

with tab1:
    st.header("Create Embeddings")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    video_file = st.file_uploader("Upload Local Video", type=["mp4"])
    youtube_url = st.text_input("YouTube URL")
    if st.button("Create Embeddings"):
        if openai_api_key:
            output = create_embeddings(openai_api_key, video_file=video_file, youtube_url=youtube_url)
            st.write(output)
        else:
            st.write("Please provide the OpenAI API key.")

with tab2:
    st.header("Chat with Video")
    question = st.text_input("Your Question")
    if st.button("Chat"):
        if question:
            response = chat_with_video(question)
            st.write(response)
        else:
            st.write("Please ask a question.")

with tab3:
    st.header("Clear Files")
    if st.button("Clear All Files"):
        output = clear_files()
        st.write(output)
