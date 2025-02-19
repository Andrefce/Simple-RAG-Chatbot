import streamlit as st
import os
import faiss
import numpy as np
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Configure API securely
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]  # Store API key securely
genai.configure(api_key=GEMINI_API_KEY)

# Load Sentence Transformer for embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Create Gemini model session if not already initialized
if 'model' not in st.session_state:
    st.session_state.model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        },
    )
    st.session_state.chat_session = st.session_state.model.start_chat(history=[])

class DocumentQA:
    """Handles document-based Q&A with embeddings and FAISS search."""

    def __init__(self, pdf_path):
        """Initialize DocumentQA with a PDF file path."""
        self.text_chunks = self._load_and_split_pdf(pdf_path)
        self.index, self.embeddings = self._create_faiss_index(self.text_chunks)

    def _load_and_split_pdf(self, pdf_path):
        """Load PDF, extract text, and split into chunks."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents(documents)
        return [chunk.page_content for chunk in text_chunks]

    def _create_faiss_index(self, text_chunks):
        """Create a FAISS index from text chunks using Hugging Face embeddings."""
        embeddings = embedder.encode(text_chunks, convert_to_numpy=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 (Euclidean) distance index
        index.add(embeddings)
        return index, embeddings

    def _retrieve_relevant_chunks(self, query, top_k=5):
        """Retrieve the most relevant text chunks for a given query."""
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.text_chunks[i] for i in indices[0] if i < len(self.text_chunks)]

    def get_answer(self, question: str) -> str:
        """Answer a question using the most relevant document chunks."""
        if not self.text_chunks:
            return "No document text available."

        relevant_chunks = self._retrieve_relevant_chunks(question)
        context = "\n\n".join(relevant_chunks)
        
        prompt_template = ChatPromptTemplate.from_template(
            """Answer the question based only on the provided context:
            <context>
            {context}
            </context>
            Question: {input}

            If the answer cannot be found in the document, say "I'm sorry, I couldn't find relevant information."
            """
        )
        
        final_prompt = prompt_template.format(context=context, input=question)

        try:
            response = st.session_state.chat_session.send_message(final_prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"

class StreamlitApp:
    """Manages the Streamlit UI and interaction."""

    def __init__(self):
        """Initialize the Streamlit app."""
        self.qa_system = None
        self._setup_page()
        self._initialize_session_state()

    @staticmethod
    def _setup_page():
        """Configure the Streamlit page."""
        st.set_page_config(page_title="PDF Q&A with AI", page_icon="📄")
        st.title("📄 PDF Q&A Assistant with AI & Embeddings")

    @staticmethod
    def _initialize_session_state():
        """Initialize session state for chat history."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []

    def run(self):
        """Run the Streamlit application."""
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            self.qa_system = DocumentQA(tmp_file_path)

        self._display_chat_history()
        self._handle_user_input()

    def _display_chat_history(self):
        """Display the chat message history."""
        for message in st.session_state.messages[-50:]:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    def _handle_user_input(self):
        """Handle user input and generate responses."""
        if prompt := st.chat_input("Ask a question about the document..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Verifica se um arquivo PDF foi carregado
        if not self.qa_system:
            response = "No document has been uploaded. Please upload a PDF first."
        else:
            with st.spinner("Analyzing your question..."):
                response = self.qa_system.get_answer(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
