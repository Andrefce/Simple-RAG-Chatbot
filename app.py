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
        if self.text_chunks:
            self.index, self.embeddings = self._create_faiss_index(self.text_chunks)
        else:
            # Initialize with empty values if no chunks
            dim = embedder.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(dim)
            self.embeddings = np.array([], dtype=np.float32).reshape(0, dim)

    def _load_and_split_pdf(self, pdf_path):
        """Load PDF, extract text, and split into chunks."""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            if not documents:
                st.error("No content could be extracted from the PDF.")
                return []
                
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
            text_chunks = text_splitter.split_documents(documents)
            return [chunk.page_content for chunk in text_chunks]
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return []

    def _create_faiss_index(self, text_chunks):
        """Create a FAISS index from text chunks using Hugging Face embeddings."""
        embeddings = embedder.encode(text_chunks, convert_to_numpy=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 (Euclidean) distance index
        index.add(embeddings)
        return index, embeddings

    def _retrieve_relevant_chunks(self, query, top_k=5):
        """Retrieve the most relevant text chunks for a given query."""
        if not self.text_chunks:
            return []
            
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.text_chunks[i] for i in indices[0] if i < len(self.text_chunks)]

    def get_answer(self, question: str) -> str:
        """Answer a question using the most relevant document chunks."""
        if not self.text_chunks:
            return "No document text available. Please upload a valid PDF document."

        relevant_chunks = self._retrieve_relevant_chunks(question)
        if not relevant_chunks:
            return "Could not find relevant information in the document."
            
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
        self.tmp_file_path = None
        self._setup_page()
        self._initialize_session_state()

    @staticmethod
    def _setup_page():
        """Configure the Streamlit page."""
        st.set_page_config(page_title="PDF Q&A with AI", page_icon="📄")
        st.title("📄 PDF Q&A Assistant with AI & Embeddings")

    @staticmethod
    def _initialize_session_state():
        """Initialize session state for chat history and file tracking."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False
        if 'qa_system' not in st.session_state:
            st.session_state.qa_system = None

    def run(self):
        """Run the Streamlit application."""
        # Add sidebar with instructions
        with st.sidebar:
            st.header("About this app")
            st.write("Upload a PDF document and ask questions about its content.")
            st.write("The app uses:")
            st.write("- FAISS for vector similarity search")
            st.write("- SentenceTransformers for embeddings")
            st.write("- Google's Gemini for generating answers")

        # Main area
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

        # Handle file upload and processing
        if uploaded_file:
            # Check if this is a new file
            current_file_name = getattr(uploaded_file, "name", "")
            last_processed_name = st.session_state.get("last_file_name", "")
            
            if current_file_name != last_processed_name:
                process_button = st.button("Process PDF")
                if process_button:
                    with st.spinner("Processing PDF... This may take a moment."):
                        try:
                            # Clean up previous temp file if it exists
                            if self.tmp_file_path and os.path.exists(self.tmp_file_path):
                                try:
                                    os.remove(self.tmp_file_path)
                                except Exception:
                                    pass
                                    
                            # Create new temp file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                self.tmp_file_path = tmp_file.name
                            
                            # Process the PDF
                            self.qa_system = DocumentQA(self.tmp_file_path)
                            
                            # Update session state
                            if self.qa_system and self.qa_system.text_chunks:
                                st.session_state.qa_system = self.qa_system
                                st.session_state.pdf_processed = True
                                st.session_state.last_file_name = current_file_name
                                st.session_state.messages = []  # Clear previous chat
                                st.success(f"✅ PDF processed successfully! Extracted {len(self.qa_system.text_chunks)} text chunks.")
                            else:
                                st.error("Could not extract text from the PDF. Please try another document.")
                                st.session_state.pdf_processed = False
                                st.session_state.qa_system = None
                        except Exception as e:
                            st.error(f"Error processing PDF: {str(e)}")
                            st.session_state.pdf_processed = False
                            st.session_state.qa_system = None
            elif st.session_state.pdf_processed:
                st.success(f"✅ PDF already processed. Ready to answer questions.")
                # Restore QA system from session state
                self.qa_system = st.session_state.qa_system
        
        # Display chat interface if PDF is processed
        if st.session_state.pdf_processed and st.session_state.qa_system is not None:
            self._display_chat_history()
            self._handle_user_input()
        elif not uploaded_file:
            st.info("👆 Please upload a PDF document to start asking questions.")

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

            # Use the QA system from session state for consistency
            qa_system = st.session_state.qa_system
            
            # Check if qa_system exists before trying to use it
            if qa_system is None:
                response = "Sorry, the document hasn't been processed correctly. Please try uploading the PDF again."
            else:
                with st.spinner("Analyzing your question..."):
                    response = qa_system.get_answer(prompt)

            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})

    def __del__(self):
        """Clean up temporary files when the app is closed."""
        if hasattr(self, 'tmp_file_path') and self.tmp_file_path and os.path.exists(self.tmp_file_path):
            try:
                os.remove(self.tmp_file_path)
            except Exception:
                pass

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()