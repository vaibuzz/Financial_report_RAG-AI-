"""
Streamlit interface for Financial Document RAG system.
Beautiful demo for portfolio presentation.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.complete_rag_system import CompleteRAGSystem
from rag.providers import AnthropicProvider, OpenAIProvider, OllamaProvider

# Page config
st.set_page_config(
    page_title="Financial RAG System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .source-card {
        background: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_system(provider_type: str, api_key: str, model: str):
    """Initialize RAG system with caching."""
    if provider_type == "Anthropic":
        provider = AnthropicProvider(api_key=api_key, model=model)
    elif provider_type == "OpenAI":
        provider = OpenAIProvider(api_key=api_key, model=model)
    else:
        # Pass empty API key for Ollama as it is ignored
        provider = OllamaProvider(api_key="", model=model)

    return CompleteRAGSystem(llm_provider=provider)


def main():
    # Header
    load_dotenv()

    st.markdown('<h1 class="main-header">📊 Financial Document Q&A System</h1>', unsafe_allow_html=True)
    st.markdown("""
    RAG (Retrieval Augmented Generation) system for financial document analysis.
    Upload your PDFs and ask questions with answers based on the actual content.
    """)

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Provider selection
        provider_type = st.selectbox(
            "LLM Provider",
            ["Anthropic", "OpenAI", "Ollama"],
            help="Select the LLM provider"
        )

        # Model selection
        if provider_type == "Anthropic":
            model_options = {
                "Claude Sonnet 4": "claude-sonnet-4-20250514",
                "Claude Sonnet 3.5": "claude-sonnet-3-5-20241022",
                "Claude Haiku 3.5": "claude-haiku-3-5-20241022"
            }
        elif provider_type == "OpenAI":
            model_options = {
                "GPT-4o": "gpt-4o",
                "GPT-4o Mini": "gpt-4o-mini",
                "GPT-4 Turbo": "gpt-4-turbo"
            }
        else:
            model_options = {
                "Llama 3 (8B)": "llama3",
                "Mistral (7B)": "mistral",
                "Phi-3 (3.8B)": "phi3",
                "Gemma (2B/7B)": "gemma",
                "Llama 3.2 (3B)": "llama3.2"
            }

        selected_model = st.selectbox(
            "Model",
            list(model_options.keys())
        )
        model = model_options[selected_model]

        # API Key (Hidden if Ollama)
        if provider_type != "Ollama":
            api_key_env = "ANTHROPIC_API_KEY" if provider_type == "Anthropic" else "OPENAI_API_KEY"
            api_key = st.text_input(
                f"API Key ({provider_type})",
                type="password",
                value=os.getenv(api_key_env, ""),
                help=f"Your {provider_type} API key"
            )
        else:
            api_key = ""  # Not needed for local Ollama
            st.info("💡 Ollama does not require an API Key. Models run locally at zero cost.")

        st.divider()

        # RAG Parameters
        st.subheader("RAG Parameters")

        k_results = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="How many document chunks to use as context"
        )

        min_score = st.slider(
            "Minimum similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum score to include a chunk (0-1)"
        )

        enable_streaming = st.checkbox(
            "Enable streaming",
            value=True,
            help="Show the response as it is being generated"
        )

        st.divider()

        # System stats
        if 'rag_system' in st.session_state and st.session_state.rag_system:
            st.subheader("📈 System Statistics")
            total_docs = st.session_state.rag_system.vector_store.index.ntotal
            st.metric("Indexed chunks", total_docs)

            if 'total_cost' in st.session_state:
                st.metric("Total cost", f"${st.session_state.total_cost:.4f}")

            if 'total_queries' in st.session_state:
                st.metric("Total queries", st.session_state.total_queries)

    # Main area - Two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📁 Document Upload")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to index"
        )

        if uploaded_files and st.button("🚀 Index Documents", type="primary"):
            if not api_key and provider_type != "Ollama":
                st.error(f"⚠️ Please enter your {provider_type} API key in the sidebar!")
            else:
                with st.spinner("Initializing RAG system..."):
                    # Initialize RAG system
                    rag_system = initialize_rag_system(provider_type, api_key, model)

                    # Save temporary files and index
                    pdf_paths = []
                    with tempfile.TemporaryDirectory() as tmpdir:
                        for uploaded_file in uploaded_files:
                            file_path = Path(tmpdir) / uploaded_file.name
                            with open(file_path, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                            pdf_paths.append(str(file_path))

                        # Index documents
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for i, pdf_path in enumerate(pdf_paths):
                            status_text.text(f"Indexing {Path(pdf_path).name}...")
                            progress_bar.progress((i + 1) / len(pdf_paths))

                        rag_system.index_documents(pdf_paths)

                    # Store in session state
                    st.session_state.rag_system = rag_system
                    st.session_state.total_cost = 0.0
                    st.session_state.total_queries = 0

                    st.success(f"✅ {len(uploaded_files)} documents indexed successfully!")
                    st.rerun()

        # Show indexed files
        if 'rag_system' in st.session_state and st.session_state.rag_system:
            st.divider()
            st.subheader("📚 Indexed Documents")

            # Extract unique sources from metadata
            if st.session_state.rag_system.vector_store.metadata:
                sources = set(
                    m['source']
                    for m in st.session_state.rag_system.vector_store.metadata
                )
                for source in sources:
                    st.text(f"📄 {source}")

    with col2:
        st.header("💬 Ask a Question")

        if 'rag_system' not in st.session_state or not st.session_state.rag_system:
            st.info("👈 Please load PDF documents in the left column first")
        else:
            # Query input
            query = st.text_area(
                "Your question",
                placeholder="Ex: What was the revenue growth in Q4 2023?",
                height=100
            )

            if st.button("🔍 Search Answer", type="primary", disabled=not query):
                with st.spinner("Searching..."):
                    # Query the system
                    response = st.session_state.rag_system.query(
                        question=query,
                        k=k_results,
                        min_score=min_score,
                        stream=False  # Handle streaming separately for Streamlit
                    )

                    # Update stats
                    st.session_state.total_cost += response.cost_usd
                    st.session_state.total_queries += 1

                    # Display response
                    st.divider()
                    st.subheader("✨ Answer")
                    st.markdown(response.answer)

                    # Display metadata
                    st.divider()
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Tokens used", response.tokens_used)
                    with cols[1]:
                        st.metric("Query cost", f"${response.cost_usd:.6f}")
                    with cols[2]:
                        st.metric("Sources used", len(response.sources))

                    # Display sources
                    if response.sources:
                        st.divider()
                        st.subheader("📖 Sources")

                        for i, source in enumerate(response.sources, 1):
                            with st.expander(
                                    f"Source {i}: {source.metadata.get('source', 'unknown')} - Page {source.metadata.get('page', '-')} "
                                    f"(Score: {source.score:.4f})"
                            ):
                                st.text(source.chunk_text)

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #6B7280; padding: 2rem;'>
        <p><strong>Financial Document RAG System</strong></p>
        <p>Powered by SentenceTransformers, FAISS, and {provider}</p>
        <p>Built by Vaibhav Patil | <a href='https://github.com/vaibuzz/Financial_report_RAG-AI-.git'>GitHub</a></p>
    </div>
    """.format(provider=provider_type), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
