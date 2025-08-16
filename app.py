"""
Streamlit application for the Intelligent Document Analyst.

This is the main user interface for the Graph RAG Document Analyzer.
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from src.loader import DocumentLoader
from src.graph_builder import get_graph_connection, GraphBuilder
from src.chains import create_graph_rag_chain

# Load environment variables from .env file
load_dotenv()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Intelligent Document Analyst",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("Intelligent Document Analyst")
    st.subheader("Graph RAG-powered Document Q&A System")
    
    st.markdown("""
    This application uses **Graph RAG** (Retrieval-Augmented Generation) to analyze documents 
    by building a knowledge graph from their content. Unlike traditional Q&A systems, 
    it can answer complex questions that require understanding relationships between entities.
    """)
    
    # Sidebar for configuration and status
    with st.sidebar:
        st.header("Configuration")
        
        # Check environment setup
        env_status = check_environment()
        display_environment_status(env_status)
        
        st.header("Graph Status")
        if env_status["all_ready"]:
            # Only connect to graph if user requests it to avoid constant connections
            if st.button("Refresh Graph Status"):
                try:
                    graph_connection = get_graph_connection()
                    schema = graph_connection.get_schema()
                    
                    st.metric("Node Labels", len(schema["node_labels"]))
                    st.metric("Relationship Types", len(schema["relationships"]))
                    st.metric("Properties", len(schema["properties"]))
                    
                    with st.expander("View Schema Details"):
                        st.write("**Node Labels:**", schema["node_labels"])
                        st.write("**Relationships:**", schema["relationships"])
                    
                    graph_connection.close()
                    
                except Exception as e:
                    st.error(f"Graph connection error: {str(e)}")
            else:
                st.info("Click 'Refresh Graph Status' to view database schema")
        
        st.header("Actions")
        if st.button("Clear Graph Database", type="secondary"):
            if env_status["all_ready"]:
                clear_database()
    
    # Main content area
    if not env_status["all_ready"]:
        st.error("Please configure your environment before proceeding.")
        st.info("Check the sidebar for missing configuration items.")
        return
    
    # Document upload and processing
    st.header("Document Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Check for existing sample document
        sample_doc_path = Path("data/Global-AI-Policy-V1.pdf")
        
        if sample_doc_path.exists():
            st.info(f"Found sample document: {sample_doc_path.name}")
            if st.button("Use Sample Document & Build Knowledge Graph", type="primary"):
                process_document(str(sample_doc_path))
        
        st.write("**Or upload your own PDF:**")
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type="pdf",
            help="Upload a PDF document to build a knowledge graph"
        )
        
        if uploaded_file is not None:
            # Save uploaded file with timestamp to avoid conflicts
            import time
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Create unique filename to avoid permission conflicts
            timestamp = int(time.time())
            base_name = uploaded_file.name.rsplit('.', 1)[0]
            extension = uploaded_file.name.rsplit('.', 1)[1] if '.' in uploaded_file.name else 'pdf'
            unique_filename = f"{base_name}_{timestamp}.{extension}"
            file_path = data_dir / unique_filename
            
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded: {uploaded_file.name}")
                
                # Process document button
                if st.button("Build Knowledge Graph", type="primary"):
                    process_document(str(file_path))
                    
            except PermissionError:
                st.error("Permission denied. Try using a different filename or check file permissions.")
            except Exception as e:
                st.error(f"Error saving file: {str(e)}")
    
    with col2:
        st.info("""
        **Tips for better results:**
        - Use well-structured documents
        - PDFs with clear headings work best
        - Longer documents provide richer graphs
        """)
    
    # Q&A Interface
    st.header("Ask Questions")
    
    question = st.text_input(
        "What would you like to know about the document?",
        placeholder="e.g., How did the strategy proposed by the CTO affect Product X?"
    )
    
    if question:
        with st.spinner("🔍 Analyzing the knowledge graph..."):
            try:
                rag_chain = create_graph_rag_chain()
                result = rag_chain.query(question)
                
                # Display answer
                st.subheader("📝 Answer")
                st.markdown(result["answer"])
                
                # Display debug information in expander
                with st.expander("🔍 Debug Information"):
                    if "generated_cypher" in result:
                        st.code(result["generated_cypher"], language="cypher")
                    
                    if "formatted_context" in result:
                        st.text_area("Context from Graph", result["formatted_context"], height=200)
                
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
    
    # Example questions
    with st.expander("💡 Try these example questions"):
        st.markdown("""
        - **Simple fact retrieval:** "Who is the CEO?"
        - **Relationship queries:** "What products did [Person] work on?"
        - **Multi-hop questions:** "How did the Q4 results affect the strategy for Product X?"
        - **Risk analysis:** "What were the key risks mentioned in the document?"
        - **Timeline questions:** "What happened after the merger was announced?"
        """)


def check_environment() -> dict:
    """Check if all required environment variables are set."""
    required_vars = [
        "OPENAI_API_KEY",
        "NEO4J_URI", 
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD"
    ]
    
    status = {}
    for var in required_vars:
        status[var] = bool(os.getenv(var))
    
    status["all_ready"] = all(status.values())
    return status


def display_environment_status(env_status: dict):
    """Display environment configuration status."""
    for var, is_set in env_status.items():
        if var == "all_ready":
            continue
        
        icon = "✅" if is_set else "❌"
        st.write(f"{icon} {var}")
    
    if not env_status["all_ready"]:
        st.warning("Copy `.env.template` to `.env` and fill in your credentials.")


def process_document(file_path: str):
    """Process a document and build knowledge graph."""
    try:
        with st.status("Processing document...", expanded=True) as status:
            # Step 1: Load and chunk document
            st.write("📖 Loading and chunking document...")
            loader = DocumentLoader()
            chunks = loader.load_and_chunk_document(file_path)
            
            stats = loader.get_document_stats(chunks)
            st.write(f"✅ Loaded {stats['total_chunks']} chunks ({stats['total_chars']} characters)")
            
            # Step 2: Build knowledge graph
            st.write("🧠 Building knowledge graph...")
            graph_connection = get_graph_connection()
            
            # Clear existing graph (optional)
            if st.checkbox("Clear existing graph", value=True):
                graph_connection.clear_database()
                st.write("🗑️ Cleared existing graph")
            
            builder = GraphBuilder(graph_connection)
            graph_stats = builder.build_graph_from_document(chunks)
            
            st.write(f"✅ Created {graph_stats['total_nodes']} nodes and {graph_stats['total_relationships']} relationships")
            
            graph_connection.close()
            status.update(label="✅ Document processed successfully!", state="complete")
        
        st.success("🎉 Knowledge graph built successfully! You can now ask questions.")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")


def clear_database():
    """Clear the Neo4j database."""
    try:
        graph_connection = get_graph_connection()
        graph_connection.clear_database()
        graph_connection.close()
        st.success("🗑️ Database cleared successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error clearing database: {str(e)}")


if __name__ == "__main__":
    main()