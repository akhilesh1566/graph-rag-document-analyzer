#!/usr/bin/env python3
"""
Test script to verify the Graph RAG setup.
"""

import os
import sys
from dotenv import load_dotenv

def test_environment():
    """Test environment variables configuration."""
    print("Testing environment configuration...")
    
    load_dotenv()
    
    required_vars = [
        "OPENAI_API_KEY",
        "NEO4J_URI", 
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD"
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            print(f"[OK] {var}: {'*' * (len(value) - 4) + value[-4:]}")
    
    if missing_vars:
        print(f"[ERROR] Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    print("[OK] All environment variables configured")
    return True

def test_imports():
    """Test that all required packages can be imported."""
    print("\nTesting package imports...")
    
    try:
        from src.loader import DocumentLoader
        print("[OK] Document loader imported successfully")
        
        from src.graph_builder import get_graph_connection
        print("[OK] Graph builder imported successfully")
        
        from src.chains import create_graph_rag_chain
        print("[OK] RAG chains imported successfully")
        
        import streamlit
        print("[OK] Streamlit imported successfully")
        
        from langchain_experimental.graph_transformers import LLMGraphTransformer
        print("[OK] LangChain Graph Transformer imported successfully")
        
        print("[OK] All packages imported successfully")
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False

def test_neo4j_connection():
    """Test Neo4j database connection."""
    print("\nTesting Neo4j connection...")
    
    try:
        from src.graph_builder import get_graph_connection
        
        connection = get_graph_connection()
        schema = connection.get_schema()
        connection.close()
        
        print("[OK] Neo4j connection successful")
        print(f"   Node labels: {len(schema['node_labels'])}")
        print(f"   Relationships: {len(schema['relationships'])}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Neo4j connection failed: {e}")
        print("   Make sure Neo4j is running and credentials are correct")
        return False

def test_document_loading():
    """Test document loading functionality."""
    print("\nTesting document loading...")
    
    try:
        from src.loader import DocumentLoader
        
        # Check for sample document
        sample_doc = "data/Global-AI-Policy-V1.pdf"
        if not os.path.exists(sample_doc):
            print(f"[ERROR] Sample document not found: {sample_doc}")
            return False
        
        loader = DocumentLoader(chunk_size=500, chunk_overlap=100)  # Smaller chunks for testing
        chunks = loader.load_and_chunk_document(sample_doc)
        
        stats = loader.get_document_stats(chunks)
        
        print(f"[OK] Document loaded successfully")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Total characters: {stats['total_chars']}")
        print(f"   Average chunk size: {stats['avg_chunk_size']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Document loading failed: {e}")
        return False

def test_llm_connection():
    """Test OpenAI LLM connection."""
    print("\nTesting OpenAI LLM connection...")
    
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke("Hello! Please respond with 'Connection successful'")
        
        print(f"[OK] OpenAI LLM connection successful")
        print(f"   Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"[ERROR] OpenAI LLM connection failed: {e}")
        print("   Check your OPENAI_API_KEY and account credits")
        return False

def main():
    """Run all tests."""
    print("Graph RAG Setup Test Suite\n")
    
    tests = [
        ("Environment Configuration", test_environment),
        ("Package Imports", test_imports),
        ("Neo4j Connection", test_neo4j_connection),
        ("Document Loading", test_document_loading),
        ("OpenAI LLM Connection", test_llm_connection),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\nAll tests passed! Your Graph RAG system is ready to use.")
        print("   Run 'streamlit run app.py' to start the application.")
    else:
        print("\nSome tests failed. Please fix the issues before proceeding.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)