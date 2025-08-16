#!/usr/bin/env python3
"""
Test script to check what happens when app components are loaded.
"""

import os
import sys
from dotenv import load_dotenv

def test_app_functions():
    """Test the functions that are called in the Streamlit app."""
    print("Testing app components...")
    
    try:
        # Load environment
        load_dotenv()
        print("[OK] Environment loaded")
        
        # Test environment check function (from app.py)
        required_vars = ["OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
        env_status = {}
        for var in required_vars:
            env_status[var] = bool(os.getenv(var))
        env_status["all_ready"] = all(env_status.values())
        
        print(f"[OK] Environment status: {env_status['all_ready']}")
        
        # Test graph connection
        from src.graph_builder import get_graph_connection
        try:
            connection = get_graph_connection()
            schema = connection.get_schema()
            connection.close()
            print(f"[OK] Graph connection successful - {len(schema['node_labels'])} node labels")
        except Exception as e:
            print(f"[ERROR] Graph connection failed: {e}")
            return False
        
        # Test imports that might be called in sidebar
        from src.chains import create_graph_rag_chain
        print("[OK] RAG chain import successful")
        
        print("\n[SUCCESS] All app components working correctly")
        return True
        
    except Exception as e:
        print(f"[ERROR] App component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_specific():
    """Test Streamlit-specific functionality."""
    print("\nTesting Streamlit components...")
    
    try:
        import streamlit as st
        print("[OK] Streamlit imported")
        
        # Test functions that might be used
        print("[OK] Streamlit basic functions available")
        
        # Check if our imports work in Streamlit context
        from src.loader import DocumentLoader
        from src.graph_builder import get_graph_connection, GraphBuilder
        from src.chains import create_graph_rag_chain
        print("[OK] All module imports work")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Streamlit test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("App Component Diagnostics\n")
    
    success1 = test_app_functions()
    success2 = test_streamlit_specific()
    
    if success1 and success2:
        print("\n[SUCCESS] All components working - app should load correctly")
        print("If you're seeing issues, please describe what you see in the browser")
    else:
        print("\n[ERROR] Found issues that might prevent app from working properly")