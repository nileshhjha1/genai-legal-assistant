import subprocess
import sys
import os
import socket


def check_port(port):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0


def find_available_port(start_port=8501, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if check_port(port):
            return port
    return start_port  # Fallback to original port


def check_requirements():
    """Check if all required packages are installed"""
    try:
        import streamlit
        import langchain
        import google.generativeai as genai
        import sentence_transformers
        import transformers
        import torch
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False


def main():
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("Error: main.py not found. Please run from the src directory.")
        return

    # Check requirements
    if not check_requirements():
        return

    # Find available port
    port = find_available_port(8501)

    if port != 8501:
        print(f"Port 8501 is busy. Using port {port} instead...")
    else:
        print(f"Using port {port}...")

    print("Starting Streamlit app...")
    print(f"App will be available at: http://localhost:{port}")
    print("Using Google Gemini AI for enhanced responses")

    # Run streamlit with available port
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "main.py",
        "--server.port", str(port),
        "--server.address", "localhost"
    ])


if __name__ == "__main__":
    main()