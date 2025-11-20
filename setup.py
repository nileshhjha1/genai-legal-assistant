# setup.py
import subprocess
import sys
import os


def install_requirements():
    """Install all required packages"""
    requirements = [
        "streamlit>=1.28.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "sentence-transformers>=2.2.2",
        "pypdf>=3.15.0",
        "python-dotenv>=1.0.0",
        "accelerate>=0.24.0",
        "Pillow>=10.0.0",
        "tqdm>=4.66.1",
        "numpy>=1.24.3",
        "protobuf>=3.20.3",
        "huggingface-hub>=0.19.4",
        "google-generativeai>=0.3.0",
        "langchain-google-genai>=0.0.2",
        "pinecone-client>=3.0.0",
        "langchain-pinecone>=0.0.3"
    ]

    print("Installing required packages...")

    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")

    print("\nSetup completed!")


if __name__ == "__main__":
    install_requirements()