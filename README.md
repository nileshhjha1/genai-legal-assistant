# ⚖️ NyayAI - Gen AI Legal Assistant

NyayAI is an intelligent legal assistant that provides comprehensive information about the Indian Constitution and Indian Penal Code (IPC) using Retrieval-Augmented Generation (RAG) and Google's Gemini AI.

## 🚀 Features

- **Smart Legal Q&A**: Ask questions about Indian Constitution and IPC
- **RAG-powered Responses**: Combines document retrieval with AI generation
- **Multiple Search Strategies**: 
  - RAG approach when relevant documents are found
  - Direct Gemini AI approach for general legal questions
- **Streamlit Web Interface**: User-friendly chat interface
- **Pinecone Vector Database**: Efficient document storage and retrieval
- **Google Gemini Integration**: Advanced AI responses
![WhatsApp Image 2025-11-19 at 12 50 42 AM](https://github.com/user-attachments/assets/202b59d5-9e41-42ba-ac74-6e644733ada7)
![WhatsApp Image 2025-11-19 at 12 52 48 AM](https://github.com/user-attachments/assets/6376489c-3dbc-414e-9fdb-5b476ddfb001)


## 📋 Prerequisites

- Python 3.8+
- Google API Key (for Gemini AI)
- Pinecone API Key (for vector database)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/nyay-ai.git
   cd nyay-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Or use the setup script:*
   ```bash
   python setup.py
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_INDEX_NAME=indian-constitution-ipc
   ```

4. **Prepare your legal document**
   - Place your PDF file (`indian_constitution_ipc.pdf`) in the `data/` directory
   - Or update the path in `main.py` to match your file location

## 🎯 Usage

### Method 1: Using the Launcher Script (Recommended)
```bash
python run_app.py
```

### Method 2: Direct Streamlit Command
```bash
streamlit run main.py
```

### Setup Steps in the App:

1. **Initialize Vector Store** (Sidebar → Setup)
   - Creates vector embeddings from your PDF document
   - This may take a few minutes for large documents

2. **Initialize Chatbot** (Sidebar → Setup)
   - Loads the AI model and connects to the vector store
   - Ready to answer legal questions

3. **Start Chatting**
   - Use sample questions from the sidebar
   - Or type your own legal questions

## 📁 Project Structure

```
nyay-ai/
├── main.py                 # Streamlit web interface
├── chatbot.py             # Core AI chatbot logic
├── vector_store.py        # Pinecone vector database management
├── run_app.py            # Application launcher
├── setup.py              # Dependency installer
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables (create this)
├── data/               # Directory for legal documents
│   └── indian_constitution_ipc.pdf
└── .gitignore
```

## 💬 Sample Questions

- "What is Article 14 of the Indian Constitution?"
- "Explain the fundamental rights in Indian Constitution"
- "What is Section 302 of IPC?"
- "What are the provisions for right to equality?"
- "Explain the punishment for theft under IPC"

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google AI Studio API key
- `PINECONE_API_KEY`: Your Pinecone database API key
- `PINECONE_INDEX_NAME`: Name for your vector index (default: `indian-constitution-ipc`)

### Model Configuration
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **AI Model**: `gemini-2.0-flash`
- **Vector Database**: Pinecone (cloud-based)

## 🛠️ Troubleshooting

### Common Issues:

1. **PDF not found**
   - Ensure the PDF file is in the correct location
   - Check the file path in `main.py`

2. **API keys not working**
   - Verify your Google and Pinecone API keys
   - Ensure they're properly set in the `.env` file

3. **Vector store initialization fails**
   - Check your internet connection
   - Verify Pinecone account and API permissions

4. **Chatbot initialization slow**
   - First-time model loading may take time
   - Subsequent runs will be faster

### Reset Options:
- Use "Delete Vector Store" in sidebar to start fresh
- Restart the application if issues persist

## 📊 System Architecture

```
User Question → Streamlit Interface → SmartLegalChatbot
                                      ↓
                 RAG Approach ← Document Relevance Check → Direct Gemini Approach
                   ↓                            ↓
           Pinecone Vector Store        Google Gemini AI
                   ↓                            ↓
           Legal Documents DB           General Knowledge
```

## 🔒 Privacy & Security

- Your legal documents are processed locally during vectorization
- Vector embeddings are stored securely in Pinecone
- API calls to Google Gemini are encrypted
- No personal data is stored permanently

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini AI for advanced language model capabilities
- Pinecone for vector database infrastructure
- LangChain for RAG framework
- Streamlit for web interface framework

## 📞 Support
nileshnirmaljha@gmail.com
+918356075288

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify your API keys and environment setup
4. Check the application logs for detailed error messages

---

**Note**: This application is designed for educational and informational purposes. For official legal advice, please consult qualified legal professionals.
