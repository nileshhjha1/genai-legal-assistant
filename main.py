import streamlit as st
import os
import sys
from chatbot import get_chatbot, ConstitutionChatbot
from vector_store import VectorStoreManager


def initialize_session_state():
    """Initialize session state variables"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'vector_initialized' not in st.session_state:
        st.session_state.vector_initialized = False
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = False
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    if 'setup_in_progress' not in st.session_state:
        st.session_state.setup_in_progress = False
    if 'vector_store_created' not in st.session_state:
        st.session_state.vector_store_created = False
    if 'chatbot_initialized' not in st.session_state:
        # Check if we already have a chatbot instance
        try:
            chatbot = get_chatbot()
            if chatbot and chatbot._initialized:
                st.session_state.chatbot_initialized = True
                st.session_state.setup_complete = True
                st.session_state.chatbot = chatbot
            else:
                st.session_state.chatbot_initialized = False
        except:
            st.session_state.chatbot_initialized = False


def setup_vector_store():
    """Setup vector store from PDF - only when explicitly called"""
    try:
        st.session_state.setup_in_progress = True

        # Get the project root directory - FIXED PATH
        project_root = os.path.dirname(os.path.abspath(__file__))

        # Try multiple possible locations for the PDF file
        possible_paths = [
            os.path.join(project_root, "..", "data", "indian_constitution_ipc.pdf"),
            os.path.join(project_root, "data", "indian_constitution_ipc.pdf"),
            os.path.join(project_root, "..", "..", "data", "indian_constitution_ipc.pdf"),
            os.path.join(project_root, "indian_constitution_ipc.pdf"),
            r"C:\Users\Nilesh\PycharmProjects\GENAI\data\indian_constitution_ipc.pdf"  # Your specific path
        ]

        pdf_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                pdf_path = abs_path
                st.success(f"✓ Found PDF at: {abs_path}")
                break

        if pdf_path is None:
            # Show all searched paths for debugging
            st.error("PDF file not found in any of the expected locations:")
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                st.write(f"• {abs_path} - {'Exists' if os.path.exists(abs_path) else 'Not found'}")

            st.info("""
            **Please ensure:**
            1. The PDF file is named exactly: `indian_constitution_ipc.pdf`
            2. It's placed in the `data/` directory next to your project
            3. Or update the path in the code to match your file location
            """)
            st.session_state.setup_in_progress = False
            return False

        with st.spinner("Creating vector store from PDF... This may take a few minutes."):
            vector_manager = VectorStoreManager()
            # Only create if it doesn't exist
            if not vector_manager.check_index_exists():
                vector_manager.create_vector_store(pdf_path)
                st.session_state.vector_store_created = True
            else:
                st.info("Vector store already exists. Using existing one.")

        st.session_state.setup_in_progress = False
        st.session_state.vector_initialized = True
        return True

    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        st.session_state.setup_in_progress = False
        return False

def delete_vector_store():
    """Delete existing vector store"""
    try:
        vector_manager = VectorStoreManager()
        if vector_manager.delete_vector_store():
            st.session_state.vector_initialized = False
            st.session_state.setup_complete = False
            st.session_state.chatbot = None
            st.session_state.messages = []
            st.session_state.vector_store_created = False
            st.session_state.chatbot_initialized = False
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting vector store: {str(e)}")
        return False


def initialize_chatbot():
    """Initialize the chatbot - only once"""
    try:
        # Check if vector store exists
        vector_manager = VectorStoreManager()
        if not vector_manager.check_index_exists():
            st.error("Vector store not found. Please initialize vector store first.")
            return False

        # Use singleton pattern to get chatbot
        chatbot = get_chatbot()

        # Only initialize if not already initialized
        if not chatbot._initialized:
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Step 1/3: Initializing chatbot framework...")
            progress_bar.progress(20)

            status_text.text("Step 2/3: Loading Google Gemini AI...")
            progress_bar.progress(50)

            # Initialize with timeout protection
            import threading
            import time

            init_success = [False]
            init_error = [None]

            def init_target():
                try:
                    chatbot.initialize_qa_chain()
                    init_success[0] = True
                except Exception as e:
                    init_error[0] = e

            # Start initialization in a thread with timeout
            init_thread = threading.Thread(target=init_target)
            init_thread.daemon = True
            init_thread.start()

            # Wait for initialization with timeout
            timeout = 60  # 1 minute
            start_time = time.time()

            while init_thread.is_alive():
                if time.time() - start_time > timeout:
                    st.error("Chatbot initialization timed out. Please try again.")
                    break
                time.sleep(0.1)

            if init_error[0]:
                st.error(f"Failed to initialize chatbot: {init_error[0]}")
                return False

            status_text.text("Step 3/3: Connecting to legal database...")
            progress_bar.progress(80)

            progress_bar.progress(100)
            status_text.text("✓ Chatbot ready!")

        else:
            st.info("Chatbot already initialized.")

        # Store in session state
        st.session_state.chatbot = chatbot
        st.session_state.chatbot_initialized = True
        st.session_state.setup_complete = True

        return True

    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        st.info("""
        **Troubleshooting tips:**
        1. Check your internet connection
        2. Verify your Google API key is valid
        3. Try clicking 'Initialize Chatbot' again
        4. Make sure the vector store is properly initialized
        """)
        return False


def display_sample_questions():
    """Display sample questions in sidebar"""
    st.sidebar.markdown("### Sample Questions")
    sample_questions = [
        "What is Article 14 of the Indian Constitution?",
        "Explain the fundamental rights in Indian Constitution",
        "What is Section 302 of IPC?",
        "What are the provisions for right to equality?",
        "Explain the punishment for theft under IPC"
    ]

    for question in sample_questions:
        if st.sidebar.button(question, key=question):
            st.session_state.user_input = question


def display_chat_history():
    """Display chat messages from history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i + 1}:**")
                        st.markdown(f"Page {source.metadata.get('page', 'N/A')}: {source.page_content[:200]}...")


def display_quick_actions():
    """Display quick action buttons"""
    if st.session_state.setup_complete:
        st.subheader("Quick Actions")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("What is Article 14?"):
                st.session_state.user_input = "What is Article 14 of the Indian Constitution?"

        with col2:
            if st.button("Explain Fundamental Rights"):
                st.session_state.user_input = "Explain the fundamental rights in Indian Constitution"

        with col3:
            if st.button("What is Section 302?"):
                st.session_state.user_input = "What is Section 302 of IPC?"


def process_question(prompt):
    """Process a question and generate response"""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    if st.session_state.chatbot and st.session_state.setup_complete:
        with st.chat_message("assistant"):
            with st.spinner("Searching legal documents..."):
                try:
                    response = st.session_state.chatbot.query(prompt)

                    st.markdown(response["answer"])

                    # Display sources if available
                    if response.get("source_documents"):
                        with st.expander("View Legal References"):
                            for i, doc in enumerate(response["source_documents"]):
                                st.markdown(f"**Reference {i + 1}**")
                                st.markdown(f"**Page:** {doc.metadata.get('page', 'N/A')}")
                                st.markdown(f"**Content:** {doc.page_content[:300]}...")
                                st.markdown("---")

                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("source_documents", [])
                    })

                except Exception as e:
                    error_msg = f"Error processing your question: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    else:
        warning_msg = "Please initialize the chatbot first using the sidebar."
        st.warning(warning_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": warning_msg
        })


def main():
    st.set_page_config(
        page_title="Indian Constitution & IPC Chatbot",
        page_icon="⚖️",
        layout="wide"
    )

    st.title("⚖️ Indian Constitution & IPC Chatbot")
    st.markdown("Ask questions about Indian Constitution and Indian Penal Code")

    initialize_session_state()

    # Sidebar for setup
    with st.sidebar:
        st.header("Setup")

        # Check if vector store exists
        vector_manager = VectorStoreManager()
        vector_store_exists = vector_manager.check_index_exists()

        if vector_store_exists:
            st.success("✓ Vector store found")
            st.session_state.vector_initialized = True
        else:
            st.warning("⚠ Vector store not found")
            st.session_state.vector_initialized = False

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Initialize Vector Store",
                         key="init_vector",
                         disabled=st.session_state.setup_in_progress or vector_store_exists,
                         help="Create vector store from PDF"):
                if setup_vector_store():
                    st.success("Vector store initialized successfully!")
                    st.rerun()

        with col2:
            if st.button("🗑️ Delete Vector Store",
                         key="delete_vector",
                         disabled=not vector_store_exists or st.session_state.setup_in_progress,
                         help="Delete existing vector store"):
                if delete_vector_store():
                    st.success("Vector store deleted successfully!")
                    st.rerun()

        if st.button("🤖 Initialize Chatbot",
                     key="init_chatbot",
                     disabled=st.session_state.setup_in_progress or not vector_store_exists or st.session_state.chatbot_initialized,
                     help="Initialize the chatbot with the vector store"):
            if initialize_chatbot():
                st.success("Chatbot initialized successfully!")
                st.rerun()

        st.markdown("---")
        display_sample_questions()
        st.markdown("---")

        # System Status
        st.markdown("### System Status")
        if st.session_state.vector_initialized:
            st.success("✓ Vector Store: Initialized")
        else:
            st.warning("⚠ Vector Store: Not Initialized")

        if st.session_state.chatbot_initialized:
            st.success("✓ Chatbot: Ready")
        else:
            st.warning("⚠ Chatbot: Not Ready")

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This chatbot can answer questions about:
        - Indian Constitution
        - Indian Penal Code (IPC)
        - Legal provisions and articles

        **Instructions:**
        1. First, initialize the vector store
        2. Then, initialize the chatbot
        3. Start asking questions!

        **Technology:**
        - Local FAISS vector storage
        - Google Gemini AI
        - Streamlit interface
        """)

    # Main chat interface
    st.subheader("Chat Interface")

    # Display chat history
    display_chat_history()

    # Display quick actions only if setup is complete
    if st.session_state.setup_complete:
        display_quick_actions()

    # Setup status
    if not st.session_state.setup_complete:
        st.info(
            "💡 **Getting Started:** Please initialize the vector store and chatbot using the sidebar setup buttons first.")

    # Show current status
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.vector_initialized:
            st.success("Vector store is initialized")
        else:
            st.warning("Vector store not initialized")

    with col2:
        if st.session_state.chatbot_initialized:
            st.success("Chatbot is ready")
        else:
            st.warning("Chatbot not ready")

    # Check if we have a user input from buttons
    if st.session_state.user_input:
        prompt = st.session_state.user_input
        # Clear the input immediately to prevent reprocessing
        st.session_state.user_input = ""
        # Process the question
        process_question(prompt)

    # Regular chat input
    if prompt := st.chat_input("Ask your question about Constitution or IPC..."):
        process_question(prompt)


if __name__ == "__main__":
    main()