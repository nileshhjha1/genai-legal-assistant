import os
import time
import re
from typing import Any, Dict, List
from dotenv import load_dotenv
from vector_store import VectorStoreManager
import logging
import google.generativeai as genai

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global chatbot instance
_chatbot_instance = None


def get_chatbot():
    """Get or create chatbot instance (singleton pattern)"""
    global _chatbot_instance
    if _chatbot_instance is None:
        logger.info("Creating new ConstitutionChatbot instance...")
        _chatbot_instance = ConstitutionChatbot()
    return _chatbot_instance


class SmartLegalChatbot:
    """Smart chatbot that uses Pinecone RAG when possible, otherwise uses Gemini directly"""

    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.client = None
        self.model_name = "gemini-2.0-flash"
        self._initialized = False
        self.last_call_time = 0
        self.min_call_interval = 1.0

    def initialize(self):
        """Initialize the chatbot"""
        if self._initialized:
            return True

        try:
            logger.info("Step 1: Loading Pinecone vector store...")

            # Check if Pinecone index exists
            if not self.vector_manager.check_index_exists():
                raise Exception("Pinecone index not found. Please initialize vector store first.")

            # Get the Pinecone vector store
            self.vector_store = self.vector_manager.get_vector_store()
            if self.vector_store is None:
                raise Exception("Pinecone vector store could not be loaded")
            logger.info("✓ Pinecone vector store loaded successfully")

            logger.info("Step 2: Initializing Gemini client...")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")

            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model_name)

            # Test the connection
            test_response = self.client.generate_content("Say hello")
            logger.info(f"✓ Gemini initialized successfully with model: {self.model_name}")

            self._initialized = True
            logger.info("✓ Smart Legal Chatbot fully initialized")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def query(self, question):
        """Smart query: Try RAG first, if no relevant docs found, use Gemini directly"""
        if not self._initialized:
            success = self.initialize()
            if not success:
                return {
                    "answer": "**❌ System Initialization Failed**\n\nPlease check your setup and try again.",
                    "source_documents": []
                }

        try:
            # Rate limiting
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            if time_since_last_call < self.min_call_interval:
                time.sleep(self.min_call_interval - time_since_last_call)

            logger.info(f"Processing question: {question}")

            # STEP 1: Search Pinecone for relevant documents
            logger.info("Searching Pinecone for relevant documents...")
            docs = self.vector_store.similarity_search(question, k=6)

            # Check if we found highly relevant documents
            has_relevant_docs = self._has_relevant_documents(question, docs)

            if has_relevant_docs:
                logger.info("Found relevant documents, using RAG approach...")
                # Use RAG approach with the found documents
                return self._rag_approach(question, docs)
            else:
                logger.info("No highly relevant documents found, using direct Gemini approach...")
                # Use direct Gemini approach
                return self._direct_gemini_approach(question)

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return self._fallback_approach(question)

    def _has_relevant_documents(self, question, docs):
        """Always use RAG if we have documents from the vector store"""
        if not docs:
            return False

        # If we found documents in the vector store, they're relevant enough
        logger.info(f"Using RAG with {len(docs)} documents found")
        return True

    def _rag_approach(self, question, docs):
        """Use RAG approach with the found documents"""
        try:
            context = self._prepare_context(docs)
            prompt = self._create_rag_prompt(question, context)

            response = self.client.generate_content(prompt)
            self.last_call_time = time.time()

            answer = response.text
            formatted_answer = self._format_rag_answer(answer, docs)

            logger.info("✓ RAG approach successful")

            return {
                "answer": formatted_answer,
                "source_documents": docs
            }

        except Exception as e:
            logger.error(f"RAG approach failed: {e}")
            # Fallback to direct approach
            return self._direct_gemini_approach(question)

    def _direct_gemini_approach(self, question):
        """Use Gemini directly without document context"""
        try:
            prompt = self._create_direct_prompt(question)

            response = self.client.generate_content(prompt)
            self.last_call_time = time.time()

            answer = response.text
            formatted_answer = self._format_direct_answer(answer, question)

            logger.info("✓ Direct Gemini approach successful")

            return {
                "answer": formatted_answer,
                "source_documents": []  # No source documents for direct approach
            }

        except Exception as e:
            logger.error(f"Direct Gemini approach failed: {e}")
            return self._fallback_approach(question)

    def _prepare_context(self, docs):
        """Prepare context from documents"""
        context_parts = ["**RELEVANT LEGAL DOCUMENTS FROM DATABASE:**\n"]

        for i, doc in enumerate(docs):
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content

            # Clean the content
            content = re.sub(r'\s+', ' ', content).strip()

            # Limit content length
            if len(content) > 400:
                content = content[:400] + "..."

            context_entry = f"**Document {i + 1}** (Page {page}): {content}"
            context_parts.append(context_entry)

        return "\n\n".join(context_parts)

    def _create_rag_prompt(self, question, context):
        """Create prompt for RAG approach"""
        return f"""You are an expert legal assistant for Indian Constitution and IPC.

**LEGAL DOCUMENTS CONTEXT:**
{context}

**USER QUESTION:**
{question}

**INSTRUCTIONS:**
Answer the question using the legal documents provided above. Provide a comprehensive legal analysis with proper citations.

**ANSWER:**
"""

    def _create_direct_prompt(self, question):
        """Create prompt for direct Gemini approach"""
        return f"""You are an expert legal assistant specializing in Indian Constitution and Indian Penal Code (IPC).

**USER QUESTION:**
{question}

**INSTRUCTIONS:**
Provide a comprehensive, accurate answer about Indian Constitution or IPC. Use your knowledge of:

- Indian Constitution (Fundamental Rights, Directive Principles, Constitutional Provisions)
- Indian Penal Code (Sections, Punishments, Legal Definitions)
- Important legal doctrines and principles
- Recent legal developments if relevant

Structure your answer with:
1. Clear definition/explanation
2. Key legal provisions (mention specific Articles/Sections)
3. Important aspects and implications
4. Practical significance

Use Markdown formatting for better readability.

**ANSWER:**
"""

    def _format_rag_answer(self, answer, docs):
        """Format RAG answer with sources"""
        # Extract pages for citation
        unique_pages = sorted(list(set(doc.metadata.get('page', 'N/A') for doc in docs)))

        formatted_answer = f"{answer}\n\n"
        formatted_answer += "---\n"
        formatted_answer += "**📚 Source Information:**\n"
        formatted_answer += f"• **Source:** Pinecone Legal Database\n"
        formatted_answer += f"• **Relevant Pages:** {', '.join(map(str, unique_pages[:5]))}\n"
        formatted_answer += f"• **Documents Analyzed:** {len(docs)}\n"

        return formatted_answer

    def _format_direct_answer(self, answer, question):
        """Format direct Gemini answer"""
        formatted_answer = f"{answer}\n\n"
        formatted_answer += "---\n"
        formatted_answer += "**💡 Note:** This answer is generated using general legal knowledge. "
        formatted_answer += "For specific document references, ensure relevant content is available in the legal database.\n"

        return formatted_answer

    def _fallback_approach(self, question):
        """Ultimate fallback approach"""
        try:
            # Simple document search as last resort
            docs = self.vector_store.similarity_search(question, k=3)

            if docs:
                # Build simple answer from documents
                answer_parts = ["**🔍 Legal Document References:**\n"]
                for doc in docs:
                    page = doc.metadata.get('page', 'N/A')
                    content = self._summarize_content(doc.page_content, 150)
                    answer_parts.append(f"• **Page {page}:** {content}")

                answer_parts.append(f"\n**Question:** {question}")
                answer_parts.append("\n*Please review the above document references for relevant information.*")

                return {
                    "answer": "\n".join(answer_parts),
                    "source_documents": docs
                }
            else:
                return {
                    "answer": f"**🤖 Legal Assistant**\n\nI understand you're asking about: \"{question}\"\n\nWhile I couldn't find specific documents in the database, this appears to be a question about Indian law. Please ensure your question relates to Indian Constitution or IPC, and try again.",
                    "source_documents": []
                }

        except Exception as e:
            return {
                "answer": "**⚠️ System Temporarily Unavailable**\n\nPlease try again in a moment.",
                "source_documents": []
            }

    def _summarize_content(self, content, max_length):
        """Summarize content for display"""
        if len(content) <= max_length:
            return content

        sentences = content.split('.')
        summary = ""
        for sentence in sentences:
            if sentence.strip() and len(summary + sentence) < max_length:
                summary += sentence + '. '
            else:
                break

        summary = summary.strip()
        if len(summary) < len(content):
            summary += ".."

        return summary


class ConstitutionChatbot:
    def __init__(self):
        self.chatbot = SmartLegalChatbot()
        self._initialized = False

    def initialize_qa_chain(self):
        """Initialize the chatbot"""
        if self._initialized:
            return True

        success = self.chatbot.initialize()
        self._initialized = success
        return success

    def query(self, question):
        """Query the chatbot"""
        return self.chatbot.query(question)


# Test function
def test_smart_chatbot():
    """Test the smart chatbot"""
    try:
        chatbot = get_chatbot()
        success = chatbot.initialize_qa_chain()

        if success:
            # Test different scenarios
            test_cases = [
                "What is Article 14?",  # Should use RAG if documents exist
                "Explain the concept of habeas corpus",  # Might use direct approach
                "What is Section 420 of IPC?"  # Should use RAG if documents exist
            ]

            results = ["✅ **Smart Legal Chatbot Working!**\n"]
            results.append("**Features:** RAG + Direct Gemini fallback")

            for i, question in enumerate(test_cases[:2]):  # Test first 2
                response = chatbot.query(question)
                approach = "RAG" if response['source_documents'] else "Direct Gemini"
                results.append(f"\n**Test {i + 1} ({approach}):**")
                results.append(f"**Q:** {question}")
                results.append(f"**A:** {response['answer'][:150]}...")
                results.append(f"**Sources:** {len(response['source_documents'])} documents")

            return "\n".join(results)
        else:
            return "❌ Chatbot initialization failed"

    except Exception as e:
        return f"❌ Test failed: {str(e)}"


if __name__ == "__main__":
    print("Testing Smart Legal Chatbot...")
    print(test_smart_chatbot())
