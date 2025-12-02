
import os
import time
import re
import json
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv
from vector_store import VectorStoreManager
import logging
import google.generativeai as genai
from langchain.memory import ConversationBufferWindowMemory

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


class ConversationMemory:
    """Manages conversation history for the chatbot"""

    def __init__(self, max_history=10):
        self.memory = ConversationBufferWindowMemory(
            k=max_history,
            memory_key="chat_history",
            return_messages=True
        )
        self.history_file = "conversation_history.json"

    def add_to_history(self, user_input: str, assistant_response: str):
        """Add a conversation turn to memory"""
        try:
            self.memory.save_context(
                {"input": user_input},
                {"output": assistant_response}
            )
            # Also save to file
            self._save_to_file(user_input, assistant_response)
        except Exception as e:
            logger.error(f"Error saving to history: {e}")

    def get_history(self) -> List[Tuple[str, str]]:
        """Get conversation history as formatted string"""
        try:
            memory_dict = self.memory.load_memory_variables({})
            chat_history = memory_dict.get("chat_history", [])

            history_text = []
            for i in range(0, len(chat_history), 2):
                if i + 1 < len(chat_history):
                    human_msg = chat_history[i].content
                    ai_msg = chat_history[i + 1].content
                    history_text.append(f"Human: {human_msg}")
                    history_text.append(f"Assistant: {ai_msg}")

            return "\n".join(history_text[-10:])  # Return last 5 exchanges
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return ""

    def get_formatted_history_for_prompt(self) -> str:
        """Get formatted history for inclusion in prompts"""
        history = self.get_history()
        if history:
            return f"\n\n**CONVERSATION HISTORY:**\n{history}\n"
        return ""

    def clear_history(self):
        """Clear conversation history"""
        self.memory.clear()
        # Clear file
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
        logger.info("Conversation history cleared")

    def _save_to_file(self, user_input: str, assistant_response: str):
        """Save conversation to file for persistence"""
        try:
            history = []
            if os.path.exists(self.history_file):
                try:
                    with open(self.history_file, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                except:
                    history = []

            history.append({
                "timestamp": time.time(),
                "user": user_input,
                "assistant": assistant_response
            })

            # Keep only last 50 conversations
            if len(history) > 50:
                history = history[-50:]

            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving to file: {e}")


class SmartLegalChatbot:
    """Smart chatbot that uses Pinecone RAG when possible, otherwise uses Gemini directly"""

    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.client = None
        self.model_name = "gemini-2.0-flash"
        self._initialized = False
        self.last_call_time = 0
        self.min_call_interval = 1.0
        self.memory = ConversationMemory(max_history=10)

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

    def query(self, question, use_history=True):
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
            docs = self.vector_store.similarity_search(question, k=8)  # Get more docs for better context

            # STEP 2: Check if documents contain the specific information needed
            # For legal queries, we need to check if the documents actually contain the specific article/section
            query_lower = question.lower()
            is_specific_legal_query = self._is_specific_legal_query(query_lower)

            if docs and is_specific_legal_query:
                # Check if the specific article/section is mentioned in the documents
                if not self._contains_specific_legal_info(query_lower, docs):
                    logger.info(f"Specific legal reference not found in documents, using direct Gemini...")
                    response = self._direct_gemini_approach(question, use_history)
                else:
                    logger.info(f"Found relevant legal documents, using enhanced RAG...")
                    response = self._enhanced_rag_approach(question, docs, use_history)
            elif docs:
                # For general questions, use enhanced RAG if we have documents
                logger.info(f"Using enhanced RAG with {len(docs)} documents...")
                response = self._enhanced_rag_approach(question, docs, use_history)
            else:
                # No documents found at all
                logger.info("No documents found, using direct Gemini...")
                response = self._direct_gemini_approach(question, use_history)

            # Add to conversation history
            if use_history:
                self.memory.add_to_history(question, response["answer"])

            return response

        except Exception as e:
            logger.error(f"Query failed: {e}")
            response = self._fallback_approach(question)
            if use_history:
                self.memory.add_to_history(question, response["answer"])
            return response

    def _is_specific_legal_query(self, query: str) -> bool:
        """Check if the query is asking for a specific legal article/section"""
        legal_indicators = [
            'article', 'section', 'art.', 'sec.',
            'article 14', 'article 21', 'article 19',
            'section 302', 'section 420', 'section 124a',
            'fundamental right', 'directive principle',
            'indian constitution', 'ipc', 'indian penal code'
        ]

        query_lower = query.lower()
        for indicator in legal_indicators:
            if indicator in query_lower:
                return True

        # Check for patterns like "Article X" or "Section X"
        if re.search(r'article\s+\d+', query_lower) or re.search(r'section\s+\d+', query_lower):
            return True

        return False

    def _contains_specific_legal_info(self, query: str, docs: List) -> bool:
        """Check if the documents contain the specific legal article/section mentioned in the query"""
        # Extract specific article/section numbers from query
        query_lower = query.lower()

        # Look for article numbers
        article_matches = re.findall(r'article\s+(\d+[a-zA-Z]*)', query_lower)
        # Look for section numbers
        section_matches = re.findall(r'section\s+(\d+[a-zA-Z]*)', query_lower)

        specific_refs = article_matches + section_matches

        if not specific_refs:
            # If no specific reference numbers, check for general legal terms
            general_legal_terms = ['constitution', 'ipc', 'fundamental right', 'directive principle']
            for term in general_legal_terms:
                if term in query_lower:
                    # For general terms, assume documents might contain them
                    return True
            return False

        # Check each document for the specific references
        for doc in docs:
            content_lower = doc.page_content.lower()
            for ref in specific_refs:
                # Look for the reference in the document
                if re.search(rf'article\s+{ref}\b', content_lower) or \
                        re.search(rf'section\s+{ref}\b', content_lower) or \
                        re.search(rf'art\.?\s*{ref}\b', content_lower) or \
                        re.search(rf'sec\.?\s*{ref}\b', content_lower):
                    logger.info(f"Found reference {ref} in document")
                    return True

        logger.info(f"Specific legal references {specific_refs} not found in documents")
        return False

    def _enhanced_rag_approach(self, question, docs, use_history=True):
        """Use enhanced RAG approach with the found documents - more creative and comprehensive"""
        try:
            context = self._prepare_enhanced_context(docs)

            # Add conversation history if enabled
            conversation_context = ""
            if use_history:
                conversation_context = self.memory.get_formatted_history_for_prompt()

            prompt = self._create_enhanced_rag_prompt(question, context, conversation_context)

            response = self.client.generate_content(prompt)
            self.last_call_time = time.time()

            answer = response.text

            # Check if Gemini says it can't answer based on documents
            if self._is_rag_failure_response(answer):
                logger.info("RAG response indicates failure, switching to direct Gemini...")
                return self._direct_gemini_approach(question, use_history)

            formatted_answer = self._format_enhanced_rag_answer(answer, docs)

            logger.info("✓ Enhanced RAG approach successful")

            return {
                "answer": formatted_answer,
                "source_documents": docs
            }

        except Exception as e:
            logger.error(f"Enhanced RAG approach failed: {e}")
            # Fallback to direct approach
            return self._direct_gemini_approach(question, use_history)

    def _prepare_enhanced_context(self, docs):
        """Prepare enhanced context from documents with more information"""
        context_parts = ["**📚 RELEVANT LEGAL DOCUMENTS FROM DATABASE:**\n"]

        for i, doc in enumerate(docs):
            page = doc.metadata.get('page', 'N/A')
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content

            # Clean the content
            content = re.sub(r'\s+', ' ', content).strip()

            # Extract key sentences for better context
            sentences = re.split(r'[.!?]', content)
            key_sentences = []

            # Look for legal keywords to prioritize
            legal_keywords = ['shall', 'punish', 'liable', 'offence', 'imprisonment',
                              'fine', 'court', 'judge', 'act', 'section', 'article',
                              'constitution', 'right', 'duty', 'power', 'penalty']

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 30:  # Avoid very short sentences
                    # Check if sentence contains legal keywords
                    if any(keyword in sentence.lower() for keyword in legal_keywords):
                        key_sentences.append(sentence)
                    elif len(key_sentences) < 3:  # Add some non-keyword sentences for context
                        key_sentences.append(sentence)

            # Limit to 5 key sentences
            key_sentences = key_sentences[:5]

            if key_sentences:
                context_entry = f"**📄 Document {i + 1}** (Page {page}):\n"
                for j, sentence in enumerate(key_sentences):
                    context_entry += f"  {j + 1}. {sentence}.\n"
                context_parts.append(context_entry)

        return "\n\n".join(context_parts)

    def _create_enhanced_rag_prompt(self, question, context, conversation_context=""):
        """Create enhanced prompt for RAG approach - more comprehensive and creative"""
        return f"""# ROLE: You are an expert legal assistant specializing in Indian Constitution and Indian Penal Code (IPC).

## OBJECTIVE: 
Provide a comprehensive, detailed, and insightful answer to the legal question below.

## CONTEXT FROM LEGAL DATABASE:
{context}

## CONVERSATION HISTORY:
{conversation_context}

## USER QUESTION:
"{question}"

## ANSWER REQUIREMENTS:

### 1. **STRUCTURE YOUR ANSWER:**
   - Start with a clear, engaging introduction
   - Provide detailed explanation with relevant legal provisions
   - Include practical implications and real-world applications
   - Add historical context or landmark cases if relevant
   - Conclude with a summary

### 2. **CONTENT GUIDELINES:**
   - **Base your answer primarily on the provided legal documents** above
   - **You can supplement** with your general legal knowledge to make the answer more comprehensive
   - **If specific details are missing** from the documents, you can carefully add relevant general knowledge
   - **Always cite** which parts come from the documents vs. general knowledge
   - **Be creative** in your explanations while maintaining accuracy
   - **Use analogies** and examples to make complex legal concepts understandable
   - **Add practical insights** about how the law works in real cases

### 3. **FORMATTING:**
   - Use clear headings and bullet points for readability
   - Bold important legal terms and provisions
   - Use emojis sparingly for visual appeal
   - Include section breaks for different parts

### 4. **SPECIFIC INSTRUCTIONS FOR THIS QUESTION:**
   - If the question asks about a specific article/section, quote it exactly from the documents if available
   - Explain its significance, interpretation, and application
   - Mention related provisions if relevant
   - Discuss any important judicial interpretations or landmark cases
   - Address common misconceptions if any

## EXAMPLE OF A GOOD ANSWER:
For "What is Section 302 of IPC?":
- **Introduction:** Overview of murder in Indian law
- **Legal Text:** Exact wording from IPC
- **Explanation:** Detailed breakdown of elements
- **Punishment:** Sentencing guidelines and options
- **Related Sections:** Connection to Section 299, 300, 303, 304
- **Landmark Cases:** Important Supreme Court interpretations
- **Practical Insights:** How courts determine murder vs culpable homicide
- **Conclusion:** Summary and importance

## YOUR COMPREHENSIVE ANSWER:
"""

    def _create_direct_prompt(self, question, conversation_context=""):
        """Create prompt for direct Gemini approach"""
        return f"""# ROLE: You are an expert legal assistant specializing in Indian Constitution and Indian Penal Code (IPC).

## CONTEXT:
This question is being answered from your general knowledge since the specific legal provisions were not found in the database.

## CONVERSATION HISTORY:
{conversation_context}

## USER QUESTION:
"{question}"

## ANSWER REQUIREMENTS:

### 1. **ACKNOWLEDGE SOURCE:**
   - Clearly state that this answer comes from general legal knowledge
   - Explain why database documents were insufficient

### 2. **PROVIDE COMPREHENSIVE ANSWER:**
   - Give detailed, accurate information about the legal topic
   - Include specific Articles/Sections with exact wording if known
   - Add historical context and significance
   - Mention important judicial interpretations
   - Discuss practical applications and real-world implications

### 3. **STRUCTURE:**
   - Engaging introduction
   - Detailed legal analysis
   - Practical insights
   - Related provisions
   - Summary and conclusion

### 4. **FORMATTING:**
   - Use clear headings and bullet points
   - Bold important terms
   - Make it visually appealing yet professional

## YOUR COMPREHENSIVE ANSWER FROM GENERAL KNOWLEDGE:
"""

    def _is_rag_failure_response(self, answer: str) -> bool:
        """Check if the RAG response indicates it couldn't find the information"""
        failure_indicators = [
            "the provided legal documents do not contain",
            "i cannot provide information based solely on these documents",
            "not found in the provided documents",
            "the documents do not mention",
            "based on the provided documents, i cannot",
            "the text is not present in the documents",
            "no information about",
            "does not contain the text of",
            "cannot find information about",
            "unable to answer from the documents"
        ]

        answer_lower = answer.lower()
        for indicator in failure_indicators:
            if indicator in answer_lower:
                return True
        return False

    def _direct_gemini_approach(self, question, use_history=True):
        """Use Gemini directly without document context"""
        try:
            # Add conversation history if enabled
            conversation_context = ""
            if use_history:
                conversation_context = self.memory.get_formatted_history_for_prompt()

            prompt = self._create_direct_prompt(question, conversation_context)

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

    def _format_enhanced_rag_answer(self, answer, docs):
        """Format enhanced RAG answer with better presentation"""
        # Extract pages for citation
        unique_pages = sorted(list(set(doc.metadata.get('page', 'N/A') for doc in docs)))

        formatted_answer = f"{answer}\n\n"
        formatted_answer += "---\n"
        formatted_answer += "**📊 Answer Analysis**\n\n"
        formatted_answer += "| Aspect | Details |\n"
        formatted_answer += "|--------|---------|\n"
        formatted_answer += f"| **Source** | Pinecone Legal Database |\n"
        formatted_answer += f"| **Documents Used** | {len(docs)} relevant documents |\n"
        formatted_answer += f"| **Key Pages** | {', '.join(map(str, unique_pages[:5]))} |\n"
        formatted_answer += "| **Answer Type** | Document-based with supplemental insights |\n"
        formatted_answer += "| **Confidence** | High (based on legal texts) |\n\n"
        formatted_answer += "*Note: This answer is primarily based on the legal documents in the database, with additional insights for comprehensiveness.*"

        return formatted_answer

    def _format_direct_answer(self, answer, question):
        """Format direct Gemini answer"""
        formatted_answer = f"{answer}\n\n"
        formatted_answer += "---\n"
        formatted_answer += "**🔍 Information Source Details**\n\n"
        formatted_answer += "| Aspect | Details |\n"
        formatted_answer += "|--------|---------|\n"
        formatted_answer += "| **Source** | Google Gemini General Knowledge |\n"
        formatted_answer += "| **Model** | Gemini 2.0 Flash |\n"
        formatted_answer += "| **Reason** | Specific legal provision not found in database |\n"
        formatted_answer += "| **Accuracy** | Based on established legal knowledge |\n"
        formatted_answer += "| **Verification** | Cross-reference with official sources recommended |\n\n"
        formatted_answer += "*Note: For official legal advice, consult a qualified legal professional.*"

        return formatted_answer

    def _fallback_approach(self, question):
        """Ultimate fallback approach"""
        try:
            # Try one more time with direct Gemini as last resort
            try:
                prompt = f"""As an expert legal assistant, provide a comprehensive answer to this question about Indian law:

                Question: {question}

                Structure your answer with:
                1. Introduction and context
                2. Legal provisions (if known)
                3. Explanation and analysis
                4. Practical implications
                5. Conclusion"""

                response = self.client.generate_content(prompt)
                answer = response.text

                return {
                    "answer": f"**🔎 Legal Analysis (General Knowledge):**\n\n{answer}\n\n---\n*Note: This answer is based on general legal knowledge. For specific cases, consult official legal resources.*",
                    "source_documents": []
                }
            except:
                # Final fallback
                return {
                    "answer": f"**⚖️ Legal Assistant Response**\n\n**Question:** {question}\n\n**Response:** While I couldn't access specific legal documents from the database for this query, I can provide general guidance:\n\n1. **For Constitution-related questions:** Refer to the official Constitution of India text\n2. **For IPC queries:** Consult the Indian Penal Code, 1860\n3. **For specific sections:** Look up the exact wording in legal databases\n4. **For interpretation:** Review Supreme Court judgments on the topic\n\n**Recommended Resources:**\n- India Code (indiacode.nic.in)\n- Supreme Court of India website\n- Official government legal portals\n\n*Please try rephrasing your question or ask about broader legal concepts.*",
                    "source_documents": []
                }

        except Exception as e:
            return {
                "answer": "**⚠️ System Temporarily Unavailable**\n\nOur legal database is currently experiencing technical difficulties. Please try again in a few moments.\n\nIn the meantime, you may want to:\n1. Check your internet connection\n2. Try a different question\n3. Return later when the system is restored",
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

    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.memory.clear_history()
        logger.info("Conversation history cleared")

    def get_conversation_history(self):
        """Get current conversation history"""
        return self.memory.get_history()


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

    def query(self, question, use_history=True):
        """Query the chatbot"""
        return self.chatbot.query(question, use_history)

    def clear_conversation_history(self):
        """Clear conversation history"""
        self.chatbot.clear_conversation_history()

    def get_conversation_history(self):
        """Get conversation history"""
        return self.chatbot.get_conversation_history()


# Test function
def test_smart_chatbot():
    """Test the smart chatbot with conversation memory"""
    try:
        chatbot = get_chatbot()
        success = chatbot.initialize_qa_chain()

        if success:
            # Test enhanced RAG approach
            test_questions = [
                "What are the consequences of murder under IPC?",
                "Explain Article 14 of the Indian Constitution",
                "What is Section 420 of IPC?"
            ]

            results = ["✅ **Enhanced Legal Chatbot - Working!**\n"]
            results.append("**Features:** Enhanced RAG + Smart Fallback + Creative Answers")

            for i, question in enumerate(test_questions[:2]):  # Test first 2
                response = chatbot.query(question)
                if response['source_documents']:
                    approach = "Enhanced RAG"
                else:
                    approach = "Direct Gemini (General Knowledge)"
                results.append(f"\n**Test {i + 1} ({approach}):**")
                results.append(f"**Q:** {question}")
                results.append(f"**A Preview:** {response['answer'][:250]}...")
                results.append(f"**Sources:** {len(response['source_documents'])} documents")

            return "\n".join(results)
        else:
            return "❌ Chatbot initialization failed"

    except Exception as e:
        return f"❌ Test failed: {str(e)}"


if __name__ == "__main__":
    print("Testing Enhanced Legal Chatbot with Creative RAG...")
    print(test_smart_chatbot())
