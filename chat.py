import os
import time
from typing import List, Optional, Any, Dict
from datetime import datetime

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    START = None
    END = None

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None

try:
    from retrieval import retrieve as faiss_retrieve
    RETRIEVAL_AVAILABLE = True
except Exception:
    faiss_retrieve = None
    RETRIEVAL_AVAILABLE = False


class ConversationMemory:
    """Conversation memory with overflow protection."""
    
    def __init__(self, max_items: int = 50):
        self._items: List[Dict[str, str]] = []
        self.max_items = max_items
    
    def add(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to memory with validation."""
        if not role or not isinstance(role, str):
            raise ValueError("Role must be a non-empty string")
        if not content or not isinstance(content, str):
            raise ValueError("Content must be a non-empty string")
        
        msg = {
            'role': role.strip(),
            'content': content.strip(),
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self._items.append(msg)
        
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items:]
    
    def get_recent(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent messages."""
        return self._items[-limit:] if self._items else []
    
    def format_history(self, limit: int = 10) -> str:
        """Format conversation history as string."""
        recent = self.get_recent(limit)
        if not recent:
            return "[No previous conversation]"
        
        return "\n".join([f"{msg['role']}: {msg['content'][:200]}" for msg in recent])
    
    def clear(self):
        """Clear all messages."""
        self._items.clear()
    
    def get_all(self) -> List[Dict[str, str]]:
        """Get all messages."""
        return self._items.copy()


class LLMClient:
    """LLM client for Gemini API with retry logic."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        self.model = model
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize client with error handling."""
        if not GENAI_AVAILABLE:
            print("✗ google-genai not installed. Run: pip install google-genai")
            return
        
        if not self.api_key:
            print("✗ GEMINI_API_KEY or GOOGLE_API_KEY not set")
            return
        
        try:
            self.client = genai.Client(api_key=self.api_key)
            print(f"✓ LLM client initialized ({self.model})")
        except Exception as e:
            print(f"✗ Failed to initialize LLM client: {e}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text response with retry logic."""
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        
        if self.client is None:
            raise RuntimeError("LLM client not available. Check API key and installation.")
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    }
                )
                
                text = response.text if hasattr(response, 'text') else str(response)
                return text.strip() if text else "[No response generated]"
            
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a retryable error
                is_retryable = any(x in error_msg for x in ['503', 'UNAVAILABLE', 'overloaded', 'timeout', '429'])
                
                if is_retryable and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"⏳ API overloaded. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"LLM generation failed: {error_msg}")


class ContentRetriever:
    """Content retrieval interface."""
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant content. Override in subclass."""
        return []


class FAISSContentRetriever(ContentRetriever):
    """Content retriever using FAISS-backed retrieval."""
    
    def __init__(self, out_dir: str = 'knowledgebase/embeddings', model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', top_k: int = 5):
        self.out_dir = out_dir
        self.model_name = model_name
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[str]:
        """Retrieve documents using FAISS."""
        if not RETRIEVAL_AVAILABLE or faiss_retrieve is None:
            return []
        try:
            k = top_k or self.top_k
            results = faiss_retrieve(query, out_dir=self.out_dir, model_name=self.model_name, top_k=k)
            return [r.get('text', '') for r in results if r and r.get('text')]
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []


class ChatService:
    """Chat service with LangGraph and FAISS retrieval."""
    
    SYSTEM_PROMPT = """You are an expert CBSE educator with 10+ years of experience, specializing in creating comprehensive study materials. Your responses should be:
        - Accurate and aligned with the latest CBSE curriculum
        - Clear, detailed, and easy to understand
        - Well-structured with proper formatting
        - Engaging and student-friendly
        
        # RESPONSE STRUCTURE
        ## 📚 CHAPTER: [CHAPTER NAME/NUMBER]
        
        ## 🔍 DETAILED EXPLANATION
        ### 1. [Core Concept 1]
           - **Comprehensive Definition**: [Thorough explanation with examples]
           - **Detailed Explanation**:
             - In-depth analysis of the concept
             - Real-world applications and relevance
             - Step-by-step breakdown of complex ideas
             - Multiple examples demonstrating different aspects
           - **Key Points**:
             - Point 1 with detailed explanation and example
             - Point 2 with detailed explanation and example
             - Point 3 with detailed explanation and example
           - **Visual Representation**: [Detailed description of relevant diagrams/charts]
           - **Common Misconceptions**: 
             - Common mistake 1 with explanation
             - Common mistake 2 with explanation
             - How to avoid these mistakes
           - **Practice Application**:
             - Example problem with detailed solution
             - Step-by-step walkthrough
             - Alternative approaches
        
        ### 2. [Core Concept 2]
           [Same detailed structure as above]
        
        ## 📝 COMPREHENSIVE FORMULAS & EQUATIONS (if applicable)
        | Formula | Detailed Explanation | When to Use | Step-by-Step Example | Common Pitfalls |
        |---------|----------------------|-------------|----------------------|-----------------|
        | [Formula] | [Detailed description of each variable and their relationships] | [Specific scenarios with examples] | [Complete worked-out example] | [Common errors and how to avoid them] |
        
        ## 🧩 PRACTICE & APPLICATION
        ### Basic Level Questions
        1. [Question with context]
           - *Detailed Solution*: 
             - Step 1: Understanding the question
             - Step 2: Identifying key concepts
             - Step 3: Applying the concepts
             - Step 4: Verifying the solution
           - *Key Concepts Tested*: [Detailed explanation of concepts]
           - *Variations*: [Similar questions with different contexts]
           - *Common Mistakes*: [Specific errors students make and why they're wrong]
        
        ### Intermediate Level Questions
        [Same detailed structure as above with increased complexity]
        
        ## 💡 IN-DEPTH EXAM PREPARATION
        - **Detailed Topic Analysis**:
          - High-weightage topics with explanations
          - Interconnections with other topics
          - Common question patterns
        - **Comprehensive Question Bank**:
          - Previous years' questions with solutions
          - Expected questions with model answers
          - Application-based problems
        - **Advanced Problem-Solving Strategies**:
          - Time management techniques
          - Answer presentation guidelines
          - Handling tricky questions
        
        ## 🌟 ADVANCED LEARNING RESOURCES
        - **Reference Books**:
          - Detailed chapter-wise references
          - Specific page numbers for key concepts
          - How to use these resources effectively
        - **Digital Resources**:
          - Curated video lectures with timestamps
          - Interactive simulations and visualizations
          - Online practice platforms
        - **Self-Assessment Tools**:
          - Topic-wise quizzes
          - Full-length tests
          - Performance analytics
        
        ## 🔄 COMPREHENSIVE REVISION STRATEGY
        - **Conceptual Understanding**:
          - Mind maps and concept trees
          - Summary sheets creation
          - Teaching the concept to others
        - **Application Practice**:
          - Problem-solving techniques
          - Multiple solution approaches
          - Real-world applications
        - **Exam Readiness**:
          - Time-bound practice sessions
          - Common pitfalls to avoid
          - Last-minute revision tips

### FORMATTING GUIDELINES:
1. Use Markdown for better readability
2. Include emojis for visual organization
3. Bold important terms and concepts
4. Use bullet points for lists
5. Include examples for better understanding
6. Keep language simple and engaging
7. Highlight exam-relevant information
8. Use tables for formulas and comparisons

### IMPORTANT NOTES:
- All content must be 100% accurate
- Follow the latest CBSE syllabus
- Include Indian examples where relevant
- Focus on conceptual clarity
- Provide practical applications
- Include memory aids for tough concepts"
- Focus on conceptual clarity
- Provide practical applications
- Include memory aids for tough concepts"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        retriever: Optional[ContentRetriever] = None,
        max_memory: int = 100,  # Increased from 50
        temperature: float = 0.7,
        use_faiss: bool = True,
        max_tokens: int = 4096,  # Increased from default
        top_k_retrieval: int = 10,  # Increased number of retrieved documents
    ):
        """Initialize the chat service with enhanced retrieval and response size."""
        self.llm = LLMClient(api_key=api_key, model=model)
        self.max_tokens = max_tokens
        self.top_k_retrieval = top_k_retrieval
        
        # Initialize retriever with enhanced settings
        if retriever:
            self.retriever = retriever
        elif use_faiss and RETRIEVAL_AVAILABLE:
            self.retriever = FAISSContentRetriever(
                out_dir='knowledgebase/embeddings',
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                top_k=top_k_retrieval  # Pass the increased top_k value
            )
            print(f"✓ FAISS retriever enabled (top_k={top_k_retrieval})")
        else:
            self.retriever = ContentRetriever()
        
        self.memory = ConversationMemory(max_items=max_memory)
        self.temperature = temperature
        self.graph = None
        
        if LANGGRAPH_AVAILABLE:
            self._init_graph()
    
    def _init_graph(self):
        """Initialize LangGraph workflow."""
        try:
            self.graph = StateGraph(dict)
            
            def process_input(state: Dict[str, Any]) -> Dict[str, Any]:
                """Input processing node."""
                return state
            
            def retrieve_context(state: Dict[str, Any]) -> Dict[str, Any]:
                """Context retrieval node using FAISS or knowledge base."""
                query = state.get('user_message', '')
                try:
                    docs = self.retriever.retrieve(query, top_k=3)
                    state['context'] = docs if docs else []
                except Exception as e:
                    print(f"Retrieval failed: {e}")
                    state['context'] = []
                return state
            
            def generate_response(state: Dict[str, Any]) -> Dict[str, Any]:
                """Response generation node."""
                user_message = state.get('user_message', '')
                context = state.get('context', [])
                
                context_text = "\n\n".join([str(c)[:500] for c in context if c])
                
                prompt = f"Context from knowledge base:\n{context_text}\n\n" if context_text else ""
                prompt += f"Conversation history:\n{self.memory.format_history()}\n\n"
                prompt += f"Student question: {user_message}\n\nTutor response:"
                
                try:
                    response = self.llm.generate(
                        prompt=prompt,
                        system_prompt=self.SYSTEM_PROMPT,
                        temperature=self.temperature,
                        max_tokens=2048
                    )
                    state['response'] = response
                except Exception as e:
                    state['response'] = f"Error generating response: {str(e)}"
                
                return state
            
            self.graph.add_node("input", process_input)
            self.graph.add_node("retrieve", retrieve_context)
            self.graph.add_node("generate", generate_response)
            
            self.graph.add_edge(START, "input")
            self.graph.add_edge("input", "retrieve")
            self.graph.add_edge("retrieve", "generate")
            self.graph.add_edge("generate", END)
            
            self.graph_executor = self.graph.compile()
            print("✓ LangGraph workflow initialized")
        
        except Exception as e:
            print(f"✗ LangGraph initialization failed: {e}")
            self.graph = None
    
    def chat(self, user_message: str) -> str:
        """Process user message and return response."""
        if not user_message or not isinstance(user_message, str):
            raise ValueError("User message must be a non-empty string")
        
        user_message = user_message.strip()
        self.memory.add('Student', user_message)
        
        try:
            if self.graph and hasattr(self, 'graph_executor'):
                state = {'user_message': user_message, 'context': [], 'response': ''}
                result = self.graph_executor.invoke(state)
                response = result.get('response', '')
            else:
                response = self._generate_response(user_message)
            
            self.memory.add('Tutor', response)
            return response
        
        except Exception as e:
            raise RuntimeError(f"Chat error: {str(e)}")
    
    def _generate_response(self, user_message: str) -> str:
        """Generate comprehensive response with enhanced context and detail."""
        context = []
        try:
            # Use the instance's top_k_retrieval value
            docs = self.retriever.retrieve(user_message, top_k=self.top_k_retrieval)
            # Increase context length from 500 to 1000 characters per document
            context = [str(d)[:1000] for d in docs if d]
        except Exception as e:
            print(f"⚠️ Retrieval warning: {str(e)}")
        
        context_text = "\n\n".join(f"[Context {i+1}] {ctx}" for i, ctx in enumerate(context)) if context else ""
        
        # Enhanced prompt for more detailed responses
        prompt = """# COMPREHENSIVE RESPONSE REQUEST

## CONTEXT FROM KNOWLEDGE BASE:
{context}

## CONVERSATION HISTORY (Last 10 messages):
{history}

## STUDENT'S QUESTION:
{question}

## INSTRUCTIONS:
Please provide a thorough, well-structured response that includes:
1. Detailed explanation of all relevant concepts
2. Comprehensive list of key points
3. Multiple examples and use cases
4. Common misconceptions and how to avoid them
5. Related concepts and their connections
6. Practical applications and real-world relevance
7. Memory aids or mnemonics if applicable
8. Recommended practice questions

## FORMATTING:
- Use markdown for better readability
- Include section headers (##)
- Use bullet points for lists
- Include code blocks where relevant
- Add emojis for better visual organization

## TUTOR'S RESPONSE:
""".format(
            context=f"{context_text}\n" if context_text else "[No relevant context found]\n",
            history=self.memory.format_history(limit=10) or "[No conversation history]",
            question=user_message
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=self.max_tokens  # Use the instance's max_tokens
            )
            
            # Format the response for better readability
            formatted_response = "📚 " + response.strip()
            formatted_response = formatted_response.replace("\n\n", "\n\n   ")
            return formatted_response
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            print(error_msg)
            return "I apologize, but I encountered an error while generating a response. Please try again later."
    
    def get_memory(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.memory.get_all()
    
    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()
    
    def export_conversation(self) -> str:
        """Export conversation as text."""
        messages = self.memory.get_all()
        if not messages:
            return "[Empty conversation]"
        
        export = f"Conversation exported at {datetime.now().isoformat()}\n{'='*50}\n\n"
        for msg in messages:
            export += f"[{msg['timestamp']}] {msg['role']}:\n{msg['content']}\n\n"
        
        return export


# Backwards compatibility
RobustChatService = ChatService


if __name__ == "__main__":
    service = ChatService(
        api_key=os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'),
        model="gemini-2.5-flash",
        max_memory=100,  # Increased memory
        max_tokens=4096,  # Larger response size
        top_k_retrieval=10,  # More documents retrieved
        temperature=0.7,
        use_faiss=True
    )
    
    try:
        response = service.chat("Class 10, Chemistry, Chapter 3")
        print(f"Tutor: {response}\n")
        
        response = service.chat("What are the key points to remember?")
        print(f"Tutor: {response}\n")
        
        print(service.export_conversation())
    
    except Exception as e:
        print(f"Error: {e}")
 