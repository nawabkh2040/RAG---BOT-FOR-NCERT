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
    
    SYSTEM_PROMPT = """You are an expert CBSE tutor specializing in comprehensive chapter summaries and study materials.

When a student provides a Class, Subject, and Chapter number, generate a detailed 1-2 page study guide with the following structure:

## CHAPTER OVERVIEW
- Brief introduction to the chapter
- Learning objectives and key concepts

## KEY TOPICS & CONCEPTS
- All major topics covered in the chapter
- Detailed explanations of each concept
- Relationships between different topics

## IMPORTANT POINTS FOR REVISION
- Main points students must memorize
- Key definitions and terminology
- Important rules and principles
- Common misconceptions to avoid

## IMPORTANT REACTIONS (if applicable - Chemistry/Physics)
- Chemical equations with proper balancing
- Physical reactions and processes
- Step-by-step mechanisms where relevant
- Real-world applications

## IMPORTANT FACTS & FIGURES
- Statistical data and numbers
- Historical information
- Examples and case studies
- Formulas and their applications

## QUESTION PATTERNS & EXAM TIPS
- Common question types from this chapter
- How questions are typically asked
- Tips to avoid common mistakes
- Strategies for quick problem-solving

## PRACTICE AREAS
- Types of problems/questions to expect
- Areas that frequently appear in exams
- Difficult topics that need special attention

## MNEMONICS & MEMORY AIDS (if applicable)
- Techniques to remember complex topics
- Acronyms and shortcuts
- Visual memory aids

Format your response with clear headings, bullet points, and structured information. Make it exam-focused and comprehensive enough that a student can use it for revision. Include specific examples relevant to the Indian CBSE curriculum.

Always prioritize accuracy and ensure all information aligns with the current CBSE syllabus."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        retriever: Optional[ContentRetriever] = None,
        max_memory: int = 50,
        temperature: float = 0.7,
        use_faiss: bool = True,
    ):
        """Initialize the chat service."""
        self.llm = LLMClient(api_key=api_key, model=model)
        
        # Initialize retriever
        if retriever:
            self.retriever = retriever
        elif use_faiss and RETRIEVAL_AVAILABLE:
            self.retriever = FAISSContentRetriever()
            print("✓ FAISS retriever enabled")
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
        """Generate response without LangGraph."""
        context = []
        try:
            docs = self.retriever.retrieve(user_message, top_k=3)
            context = [str(d)[:500] for d in docs if d]
        except Exception:
            pass
        
        context_text = "\n\n".join(context) if context else ""
        
        prompt = f"Context:\n{context_text}\n\n" if context_text else ""
        prompt += f"Conversation history:\n{self.memory.format_history()}\n\n"
        prompt += f"Student: {user_message}\nTutor:"
        
        return self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=2048
        )
    
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
        max_memory=50,
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