from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
import re
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import os
import weaviate
from langchain_community.vectorstores import Weaviate
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

load_dotenv()

# Define AI model (OpenAI model)
# test out o4-mini too later; o4-mini tends to hallucinate a lot
# gpt-4o seems to be best
model = ChatOpenAI(model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Weaviate client and vector store
try:
    client = weaviate.Client("http://localhost:8080")
    
    # Initialize embeddings
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Delete existing class if it exists to recreate with correct schema
    try:
        client.schema.delete_class("ChatMemory")
        print("🗑️ Deleted existing ChatMemory class")
    except:
        pass
    
    # Create schema for ChatMemory class with proper configuration
    schema = {
        "class": "ChatMemory",
        "vectorizer": "none",  # Use external embeddings
        "properties": [
            {"name": "text", "dataType": ["text"]},
            {"name": "timestamp", "dataType": ["date"]},
            {"name": "user_input", "dataType": ["text"]},
            {"name": "ai_response", "dataType": ["text"]}
        ]
    }
    
    # Create the class
    client.schema.create_class(schema)
    print("✅ Created ChatMemory class with correct schema")
    
    # Initialize vectorstore with external embeddings
    vectorstore = Weaviate(
        client, 
        "ChatMemory", 
        "text",  # text_key parameter
        embedding=embedding  # Use external embeddings
    )
    print("✅ Connected to Weaviate vector store")
except Exception as e:
    print(f"⚠️ Could not connect to Weaviate: {e}")
    vectorstore = None

# Initialize sentence transformer for re-ranking
try:
    sentence_model = SentenceTransformer("all-mpnet-base-v2")
    print("✅ Loaded sentence transformer for re-ranking")
except Exception as e:
    print(f"⚠️ Could not load sentence transformer: {e}")
    sentence_model = None

# Use a simple conversation buffer memory
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True
)

# Define the prompt template using message objects
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expressive chat working under the company ebot.bio. Use the conversation history and context of the company to provide context-aware responses. If you don't know the answer to a question, tell the user you don't have enough information to answer at this time and email Shun for more info. Keep responses not too long, and format responses to be grammatically correct."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# Create the chatbot pipeline
chain = prompt | model

# Single memory file
MEMORY_FILE = "chat_memory.json"
LONG_TERM_MEMORY_FILE = "long_term_memory.json"

def load_long_term_memory():
    """Load conversation history from JSON file."""
    if os.path.exists(LONG_TERM_MEMORY_FILE):
        try:
            with open(LONG_TERM_MEMORY_FILE, 'r') as f:
                data = json.load(f)
                # Restore memory from saved data
                for context in data.get('context', []):
                    memory.save_context(
                        {"input": context['user']},
                        {"output": context['ai']}
                    )
                print("📂 Loaded context")
                
                # Repopulate Weaviate with historical data
                if vectorstore:
                    # clear_history() # clear previous chat history
                    repopulate_weaviate_from_memory(data.get('context', []))
        except Exception as e:
            print(f"⚠️  Could not load long term memory file: {e}")
    else:
        print("🆕 No previous long term memory found.")

def load_memory_from_file(): # should not need this i think ever
    """Load conversation history from JSON file."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
                # Restore memory from saved data
                for interaction in data.get('interactions', []):
                    memory.save_context(
                        {"input": interaction['user']},
                        {"output": interaction['ai']}
                    )
                print(f"📂 Loaded {len(data.get('interactions', []))} previous interactions.")
                
                # Repopulate Weaviate with historical data
                if vectorstore:
                    repopulate_weaviate_from_memory(data.get('interactions', []), True)
        except Exception as e:
            print(f"⚠️  Could not load memory file: {e}")
    else:
        print("🆕 No previous memory found.")

def test_weaviate_connection():
    """Test Weaviate connectivity and functionality."""
    if not vectorstore:
        print("❌ Weaviate not connected")
        return False
    
    try:
        # Test basic connectivity
        print("🔍 Testing Weaviate connection...")
        
        # Try to add a test document
        test_doc = Document(
            page_content="Test: This is a test document for connectivity",
            metadata={
                "text": "Test: This is a test document for connectivity",
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000Z"),  # RFC3339 format
                "user_input": "test",
                "ai_response": "test response"
            }
        )
        
        vectorstore.add_documents([test_doc])
        print("✅ Successfully added test document to Weaviate")
        
        # Try to search for it using explicit vector embedding
        query_vec = embedding.embed_query("test")
        results = vectorstore.similarity_search_by_vector(
            embedding=query_vec,
            k=1
        )
        if results:
            print("✅ Successfully retrieved document from Weaviate")
            
            # Clean up test document
            # Note: Weaviate doesn't have a simple delete method, so we'll leave it
            print("✅ Weaviate is working correctly!")
            return True
        else:
            print("❌ Could not retrieve documents from Weaviate")
            return False
            
    except Exception as e:
        print(f"❌ Weaviate test failed: {e}")
        return False

def repopulate_weaviate_from_memory(interactions, long_term=False):
    """Repopulate Weaviate vector store with historical interactions."""
    if not vectorstore or not interactions:
        return
    
    try:
        print("🔄 Repopulating Weaviate with historical data...")
        documents = []
        
        for interaction in interactions:
            doc = Document(
                page_content=f"User: {interaction['user']}\nAI: {interaction['ai']}",
                metadata={
                    "text": f"User: {interaction['user']}\nAI: {interaction['ai']}",
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000Z"),  # RFC3339 format
                    "user_input": interaction['user'],
                    "ai_response": interaction['ai']
                }
            )
            documents.append(doc)
        
        if documents:
            vectorstore.add_documents(documents)
            print(f"✅ Repopulated Weaviate with {len(documents)} historical interactions")
    except Exception as e:
        print(f"⚠️ Could not repopulate Weaviate: {e}")

def save_memory_to_file():
    """Save conversation history to JSON file."""
    try:
        memory_variables = memory.load_memory_variables({})
        history = memory_variables.get("history", [])
        
        interactions = []
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                user_msg = history[i]
                ai_msg = history[i + 1]
                
                if isinstance(user_msg, HumanMessage) and isinstance(ai_msg, AIMessage):
                    interaction = {
                        "user": user_msg.content,
                        "ai": ai_msg.content,
                        "timestamp": datetime.now().isoformat()
                    }
                    interactions.append(interaction)
        
        data = {
            "last_updated": datetime.now().isoformat(),
            "interactions": interactions
        }
        
        with open(MEMORY_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved {len(interactions)} interactions.")
    except Exception as e:
        print(f"⚠️  Could not save memory file: {e}")

def store_interaction(user_input, ai_response):
    """Stores the interaction into both conversation memory and Weaviate vector store."""
    # Add the interaction to memory
    memory.save_context(
        {"input": user_input},
        {"output": ai_response}
    )
    
    # Store in Weaviate vector store
    if vectorstore:
        try:
            # Create document with proper text field for Weaviate
            doc = Document(
                page_content=f"User: {user_input}\nAI: {ai_response}",
                metadata={
                    "text": f"User: {user_input}\nAI: {ai_response}",
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"),  # RFC3339 format
                    "user_input": user_input,
                    "ai_response": ai_response
                }
            )
            vectorstore.add_documents([doc])
            print(f"💾 Stored interaction in Weaviate")
        except Exception as e:
            print(f"⚠️ Could not store in Weaviate: {e}")
    
    # Also persist to file as backup
    save_memory_to_file()

def get_chat_history(n=None):
    """Retrieve the most recent n interactions from conversation memory."""
    memory_variables = memory.load_memory_variables({})
    history = memory_variables.get("history", [])
    
    if not history:
        return []
    
    interactions = []
    for i in range(0, len(history), 2):
        if i + 1 < len(history):
            user_msg = history[i]
            ai_msg = history[i + 1]
            
            if isinstance(user_msg, HumanMessage) and isinstance(ai_msg, AIMessage):
                interaction = {
                    "user": user_msg.content,
                    "ai": ai_msg.content,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                interactions.append(interaction)
    
    if n is None:
        return interactions
    else:
        return interactions[-n:]

def rerank_by_relevance(query, docs, k=3):
    """Re-rank documents by combining similarity and recency scores."""
    if not sentence_model or not docs:
        return docs[:k]
    
    try:
        query_vec = sentence_model.encode(query, convert_to_tensor=True)
        now = datetime.utcnow()
        scored = []

        for doc in docs:
            doc_vec = sentence_model.encode(doc.page_content, convert_to_tensor=True)
            sim = util.pytorch_cos_sim(query_vec, doc_vec).item()
            
            # Calculate recency score
            try:
                timestamp = datetime.fromisoformat(doc.metadata.get('timestamp', now.isoformat()))
                time_diff = (now - timestamp).total_seconds()
                recency = 1 / (1 + time_diff / 3600)  # Normalize by hours
            except:
                recency = 0.5  # Default score if timestamp parsing fails
            
            # Combine similarity and recency (70% similarity, 30% recency)
            score = 0.7 * sim + 0.3 * recency
            scored.append((score, doc))

        # Sort by combined score and return top k
        top_docs = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
        return [doc for _, doc in top_docs]
    except Exception as e:
        print(f"⚠️ Error in re-ranking: {e}")
        return docs[:k]

def get_relevant_memory(query, k=3):
    """Retrieve the most relevant k interactions from Weaviate vector store with re-ranking."""
    if not vectorstore:
        return []
    
    # from short term memory
    try:
        # Get more results initially for better re-ranking using explicit vector embedding
        query_vec = embedding.embed_query(query)
        raw_results = vectorstore.similarity_search_by_vector(
            embedding=query_vec,
            k=min(10, k * 3)
        )
        
        # Re-rank by relevance and recency
        top_results = rerank_by_relevance(query, raw_results, k)
        
        relevant_interactions = []
        
        for doc in top_results:
            # Parse the stored text back into user/ai parts
            content = doc.page_content
            if "User: " in content and "AI: " in content:
                parts = content.split("AI: ", 1)
                if len(parts) == 2:
                    user_part = parts[0].replace("User: ", "").strip()
                    ai_part = parts[1].strip()
                    
                    relevant_interactions.append({
                        "user": user_part,
                        "ai": ai_part,
                        "timestamp": doc.metadata.get("timestamp", "Unknown"),
                        "relevance_score": "High"
                    })
            else:
                # Fallback to metadata if page_content parsing fails
                relevant_interactions.append({
                    "user": doc.metadata.get("user_input", "Unknown"),
                    "ai": doc.metadata.get("ai_response", "Unknown"),
                    "timestamp": doc.metadata.get("timestamp", "Unknown"),
                    "relevance_score": "High"
                })
        
        return relevant_interactions
    except Exception as e:
        print(f"⚠️ Could not retrieve from Weaviate: {e}")
        return []
    

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '')
        
        if not user_message.strip():
            return jsonify({'error': 'Message cannot be empty'})
        
        # Get conversation history from memory (use structured message objects directly)
        history = memory.load_memory_variables({})["history"]
        
        # Get relevant memory from Weaviate
        relevant_memory = get_relevant_memory(user_message, k=3)
        
        # Create enhanced context with relevant memory
        enhanced_context = ""
        if relevant_memory:
            enhanced_context = "\n\nRelevant previous conversations:\n"
            for i, mem in enumerate(relevant_memory, 1):
                enhanced_context += f"{i}. User: {mem['user']}\n   AI: {mem['ai']}\n"
        
        # Log the full prompt being sent to the LLM
        full_prompt = user_message + enhanced_context
        print(f"\n🤖 PROMPT SENT TO LLM:")
        print(f"User Message: {user_message}")
        if relevant_memory:
            print(f"📚 Found {len(relevant_memory)} relevant memories from Weaviate:")
            for i, mem in enumerate(relevant_memory, 1):
                print(f"   {i}. User: {mem['user'][:50]}...")
        else:
            print("📚 No relevant memories found in Weaviate")
        print(f"📝 Full Prompt: {full_prompt[:200]}...")
        
        # Generate response with LLM
        result = chain.invoke({
            "history": history,
            "question": full_prompt
        })
        
        # Clean up the response
        #ai_response = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        ai_response = re.sub(r"<think>.*?</think>", "", result.content, flags=re.DOTALL).strip()
        
        # Store this conversation for future context and persist to file
        store_interaction(user_message, ai_response)
        
        return jsonify({
            'response': ai_response,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'relevant_memory': relevant_memory
        })
        
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'})


# @app.route('/history', methods=['GET'])
def get_history():
    try:
        n_param = request.args.get('n', None)
        if n_param is not None:
            n = int(n_param)
        else:
            n = None
        history = get_chat_history(n)
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'})

# @app.route('/clear', methods=['POST'])
def clear_history():
    try:
        # Clear memory
        memory.clear()
        
        # Clear Weaviate vector store
        if vectorstore and client:
            try:
                # Delete the ChatMemory class and recreate it
                client.schema.delete_class("ChatMemory")
                
                # Recreate the class with correct schema
                schema = {
                    "class": "ChatMemory",
                    "vectorizer": "none",  # Use external embeddings
                    "properties": [
                        {"name": "text", "dataType": ["text"]},
                        {"name": "timestamp", "dataType": ["date"]},
                        {"name": "user_input", "dataType": ["text"]},
                        {"name": "ai_response", "dataType": ["text"]}
                    ]
                }
                client.schema.create_class(schema)
                print("🗑️ Cleared Weaviate vector store")
            except Exception as e:
                print(f"⚠️ Could not clear Weaviate: {e}")
        
        # Remove memory file
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
        
        return ("Chat history cleared successfully")
    except Exception as e:
        return ("Error")

if __name__ == '__main__':
    
    clear_history()
    load_long_term_memory() # long long term context memory on startup
    # load_memory_from_file() # Load previous conversations on startup, should not need this ever i think
    
    # Test Weaviate connection
    if vectorstore:
        test_weaviate_connection()
    
    print("🚀 Starting AI Chatbot Web Server...")
    print("📱 Open your browser and go to: http://localhost:5001")
    print("💾 Persistent memory enabled")
    
    app.run(debug=True, host='0.0.0.0', port=5001) 