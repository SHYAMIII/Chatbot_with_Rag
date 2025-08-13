import os
os.environ["HF_HOME"] = "/data"

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
import pinecone




load_dotenv()
history = []



# These will load only when first needed
embeddings = None
vector_store = None
retriever = None
model = None

def load_resources():
    """Lazy-loads heavy models only when first needed."""
    global embeddings, vector_store, retriever, model

    if embeddings is not None:
        return  # Already loaded
    
    

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("Loading Pinecone index...")

    vector_store = PineconeVectorStore(
        index_name="chatbotapv",
        embedding=embeddings,
    )

    print("Loading retriever...")

    # retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    print("Loading Groq model...")
    model = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=512
    )

prompt = PromptTemplate(
    template="""
      You are a helpful representative of the company= (auto-pilot-verse), named - alex.
      Answer very shortly and only provide what the user asked for.
      ONLY use the provided transcript context.
      Don't introduce yourself unless explicitly asked.
      If the context is insufficient, reply with a friendly fallback message.

      {context}
      Question: {question}
    """,
    input_variables=['context', 'question']
)

def reset_history():
    global history
    history = []

def get_bot_response(user_input: str) -> str:
    load_resources()  # Only load models when first needed

    history_text = "\n".join([f"User: {u}\nBot: {b}" for u, b in history])

    # Retrieve context
    docs = vector_store.similarity_search(user_input, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    combined_context = f"Conversation so far:\n{history_text}\n\nRelevant documents:\n{context}"
    final_prompt = prompt.format(context=combined_context, question=user_input)

    ans = model.invoke(final_prompt).content
    history.append((user_input, ans))
    return ans