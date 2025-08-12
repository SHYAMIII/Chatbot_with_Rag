from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
history = []

# ---------------- Load prebuilt FAISS index ----------------
# Using the smallest HuggingFace embedding model to save memory
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.load_local(
    "my_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
# ------------------------------------------------------------

# ---------------- Groq LLM configuration -------------------
model = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",  # Other options: llama3-8b-8192, llama3-70b-8192, gemma-7b-it
    temperature=0.7,
    max_tokens=512
)
# ------------------------------------------------------------

# ---------------- Prompt template --------------------------
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
# ------------------------------------------------------------

def reset_history():
    global history
    history = []

def get_bot_response(user_input: str) -> str:
    history_text = "\n".join([f"User: {u}\nBot: {b}" for u, b in history])

    # Retrieve context from FAISS
    docs = vector_store.similarity_search(user_input, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    combined_context = f"Conversation so far:\n{history_text}\n\nRelevant documents:\n{context}"
    final_prompt = prompt.format(context=combined_context, question=user_input)

    ans = model.invoke(final_prompt).content
    history.append((user_input, ans))
    print(ans)
    return ans

# Test run
if __name__ == "__main__":
    get_bot_response("who is shyam?")
