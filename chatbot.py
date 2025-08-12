from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate


from dotenv import load_dotenv

load_dotenv()
loader = PyPDFLoader("bro.pdf")
docs = loader.load()
history = []


splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)



# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("my_faiss_index")


retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})


# LLM configuration
model = ChatGroq(
    model_name="llama-3.1-8b-instant",         # Options: llama3-8b-8192, llama3-70b-8192, gemma-7b-it
    temperature=0.7,
    max_tokens=512
)




# Prompt template (update this with your actual one from notebook)
prompt = PromptTemplate(
    template="""
      You are a helpful representative of the company= (auto-pilot-verse), named - alex.
      Answer very shortely and give ans only the user has asked for and dont answer anything other he asked and  ONLY from the provided transcript context.dont introduce yourself untill you are not asked to do so.
      If the context is insufficient, then answer a freindly message accordingly the question.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

def reset_history():
    global history
    history = []



def get_bot_response(user_input: str) -> str:

    history_text = "\n".join([f"User: {u}\nBot: {b}" for u, b in history])
    # Retrieve relevant context
    docs = vector_store.similarity_search(user_input, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    combined_context = f"Conversation so far:\n{history_text}\n\nRelevant documents:\n{context}"


    # Fill prompt
    final_prompt = prompt.format(context = combined_context, question=user_input)

    # Generate answer
    ans = model.invoke(final_prompt).content

    history.append((user_input, ans))
    print(ans)
    return ans
get_bot_response("who is alex?")