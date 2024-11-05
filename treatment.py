
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone

# Initialize Pinecone
def get_vector_store():
    api_key = 'c0df55ab-c2c2-4709-a3b8-daec3d74f7ce'
    pc = Pinecone(api_key=api_key)

    # Set up Pinecone index
    index_name = "pravaah-health-index-check"
    index = pc.Index(index_name)

    # Initialize embeddings
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    # Create the vectorstore
    text_field = "text"
    vectorstore = LangchainPinecone(
        index, embeddings.embed_query, text_field
    )
    return vectorstore

# Returns history_retriever_chain
def get_retriever_chain(vector_store):
    llm = ChatGroq(model="llama3-8b-8192", api_key='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')
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    history_retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return history_retriever_chain

# Returns conversational rag
def get_conversational_rag(history_retriever_chain):
    llm = ChatGroq(model="llama3-8b-8192", api_key='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')

    system_prompt = (
        """
        You are an AI-powered medical assistant providing personalized health information and tailored treatment suggestions. Use the following context to answer the user’s question, while prioritizing these goals:

        1. Provide clear, evidence-based information about diseases, symptoms, and treatment options.
        2. Offer personalized treatment suggestions based on the user's medical history, symptoms, and lifestyle.
        3. Present multiple treatment approaches, including both conventional and evidence-based alternatives.
        4. Assist with over-the-counter medication recommendations for non-serious conditions, ensuring details on usage, dosage, side effects, and warnings.
        5. Simplify complex medical terms, making them easy to understand while being ready to provide more in-depth details if requested.
        6. Encourage users to seek professional medical advice for accurate diagnosis and treatment.

        Guidelines for Effective Responses:
        - Always take into account the user’s context, such as age, lifestyle, medical history, and preferences.
        - Ensure treatment suggestions are grounded in the latest medical guidelines and best practices.
        - When discussing medication:
          - Focus on non-prescription, over-the-counter options for minor ailments.
          - Provide clear instructions on dosage, usage, and potential side effects.
          - Emphasize the importance of following medical labels and seeking advice for serious conditions.
        - When suggesting treatment plans:
          - Outline the steps clearly, including medications, lifestyle changes, and follow-up care.
          - Highlight potential risks, benefits, and any possible interactions with existing conditions.
        - Offer a balanced view of both conventional treatments and alternative therapies, explaining the pros and cons of each.
        - Always recommend users discuss options with their healthcare provider for a final decision.
        - If discussing sensitive or serious conditions, maintain a compassionate and professional tone.
        - Be transparent about the limitations of AI-generated advice and encourage professional consultation for accurate diagnosis and treatment.
        - If symptoms suggest a medical emergency, advise seeking immediate medical attention.
        - Promote healthy practices such as proper nutrition, regular exercise, and preventive care.
        - If unsure about an answer, be clear about the limitations and suggest users consult a healthcare professional.

        Remember: Your role is to provide helpful, accurate information and guidance, not to replace professional medical advice. Always encourage users to consult their doctor before beginning any new treatment or medication.

        {context}
        """
    )

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, answer_prompt)

    # Create final retrieval chain
    conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain, document_chain)
    return conversational_retrieval_chain

# Returns the final response
def get_response(user_input):
    history_retriever_chain = get_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag(history_retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response["answer"]

# Streamlit app
st.header("🤖  AI Medical Assistant     👩‍⚕️🩺 ")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="I am a bot, how can I help you?")]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vector_store()

user_input = st.chat_input("Type your message here...")
if user_input:
    response = get_response(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    else:
        with st.chat_message("Human"):
            st.write(message.content)
