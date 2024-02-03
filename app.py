import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import asyncio
from session_state import get
from httpx_oauth.clients.google import GoogleOAuth2
import csv
import os
import pandas as pd

# Global variables for storing parameters
google_client_id = ''
google_client_secret = ''
redirect_uri = 'http://localhost:8501'
authorization_url = None

# Get the directory of the current script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Define the relative path for the CSV file
csv_file_path = os.path.join(current_directory, "user_data.csv")

# Function to get text from PDF documents


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversational retrieval chain


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Async function to write authorization URL


async def write_authorization_url():
    global authorization_url
    client = GoogleOAuth2(google_client_id, google_client_secret)
    authorization_url = await client.get_authorization_url(
        redirect_uri,
        scope=["profile", "email"],
        extras_params={"access_type": "offline"},
    )

# Async function to write access token


async def write_access_token(code):
    client = GoogleOAuth2(google_client_id, google_client_secret)
    token = await client.get_access_token(code, redirect_uri)
    return token

# Async function to get email


async def get_email(token):
    client = GoogleOAuth2(google_client_id, google_client_secret)
    user_id, user_email = await client.get_id_email(token)
    return user_id, user_email

# Function to save the user input and chat history to a CSV file


def save_to_csv(user_email, user_question, bot_response):
    data = {'User': [user_email], 'Question': [
        user_question], 'Bot Response': [bot_response]}
    df = pd.DataFrame(data)

    # Create the CSV file if it doesn't exist
    if not os.path.exists(csv_file_path):
        print("Creating a new CSV file.")
        df.to_csv(csv_file_path, index=False)
    else:
        # Append the user input and chat history to the CSV file
        print("Appending to the existing CSV file.")
        df.to_csv(csv_file_path, mode='a', header=False, index=False)

    print(f"Data saved to CSV: {csv_file_path}")

# Function to display previous conversations


def display_previous_conversations():
    if os.path.exists(csv_file_path):
        st.subheader("Previous Conversations")
        with st.spinner("Loading"):
            df = pd.read_csv(csv_file_path)
            user_email_filter = df['User'] == st.session_state.user_email
            user_df = df[user_email_filter]

            if not user_df.empty:
                reversed_df = user_df.iloc[::-1]  # Reverse the DataFrame
                table_md = reversed_df[['Question', 'Bot Response']].to_markdown(
                    index=False)
                st.markdown(table_md, unsafe_allow_html=True)
            else:
                st.write("No previous conversations found.")

# Function to handle user input and update conversation


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Separate user input and bot responses
    user_inputs = []
    bot_responses = []

    for message in st.session_state.chat_history:
        # if isinstance(message, HumanMessage):
        #     # Handle HumanMessage
        #     user_inputs.append(user_question)
        #     bot_responses.append('')
        # elif isinstance(message, AIMessage):
        #     # Handle AIMessage
        #     user_inputs.append('')
        #     bot_responses.append(message.content)
        # else:
        #     # Handle other types of messages or provide a default behavior
        #     pass
        print('hello')

    # Display the conversation in the chat interface
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            user_inputs.append(user_question)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            bot_responses.append(message.content)

    # Save user inputs and bot responses to CSV
    save_to_csv(st.session_state.user_email, user_question,
                st.session_state.chat_history[-1].content)

    user_inputs.clear()
    bot_responses.clear()

# Main function


def main():
    global authorization_url
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Explicitly initialize the session state
    session_state = get(token=None)
    if 'token' not in session_state:
        session_state.token = None

    if session_state.token is None:
        st.write('''<h1> Ask PDF </h1>''', unsafe_allow_html=True)
        login_button = st.button("", key="login_button")

        # Use HTML and CSS to set the button's visibility to hidden
        st.markdown("""
            <style>
                #login_button {
                    visibility: hidden;
                    width: 0;
                    height: 0;
                    padding: 0;
                    margin: 0;
                }
            </style>
        """, unsafe_allow_html=True)

        # st.write('''<h3>Unleash the power of AI-driven document interaction with 'Chat with multiple PDFs': Seamlessly converse and extract insights from your PDF documents through intuitive chat, making information retrieval a conversation</h3>''', unsafe_allow_html=True)
        if login_button:
            asyncio.run(write_authorization_url())
            st.write(
                f'''<h2>Please login using this <a target="_self" href="{authorization_url}">link</a></h2>''', unsafe_allow_html=True)
            return

        try:
            code = st.experimental_get_query_params()['code']
        except:
            asyncio.run(write_authorization_url())
            st.write(
                f'''<h2>Please login using this <a target="_self" href="{authorization_url}">link</a></h2>''', unsafe_allow_html=True)
        else:
            try:
                token = asyncio.run(write_access_token(code))
            except:
                asyncio.run(write_authorization_url())
                st.write(f'''<h2>This account is not allowed or the page was refreshed or you have logged out successfully.</h2>
                            \n <h2>Please try again: <a target="_self" href="{authorization_url}">link</a></h2>''', unsafe_allow_html=True)
            else:
                if token.is_expired():
                    asyncio.run(write_authorization_url())
                    st.write(
                        f'''<h2>Login session has ended, please <a target="_self" href="{authorization_url}">link</a> again.</h2>''')
                else:
                    session_state.token = token
                    user_id, user_email = asyncio.run(
                        get_email(token=token['access_token'])
                    )
                    session_state.user_id = user_id
                    session_state.user_email = user_email

                    st.write(f"You're logged in as {user_email}")

                    # Initialize conversation chain if it doesn't exist yet
                    if "conversation" not in st.session_state:
                        # You can initialize with an empty text for now
                        text_chunks = [""]
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(
                            vectorstore)

                    st.header("Chat with multiple PDFs :books:")
                    user_question = st.text_input(
                        "Ask a question about your documents:")
                    if user_question:
                        handle_userinput(user_question)

                    with st.sidebar:
                        st.subheader("Your documents")

                        # Add your PDF chat functionality here
                        pdf_docs = st.file_uploader(
                            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
                        if st.button("Process"):
                            with st.spinner("Processing"):
                                raw_text = get_pdf_text(pdf_docs)
                                text_chunks = get_text_chunks(raw_text)
                                vectorstore = get_vectorstore(text_chunks)
                                st.session_state.conversation = get_conversation_chain(
                                    vectorstore)

                        # Add logout button at the extreme bottom
                        if st.button("Logout"):
                            session_state.clear()
                            st.experimental_rerun()

    else:
        st.write(f"You're already logged in as {session_state.user_email}")

        with st.sidebar:
            st.subheader("Your documents")

            # Add your PDF chat functionality here
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)

            # Add logout button at the extreme bottom
            if st.button("Logout"):
                session_state.clear()
                st.experimental_rerun()

        st.header("Chat with multiple PDFs :books:")
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

        # Display previous conversations
        display_previous_conversations()


if __name__ == '__main__':
    main()
