import streamlit as st
from openai import OpenAI


from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve API credentials from environment variables
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

# Set OpenAI API key and base URL
api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('BASE_URL')
client = OpenAI(
    api_key=api_key,
    base_url=base_url

)

# Initialize conversation history in session state
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []  # Stores both user and bot messages

# Streamlit app
st.title("Comfort Cove")
st.subheader("A safe space for your thoughts, where empathy meets understanding.")

# Display previous conversations in chat format
for message in st.session_state['conversation_history']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user input using the new chat input box
if user_input := st.chat_input("Ask me anything:"):
    # Append user's message to session state
    st.session_state['conversation_history'].append({"role": "user", "content": user_input})

    # Display user's message in chat format
    with st.chat_message("user"):
        st.markdown(user_input)

    # Function to handle chatbot response streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()  # Placeholder for streaming response
        full_response = ""

        # Prepare full conversation history as context for the bot
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state['conversation_history']]

        # Create a chat completion stream
        response_stream = client.chat.completions.create(
            model="rath1991/llama_32_3b_ft_comfort_cove_merged",
            messages=messages,  # Use entire conversation history as context
            temperature=0,
            max_tokens=500,
            stream=True,
        )

        # Stream the response and update the placeholder
        for response in response_stream:
            response_content = response['choices'][0]['delta'].get('content', '')
            if response_content:
                full_response += response_content
                response_placeholder.markdown(full_response)

        # Append bot's response to session state
        st.session_state['conversation_history'].append({"role": "assistant", "content": full_response})

        # Limit conversation history to the last 10 exchanges (5 user + 5 bot messages)
        if len(st.session_state['conversation_history']) > 3:
            st.session_state['conversation_history'] = st.session_state['conversation_history'][-3:]
