import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve API credentials from environment variables
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

# Initialize the OpenAI Client using the credentials from the .env file
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

# Streamlit app
st.title("Comfort Cove")
st.subheader("A safe space for your thoughts, where empathy meets understanding.")

# Input text box
input_text = st.text_input("Ask me anything:", placeholder="Type your message here...")

# Function to handle chatbot response streaming
def generate_response(input_text):
    if input_text:
        # Display a placeholder for the streamed response
        response_placeholder = st.empty()

        # Create a chat completion stream
        response_stream = client.chat.completions.create(
            model="rath1991/llama_32_3b_ft_comfort_cove_merged",
            messages=[{"role": "user", "content": input_text}],
            temperature=0,
            max_tokens=500,
            stream=True,
        )

        # Stream the response and update the placeholder
        full_response = ""
        for response in response_stream:
            response_content = response.choices[0].delta.content or ""
            full_response += response_content
            response_placeholder.text(full_response)  # Update the placeholder with the streaming content

# Trigger response generation when the user inputs text
if input_text:
    generate_response(input_text)