from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
import os
from dotenv import load_dotenv
load_dotenv()

def load_chat_model(model_id:str,temperature: int,max_tokens: int):
    """
    Load the chat model from Hugging Face endpoint.
    """
    # Define the Hugging Face endpoint
    endpoint = HuggingFaceEndpoint(
        model_id=model_id,
        temperature=temperature,
        max_new_tokens=max_tokens,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    )
    # Create a chat model using the endpoint
    chat_model = ChatHuggingFace(endpoint)
    return chat_model