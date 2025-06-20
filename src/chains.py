from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from src.chat_model import load_chat_model
from config import REPO_ID, TEMPERATURE, MAX_NEW_TOKENS

llm=load_chat_model(REPO_ID, TEMPERATURE, MAX_NEW_TOKENS)

def generator_chain(state):
    generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a professional AI writing assistant that helps users create engaging LinkedIn posts.

    create an best engaging LinkedIn post for user's request.

    Guidelines:
    - Use a strong opening line to hook attention.
    - Keep it concise but meaningful.
    - Avoid emojis, and overly technical jargon.
    - End with a question or call to action.

    if user provides recommendations or critique , respond with revised version of your previous post.

    Return only the post content.
    """),
            MessagesPlaceholder(variable_name="state"),
        ]
    )
    return generator_prompt | llm
    
def reflection_chain(state):
    reflector_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a critical reviewer for professional social media content. Your task is to critique the post constructively.

    Evaluate the post based on:
    1. Clarity
    2. Engagement
    3. Relevance to the audience
    4. Strength of the hook and ending
    5. length and conciseness 
    etc

    List specific improvements to enhance the post. Be helpful, not harsh. Return your reflection as bullet points.
    """),
            MessagesPlaceholder(variable_name="state"),
        ]
    )
    return reflector_prompt | llm
