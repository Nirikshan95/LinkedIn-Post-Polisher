from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from src.chat_model import load_chat_model
from src.schema import Post, Critique,content_parser
from config import REPO_ID, TEMPERATURE, MAX_NEW_TOKENS

llm=load_chat_model(REPO_ID, TEMPERATURE, MAX_NEW_TOKENS)


def generator_chain():
    generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a professional AI writing assistant that helps users create engaging LinkedIn posts.

    create an best engaging LinkedIn post for user's request in the average post length  150 to 300 words.

    Guidelines:
    - Use a strong opening line to hook attention.
    - Keep it concise but meaningful.
    - Avoid emojis, and overly technical jargon.
    - End with a question or call to action.

    if user provides recommendations or critique , respond with revised version of your previous post.

    Return only the string of post content.
    format instructions :
    {format_instructions}
    """),
            MessagesPlaceholder(variable_name="state"),
        ]
    ).partial(format_instructions= content_parser(Post).get_format_instructions())
    return generator_prompt | llm | content_parser(Post) | RunnableLambda(lambda out:out.post)

def reflection_chain():
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
    Return all the bullet points as a single string.
    
    format instructions :
    {format_instructions}
    """),
            MessagesPlaceholder(variable_name="state"),
        ]
    ).partial(format_instructions= content_parser(Critique).get_format_instructions())

    return reflector_prompt | llm | content_parser(Critique) | RunnableLambda(lambda reflection:reflection.contents)