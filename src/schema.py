from pydantic import BaseModel, Field
from typing import Annotated
from langchain_core.output_parsers import PydanticOutputParser

class Post(BaseModel):
    """
    Model to represent a LinkedIn post or suggestion or critiques .
    """
    post: Annotated[str ,Field(..., description="LinkedIn post Content")]

class Critique(BaseModel):
    """
    Model to represent a critique of a LinkedIn post.
    """
    contents: Annotated[str, Field(..., description="Content of the critique or suggestions for improvement.")]

def content_parser(pydantic_object=Post):
    """
    Function to create a Pydantic output parser for the Post model.
    This parser will be used to parse the output of the LLM into a object [Post or Critique].
    """
    return PydanticOutputParser(pydantic_object=pydantic_object)