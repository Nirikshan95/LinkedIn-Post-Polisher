from langgraph.graph import StateGraph,END,MessageGraph
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import TypedDict,List,Annotated
from src.chains import generator_chain, reflection_chain
from config import GENERATOR, REFLECTOR

class AgentState(TypedDict):
    """
    TypedDict to define the structure of the agent's state.
    This can be extended with more fields as needed.
    """
    messages: Annotated[List[str],Field(description="list of messages i.e.. AI Message, system message or human messages etc ",)]  # List of messages in the conversation
    
# nodes creation
'''def generator_node(state):
    """
    Node function for the generator chain.
    This function processes the state and generates a LinkedIn post.
    """
    post = generator_chain.invoke(state)
    return post.content

def reflection_node(state):
    """
    Node function for the reflection chain.
    This function processes the state and generates a reflection on the LinkedIn post.
    """
    reflection = reflection_chain.invoke(state)
    return HumanMessage(reflection.content)'''

def conditional_logic(state):
    """
    Conditional logic to determine the next step in the graph.
    If the state contains a reflection, it will return the REFLECTOR node.
    Otherwise, it will return the GENERATOR node.
    """
    if len(state)<5:
        return REFLECTOR
    else:
        return END    

def load_graph():
    """
    Function to build and return the message graph.
    This function sets up the nodes and edges of the graph.
    """
    graph_builder = MessageGraph()
    # Adding nodes to graph
    reflection_node=reflection_chain| RunnableLambda(lambda reflection: HumanMessage(reflection.content))  # reflector
    graph_builder.add_node(GENERATOR, generator_chain)      # responder
    graph_builder.add_node(REFLECTOR, reflection_node)     # reflector
    #  Entry point for the graph
    graph_builder.set_entry_point(GENERATOR)
    # Adding edges/conditional-edges to graph
    graph_builder.add_conditional_edges(GENERATOR, conditional_logic)
    graph_builder.add_edge(REFLECTOR,GENERATOR)
    

    # compile graph
    return graph_builder.compile()
def get_mermaid_graph():
    """
    Function to build and return the Mermaid graph representation.
    This function generates a string that represents the graph in Mermaid syntax.
    """
    graph = load_graph()
    #print("ascii graph: \n ", graph.get_graph().draw_ascii())
    #print( graph.get_graph().draw_png(output_file_path="mermaid_graph.png"))
    
    return graph.get_graph().draw_mermaid()