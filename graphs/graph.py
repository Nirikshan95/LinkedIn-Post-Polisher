from langgraph.graph import END,MessageGraph
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from src.chains import generator_chain, reflection_chain
from config import GENERATOR, REFLECTOR
    
# nodes creation
def generator_node(state):
    """
    Node function for the generator chain.
    This function processes the state and generates a LinkedIn post.
    """
    post_content = generator_chain().invoke({'state':state})
    return [AIMessage(post_content)]

def reflection_node(state):
    """
    Node function for the reflection chain.
    This function processes the state and generates a reflection on the LinkedIn post.
    """
    reflection = reflection_chain().invoke({'state':state})
    return [HumanMessage(reflection)]

def conditional_logic(state):
    """
    Conditional logic to determine the next step in the graph.
    If the state contains a reflection, it will return the REFLECTOR node.
    Otherwise, it will return the GENERATOR node.
    """
    ai_message_count= sum(1 for msg in state if isinstance(msg, AIMessage))
    if ai_message_count<=3:
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
    #reflection_node=reflection_chain| RunnableLambda(lambda reflection: HumanMessage(reflection.content))  # reflector
    graph_builder.add_node(GENERATOR, generator_node)      # responder
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