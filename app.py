import streamlit as st
from graphs.graph import load_graph, get_mermaid_graph
from langchain_core.messages import HumanMessage

def main():
    st.set_page_config(page_title="LinkedIn Post Generator", page_icon=":memo:")
    st.title("LinkedIn Post Generator")
    
    if st.button("graph"):
        # Display the Mermaid graph
        st.subheader("Graph Representation")
        mermaid_graph = get_mermaid_graph()
        st.code(mermaid_graph, language='mermaid')
    
    topic=st.text_input("Describe your topic for your LinkedIn post generation:")
    if st.button("Generate Post"):
        if not topic:
            st.warning("Please enter a topic to generate a LinkedIn post.")
            return
        else:
            st.subheader("Post Generation")
            # Load the message graph
            graph = load_graph()
            with  st.spinner("Generating post..."):
                result=graph.invoke([HumanMessage(content=topic)])
                print(f'type of result : {type(result)}')
                st.subheader("Graph Result")
                st.markdown(result)
                print(f"\n\n optimized Result: {result[-1].content}")
                st.markdown(result[-1].content)
    
if __name__ == "__main__":
    main()