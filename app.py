import streamlit as st
from graphs.graph import load_graph
from langchain_core.messages import HumanMessage

def main():
    st.set_page_config(page_title="LinkedIn Post Generator", page_icon=":memo:")
    st.title("LinkedIn Post Generator")
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
                try:
                    result=graph.invoke([HumanMessage(content=topic)])
                    st.markdown(result[-1].content)
                except Exception as e:
                    st.error(f"An error occurred while generating the post: {e}")
    
if __name__ == "__main__":
    main()