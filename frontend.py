import streamlit as st
import requests
import time

st.title("RAG Doc Assistant")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Process document button
if st.button("Process Document") and uploaded_file:
    files = {'file': uploaded_file}
    response = requests.post('http://localhost:5000/process_document', files=files)
    
    if response.status_code == 200:
        st.success("Document processed successfully!")
    else:
        st.error(f"Error: {response.json().get('error')}")

# Query input
prompt = st.text_input("Enter Your Question About the Document")

if prompt:
    try:
        response = requests.post('http://localhost:5000/query', json={'query': prompt})
        
        if response.status_code == 200:
            data = response.json()
            st.write(f"Response time: {data['elapsed_time']:.2f} seconds")
            st.write(data['answer'])
            
            with st.expander("Document Similarity Search"):
                for i, context in enumerate(data['context']):
                    st.markdown(f"**Relevant Section {i+1}:**")
                    st.write(context)
                    st.divider()
        else:
            st.error(f"Error: {response.json().get('error')}")
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")