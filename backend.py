from flask import Flask, request, jsonify
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()
groq_api_key = os.getenv('groq_api')

# Global variables
vectors = None
embeddings = None
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template("""
    You are a document assistant that helps users find information in a context.
    Please provide the most accurate response based on the context and inputs.
    Only give information that is in the context, not in general.
    
    <context>
    {context}
    </context>
    
    Question: {input}
""")

@app.route('/process_document', methods=['POST'])
def process_document():
    global vectors, embeddings
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Invalid file type'}), 400
        
    try:
        # Save temporary file
        temp_path = "temp_file.pdf"
        file.save(temp_path)
        
        # Process document
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        final_documents = text_splitter.split_documents(docs)
        vectors = FAISS.from_documents(final_documents, embeddings)
        
        return jsonify({'message': 'Document processed successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/query', methods=['POST'])
def query_document():
    if not vectors:
        return jsonify({'error': 'No document processed yet'}), 400
    
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': query})
        elapsed_time = time.process_time() - start
        
        return jsonify({
            'answer': response['answer'],
            'context': [doc.page_content for doc in response['context']],
            'elapsed_time': elapsed_time
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)