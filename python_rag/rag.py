def chatbot_response(query):
    from llama_parse import LlamaParse

    parser = LlamaParse(
        api_key="llx-juiHEGjTVNPzgpcPIVxg8EQR81OdNSkH71uHqH3eK0GeDneM",
        result_type="markdown", 
        verbose=True
    )
    groq_api_key="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"
    llamaparse_api_key="llx-juiHEGjTVNPzgpcPIVxg8EQR81OdNSkH71uHqH3eK0GeDneM"

    from llama_parse import LlamaParse

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from groq import Groq
    from langchain_groq import ChatGroq
    import joblib
    import os
    import nest_asyncio  
    nest_asyncio.apply()

    def load_or_parse_data():
        data_file = "python_rag\data\parsed_data.pkl"

        if os.path.exists(data_file):
            # Load the parsed data from the file
            parsed_data = joblib.load(data_file)
        else:
            parsingInstructionUber10k = """The provided document is a resume of a person,
            This form provides detailed information about the person , like what role he is applying for and other details like
            his projects, work experience, hobbies , certifications, education etc. 
            The document might contain tables.
            Try to be precise while answering the questions.
            Note that you need to use a friendly and informal tone while answering the question and ensure of maintaining the positive sentiment throughout
            your answer."""
            parser = LlamaParse(api_key=llamaparse_api_key,
                                result_type="markdown",
                                parsing_instruction=parsingInstructionUber10k,
                                max_timeout=5000,)
            llama_parse_documents = parser.load_data(".\python_rag\data\Jaganath_resume2024.pdf")

            joblib.dump(llama_parse_documents, data_file)
            parsed_data = llama_parse_documents

        return parsed_data

    def create_vector_database():
        
        llama_parse_documents = load_or_parse_data()
        
        with open('python_rag\data\output.md', 'a') as f: 
            for doc in llama_parse_documents:
                f.write(doc.text + '\n')
        markdown_path = "python_rag\data\output.md"
        loader = UnstructuredMarkdownLoader(markdown_path)

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        
        vs = Chroma.from_documents(
            documents=docs,
            embedding=embed_model,
            persist_directory="chroma_db_llamaparse1",  
            collection_name="rag"
        )

        return vs,embed_model
        
    vs,embed_model = create_vector_database()
    chat_model = ChatGroq(temperature=0,
                        model_name="mixtral-8x7b-32768",
                        api_key=groq_api_key)

    vectorstore = Chroma(embedding_function=embed_model,
                        persist_directory="chroma_db_llamaparse1",
                        collection_name="rag")

    retriever=vectorstore.as_retriever(search_kwargs={'k': 3})

    custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    def set_custom_prompt():

        prompt = PromptTemplate(template=custom_prompt_template,
                                input_variables=['context', 'question'])
        return prompt

    prompt = set_custom_prompt()

    qa = RetrievalQA.from_chain_type(llm=chat_model,
                                chain_type="stuff",
                                retriever=retriever,
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": prompt})
    response = qa.invoke({"query": query})
    return response['result']
