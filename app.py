import streamlit as st
from dotenv import load_dotenv
from utils import *
import uuid

if 'unique_id' not in st.session_state:
    st.session_state['unique_id']=''
    
def main():
    load_dotenv()
    
    st.set_page_config(page_title="Resume Screening Assistance")
    st.title("HR - Resume Screening Assistance ...")
    st.subheader("I can help you in resume screening process")
    
    job_description = st.text_area("please paste the job description here ..", key=1)
    document_count = st.text_input("No. of resume to return", key=2)
    
    pdf= st.file_uploader("Upload resumes here, only PDF files allowed", type=["pdf"], accept_multiple_files=True)
    
    submit = st.button("Help me with the analysis")
    
    if submit:
        with st.spinner("Wait for it ...."):
            st.session_state['unique_id'] = uuid.uuid4().hex
            final_docs_list = create_docs(pdf, st.session_state['unique_id'])
            #Displaying the count of resumes that have been uploaded
            st.write("*Resumes uploaded* :"+str(len(final_docs_list)))

            embeddings = create_embeddings_load_data()
            push_to_pinecone(embeddings,final_docs_list)
            relavant_docs=similar_docs(job_description,document_count,embeddings,st.session_state['unique_id'])

            st.write(relavant_docs)

            #Introducing a line separator
            st.write(":heavy_minus_sign:" * 30)

            #For each item in relavant docs - we are displaying some info of it on the UI
            for item in range(len(relavant_docs)):
                
                st.subheader("👉 "+str(item+1))

                #Displaying Filepath
                st.write("**File** : "+relavant_docs[item][0].metadata['name'])

                #Introducing Expander feature
                with st.expander('Show me 👀'): 
                    st.info("**Match Score** : "+str(relavant_docs[item][1]))
                    #st.write("***"+relavant_docs[item][0].page_content)
                    
                    #Gets the summary of the current item using 'get_summary' function that we have created which uses LLM & Langchain chain
                    summary = get_summary(relavant_docs[item][0])
                    st.write("**Summary** : "+summary)

        st.success("Hope I was able to save your time❤️")
#Invoking main function
if __name__ == '__main__':
    main()
