import streamlit as st
import os
from utils import *
import pandas as pd
from streamlit_pdf_viewer import pdf_viewer
import concurrent
from dotenv import load_dotenv


# Function to save uploaded files
def save_uploadedfile(uploadedfile, save_folder):
    with open(os.path.join(save_folder, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

# Main function to run the Streamlit app

# Main function to run the Streamlit app
def main():
    st.title("SoftsensorX - AI Resume Screener")

    # Create the directory if it doesn't exist
    save_folder = "input_file"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Read and inject custom CSS for sidebar and title
    with open('mystyle.css') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    # File uploader widget
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    # Sidebar elements
    with st.sidebar:
        # Toggle for Top K input
        top_k_toggle = st.checkbox("Enable Top K input", value=False)

        # Conditional slider input for Top K
        top_k = None
        if top_k_toggle:
            top_k = st.slider("Select Top K value", min_value=1, max_value=50, value=10)

        # Toggle for Percentage Criteria input
        percentage_toggle = st.checkbox("Enable Percentage Criteria input", value=False)

        # Conditional slider input for Percentage Criteria
        percentage_criteria = None
        if percentage_toggle:
            percentage_criteria = st.slider("Select Percentage Criteria value", min_value=10, max_value=100, value=50)

        # Filter button
        filter_button = st.button("Apply Filters")

    # Text area for job description
    job_description = st.text_area("Enter Job Description")

    # Upload button
    if st.button('Upload Resumes'):
        if uploaded_files and job_description:
            # Save uploaded files
            for uploaded_file in uploaded_files:
                save_uploadedfile(uploaded_file, save_folder)

            # Load resumes and extract text
            file_paths = [os.path.join(save_folder, uploaded_file.name) for uploaded_file in uploaded_files]
            resumes = load_resumes(file_paths)

            # Generate embeddings for job description
            # job_description_embedding = create_openai_embeddings(job_description)

            # Compute similarity scores
        # Function to process a single resume
            def process_resume(file_path, text, job_description):
                score, analysis = new_llm_response(text, job_description)
                return file_path, score, analysis



            # Use ThreadPoolExecutor to process resumes in parallel
            similarity_scores = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_resume = {executor.submit(process_resume, file_path, text, job_description): file_path for file_path, text in resumes.items()}
                for future in concurrent.futures.as_completed(future_to_resume):
                    file_path = future_to_resume[future]
                    try:
                        file_path, score, analysis = future.result()
                        similarity_scores[file_path] = {"score": score, "analysis": analysis}
                    except Exception as exc:
                        print(f'{file_path} generated an exception: {exc}')
            
                # Sort similarity scores in descending order
                sorted_similarity_scores = dict(sorted(similarity_scores.items(), key=lambda item: item[1]['score'], reverse=True))


                # Prepare data for the table
                data = {
                    "Resume Name": [os.path.basename(file_path) for file_path in sorted_similarity_scores.keys()],
                    "Similarity Score": [details['score'] for details in sorted_similarity_scores.values()],
                    "Analysis": [details["analysis"] for details in sorted_similarity_scores.values()],
                    "File Path": list(sorted_similarity_scores.keys())
                }

                # Create DataFrame
                df = pd.DataFrame(data)


                # Store DataFrame in session state
                st.session_state['df'] = df

                # If Top K is enabled, display only the top K entries
                if top_k_toggle and top_k is not None:
                    df = df.head(top_k)


                # If Percentage Criteria is enabled, filter by percentage
                if percentage_toggle and percentage_criteria is not None:
                    print("percent toggle")
                    df = df[df["Similarity Score"] >= (percentage_criteria)]

                # Display similarity scores as a table
                st.write("Similarity Scores:")
                st.session_state['modifed_after_process'] = df
        
                md_df = df.drop(columns=['File Path'])
                md_df['Similarity Score'] = md_df['Similarity Score'].apply(lambda x: f"{x}%")
    
                st.table(md_df)

        else:
                st.warning("Please upload files and enter a job description")

    if filter_button and 'df' in st.session_state:
        df = st.session_state['df']

        # If Top K is enabled, display only the top K entries
        if top_k_toggle and top_k is not None:
            df = df.head(top_k)
        # If Percentage Criteria is enabled, filter by percentage
        if percentage_toggle and percentage_criteria is not None:
            df = df[df["Similarity Score"] >= (percentage_criteria)]

        st.session_state['modifed_after_process'] = df
        # Display the filtered table
        st.write("Filtered Similarity Scores:")
        md_df = df.drop(columns=['File Path'])
        md_df['Similarity Score'] = md_df['Similarity Score'].apply(lambda x: f"{x}%")
        st.table(md_df)


    if 'modifed_after_process' in st.session_state:
        df = st.session_state['modifed_after_process']
        for index, row in df.iterrows():
            if st.button(f"View {row['Resume Name']}", key=index):
                with open(row['File Path'], "rb") as f:
                    binary_data = f.read()
                pdf_viewer(input=binary_data, width=700)



if __name__ == "__main__":

    load_dotenv()
    main()