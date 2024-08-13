from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy as np
import pymupdf
import json


client = OpenAI()
embeddings = OpenAIEmbeddings()



from string import Template
 
def new_llm_response(resume_text, job_description):
    prompt_template = Template("""
    You are an experienced HR professional with extensive experience in analyzing resumes based on Job Descriptions.
    Your task is to compare the following resume text against the given Job Description and provide a similarity score along with a brief analysis.
 
    ### Job Description:
    ${job_description}
 
    ### Resume:
    ${resume_text}
 
    ### Task:
    1. Compare the resume against the Job Description.
    2. Provide a similarity score for the resume (on a scale of 0-100%).
    3. Offer a brief analysis explaining the score.
 
    Format the output in JSON only:
    {
      Similarity Score: [Score] ,
      Analysis: [Details]
    }
 
    Note: Strongly follow the Output format.
    """)
 
    # Create the prompt by substituting the template with actual values
    prompt = prompt_template.substitute(
        resume_text=resume_text,
        job_description=job_description
    )
 
    # Generate the completion
    completion = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt}
        ]
    )

    ans = completion.choices[0].message.content
    j_ans= json.loads(ans)
    score = int(j_ans['Similarity Score'])
    analysis = j_ans['Analysis']
    return score , analysis





def llm_response(resume_text):
  prompt = f"""
        "You are an AI that extracts structured information from {resume_text}. "
        "Identify and extract the following sections from the resume if available: "
        "Work Experience, Skills, Certifications, Projects.\n"
        "If a section is not explicitly mentioned, try to infer it from the context.\n\n"
        "Format the output as:\n"
        "Work Experience:\n[Details]\n"
        "Skills:\n[Details]\n"
        "Certifications:\n[Details]\n"
        "Projects:\n[Details]\n\n"
        "Extracted Sections:"
    """
  completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
      {"role": "system", "content": prompt},
      {"role": "user", "content": resume_text}
    ]
  )
 
  return completion.choices[0].message.content


def load_resumes(file_path):
  resume_dic = {}
  for i in file_path:
       doc = pymupdf.open(i) # open a supported document
       all_text = ""
       for page in doc:
        all_text += page.get_text() + chr(12)
        resume_text = llm_response(all_text)
        resume_dic[i] = resume_text
        
  return resume_dic

def create_openai_embeddings(text):
  text_embeddings = client.embeddings.create(
    model="text-embedding-3-large",
    input=text,
    encoding_format="float"
  )
  embeded_text = text_embeddings.data[0].embedding
  return embeded_text



def cosine_similarity_np(vec1, vec2):
    # Ensure inputs are numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # Compute dot product
    dot_product = np.dot(vec1, vec2)

    # Compute magnitudes
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)

    # Compute cosine similarity
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return   0
    else:
        return dot_product / (magnitude_vec1 * magnitude_vec2)
