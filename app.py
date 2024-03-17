import os
import requests 
from flask import Flask, request, jsonify

app = Flask(__name__)


## Required Imports
from gensim.models import Word2Vec
import glob
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
from pdfminer import high_level
from docx import Document
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import PhraseMatcher

# load default skills data base
from skillNer.general_params import SKILL_DB
# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def actual(context,
    category,
    threshold,
    num_matches,
    input_path ):
    ## Code Begins
    # defining resume and jd, using string for now, to be replaced with file read functional
    
        folder_path = input_path
        no_of_matches = num_matches
        # threshold = threshold

        resume_strings=[]

        jd_strings=["""Job Responsibilities:
        Design, Develop and Architect highly scalable applications and research technologies to solve complex business problems.
        Develop reusable solutions that can be shared with multiple groups.
        Define opportunities across IT to maximize business impact and innovate engineering processes to reduce software construction and maintenance costs.
        Expected to contribute towards integrating complex platforms including several components with business domain and process context.
        Focus on building relevant engineering and business capabilities in the organization to keep pace with demand and best practices in the industry.
        Coordinate implementation activities across a broad range of functions and departments; work with client groups to identify, arrange, and/or deliver training needs.
        Lead organizational initiatives. Work with stakeholders to research new frameworks, tools & proof of concepts.
        Develop and lead focused groups and communities to facilitate technical discussions, source ideas, and provide engineering leadership.
        Essential Qualifications:
        6+ years of experience in building end-to-end business solutions using Big data technologies like HDFS, Hive, Kafka, Scala, Python and Spark.
        Demonstrated strength in data modeling, ETL development, and data warehousing.
        Preferring knowledge and hands-on experience working with Hadoop Ecosystem and Big data technologies like HDFS, Hive, Kafka, Spark, Scala and Python.
        Experience in API design & development.
        Candidate should have good knowledge of Linux and have ETL development, deployment and optimization experience using standard big data tools.
        Should have good understanding of Git, JIRA, Change / Release management,  build/deploy, CI/CD & Share Point.
        Continually develop depth and breadth in key competencies.
        Demonstrate curiosity towards learning and treat negative events as opportunities for learning.
        Ability to communicate clearly and concisely and use strong writing and verbal skills to communicate facts, figures, and ideas to others.
        Deliver effective presentations and talks.
        Desired Qualifications:
        Preferred certification in one of the major distributions like Cloudera / Hortonworks /Cloud.
        Preferred certification/working experience in cloud technologies.
        Expertise in banking domain (Good to have).
        """]

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        model_ckpt = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model = AutoModel.from_pretrained(model_ckpt)


        def tokenize_and_embed(text):
            encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                model_output = model(**encoded_input)
            return model_output, encoded_input

        def pdf2string(path):
            text = high_level.extract_text(path)
            str_list = text.split()
            str_list = str_list[:]
            string = ' '.join(str_list)
            return string


        def doc2string(path):
            document = Document(path)
            fullText = []
            for para in document.paragraphs:
                fullText.append(para.text)
            string = ' '.join(fullText)
            return string

        resume_path=[]

        def read_files(path):
            if os.path.isdir(path):  # If it's a local folder path
                read_files_in_folder(path)
            else:  # If it's a URL
                read_files_from_url(path)

    

        # reading files from folder
        def read_files_in_folder(folder_path):
            # # Ensure the folder path ends with a '/'
            if not folder_path.endswith('/'):
                folder_path += '/'

            # Use glob to find all PDF and DOC files in the folder
            pdf_files = glob.glob(folder_path + '*.pdf')
            docx_files = glob.glob(folder_path + '*.docx')

            # Read PDF files
            for pdf_file in pdf_files:
                text = pdf2string(pdf_file)
                resume_path.append(pdf_file)
                resume_strings.append(text)
                print(f"File: {pdf_file}\nContent:\n{text}\n")

            # Read DOCX files
            for docx_file in docx_files:
                text = doc2string(docx_file)
                resume_path.append(docx_file)
                resume_strings.append(text)
                print(f"File: {docx_file}\nContent:\n{text}\n")

        def read_files_from_url(url):
            response = requests.get(url)
            if response.status_code == 200:
                content_type = response.headers.get('content-type')
                if 'pdf' in content_type:
                    text = pdf2string(response.content)
                    resume_path.append(response.content)
                    resume_strings.append(text)
                    print(f"URL: {url}\nContent:\n{text}\n")
                elif 'docx' in content_type:
                    text = doc2string(response.content)
                    resume_path.append(response.content)
                    resume_strings.append(text)
                    print(f"URL: {url}\nContent:\n{text}\n")
                else:
                    print("Unsupported file type.")

        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        def preprocess(string_list):
            final_str_list=[]
            for text in string_list:
                words = word_tokenize(text)
                words = [w for w in words if not w in stop_words]
                words = [lemmatizer.lemmatize(w) for w in words]
                words = ' '.join(words)
                final_str_list.append(words)
            return final_str_list

        read_files(folder_path)

        jd_temp=[]
        def jdStr(context):
            jd_temp.append(context)
            return jd_temp


        resume_strings=preprocess(resume_strings)
        jd_strings=jdStr(context)
        jd_strings=preprocess(jd_strings)

        # init params of skill extractor
        nlp = spacy.load("en_core_web_lg")
        # init skill extractor
        skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

        # extracting skills from jd

        annotations = skill_extractor.annotate(jd_strings[0])
        full_matches=annotations['results']['full_matches']
        filtered_full_match =[item['doc_node_value'] for item in full_matches if item['score'] > 0.9]
        # Using set comprehension to remove duplicates and convert back to list
        unique_filtered_full = list({value for value in filtered_full_match})
        ngram_scored_matches=annotations['results']['ngram_scored']

        filtered_ngram_scored =[item['doc_node_value'] for item in ngram_scored_matches if item['score'] > 0.9]
        unique_filtered_ngram = list({value for value in filtered_ngram_scored})

        # joining array to get full list
        skills_jd=unique_filtered_full+unique_filtered_ngram
        print(skills_jd)

        skills_resume_list = []
        # extracting skills from resume
        for resume in resume_strings:
            annotations_resume = skill_extractor.annotate(resume)
            # Taking the full match from resume
            resume_full_matches=annotations_resume['results']['full_matches']
            filtered_resume_full=[item['doc_node_value'] for item in resume_full_matches if item['score'] > 0.9]
            # Using set comprehension to remove duplicates and convert back to list
            unique_filtered_full_resume = list({value for value in filtered_resume_full})

            resume_ngram_matches=annotations_resume['results']['ngram_scored']
            filtered_resume_ngram =[item['doc_node_value'] for item in resume_ngram_matches if item['score'] > 0.9]

            # Using set comprehension to remove duplicates and convert back to list
            unique_filtered_ngram_resume = list({value for value in filtered_resume_ngram})
            skills_resume=unique_filtered_full_resume+unique_filtered_ngram_resume
            print(skills_resume)
            skills_resume_list.append(skills_resume)

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        # tokenizing and encoding resume and jd

        # using sentence transformer to calculate similarity
        # Token Embeddings for resume

        # print(f"Token embeddings shape: {resume_embeddings.size()}")
        # Token Embeddings for JD
        skills_jd = ' '.join(skills_jd)
        jd_model_output , jd_encoded_input = tokenize_and_embed(skills_jd)
        jd_embeddings = jd_model_output.last_hidden_state
        # print(f"Token embeddings shape: {jd_embeddings.size()}")

        # normalize resume embeddings

        # print(f"Resume embeddings shape: {resume_embeddings.size()}")

        # normalize jd embeddings

        jd_embeddings = mean_pooling(jd_model_output, jd_encoded_input["attention_mask"])
        # Normalize the embeddings
        jd_embeddings = F.normalize(jd_embeddings, p=2, dim=1)
        # print(f"JD embeddings shape: {jd_embeddings.size()}")
        jd_embeddings = jd_embeddings.detach().numpy()


        resume_list= []
        for skills_resume in skills_resume_list:
            skills_resume = ' '.join(skills_resume)
            resume_list.append(skills_resume)

        resume_model_output , resume_encoded_input = tokenize_and_embed(resume_list)
        resume_embeddings = resume_model_output.last_hidden_state
        resume_embeddings = mean_pooling(resume_model_output, resume_encoded_input["attention_mask"])
        # Normalize the embeddings
        resume_embeddings = F.normalize(resume_embeddings, p=2, dim=1)
        resume_embeddings = resume_embeddings.detach().numpy()

        scores = np.zeros( (jd_embeddings.shape[0], resume_embeddings.shape[0]))
        for idx in range(jd_embeddings.shape[0]):
            scores[idx, :] = cosine_similarity([jd_embeddings[idx]], resume_embeddings)[0]


        result = []

        for idx in range(jd_embeddings.shape[0]):
            for jd_idx in range(resume_embeddings.shape[0]):
                if(scores[idx, jd_idx] > threshold):
                    result.append({"id":jd_idx,"score":scores[idx, jd_idx], "path": resume_path[jd_idx]})

        result = sorted(result, key=lambda d: d["score"], reverse=True)
        print(result[:no_of_matches])
        return result                           


@app.route('/', methods=['GET'])
def hello_world():
    message = {'message': 'Hello, World!'}
    return jsonify(message)


@app.route('/api/process', methods=['POST'])
def process_input():
    # Get the JSON data from the request
    data = request.get_json()

    # Check if all required keys are present in the JSON data
    required_keys = ['context', 'category', 'threshold', 'numberOfMatches', 'inputPath']
    if not all(key in data for key in required_keys):
        return jsonify({'error': 'Missing required keys in JSON data'}), 400

    # Accessing the values from the JSON data
    context = data['context']
    category = data['category']
    threshold = data['threshold']
    num_matches = data['numberOfMatches']
    input_path = data['inputPath']


    # Placeholder for actual processing logic
    # For this example, generating a sample response
    sample_results = actual(context, category, threshold, num_matches, input_path)


    # Count the number of results that match the threshold
    # count_matches = sum(1 for result in sample_results if result['score'] >= threshold)

    # Construct the response JSON
    response_data = {
        'status': 'success',
        'count': len(sample_results),
        'metadata': {
            'confidenceScore': threshold
        },
        'results': sample_results
    }

    # Return the response as JSON
    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=5000)



