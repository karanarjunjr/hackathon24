from flask import Flask, request, jsonify

app = Flask(__name__)


## Required Imports
from gensim.models import Word2Vec
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

## Code Begins
# defining resume and jd, using string for now, to be replaced with file read functional
resume_strings=["""HRPERSONNELASSISTANT
Summary
Iama U.S.citizenwho isauthorized to work in the US forany employer. I have worked 8 yearsasanOffice Clerk, 2 yearsasa Student
Intern/Office Assistant,and 4 yearsasa Contractor. Iamapplying for the Data EntryClerk position (Advert ID# 224278 Advert ID# 224278).
My skillsand experiences include:Administrative Support, Auditing, File Management, Meeting Facilitation, Office Materials Management, &
InventoryManagement.
Highlights
COMPUTERSKILLS:Microsoft Word, MS Excel, MS Outlook, MS PowerPoint, PeopleSoft. TYPINGSKILLS: 40-60 WPM.
ADDITIONAL SKILLS:Administrative Support, Auditing, Clerical, Copy, Customer Service, Data Entry, Delivery, Documentation, Fax, File
Management, Letters, Meeting Facilitation, OrganizationalSkills, Proofreading, Receptionist, Research, Scanning, Scheduling, Secretarial,
Telephone Skills, Office Equipment Maintenance,and InventoryManagement.
Experience
CompanyName City , State HRPersonnelAssistant 03/2013 to 04/2014
Supported human resources staffwith newhire orientationsand monthly departmentalmeetings.
Entered updated personneland subcontractor datainto acentral database.
Reviewed human resource paperwork foraccuracy and completeness by verifying,collecting and correcting additionalemployee data.
Composed and drafted all outgoing correspondenceand reports for managers.
Answered and managed incoming and outgoing calls whilerecording accurate messages.
Maintained Heavy Filing ofHRPersonnelAction Forms to include newhireletters,awards,certificationsand promotions, providing a
stronger framework forconsistency and detail.
Assisted Senior Personnelsupervisors inCoordinating and conducted newhire pre-interviews.
Developed and maintained an internal newhirefiling system.
Received, maintained and filed appropriate paperwork as back up forallHROfficetransactions.
Typed memorandums, forms,and letters using softwareapplications to complete HRofficeassignmentsand projects given on aregular
basis.
Utilized and Navigated PeopleSoft (HRS) to input, manageand research data.
Operated fax and copymachines to completetasksas needed.
Initiated and maintained emailcorrespondence with teachers, principals, VPs,and other members ofNPS regarding personalID
information, teaching background &certification info,and criminal history clearance.
CompanyName City , State Assistant Store Clerk 05/2011 to 01/2012
Sellingmerchandiseand providing customer services for patrons.
Monitoring patrons to assist thestorein lossand theft prevention.
Restocking shelves.
Improving data maintenance of produceinventory.
Maintaining theappearance ofthestore.
Assistingwithmaintenance of produceinventory
CompanyName City , State Office Clerk/ FederalContractor 06/2008 to 10/2009
Checking and validatingRetirement&NRC files inNFTS and CIS systems.
CallingApplicant to Counter Windowto SignCertificate.
Completing File Maintenance ofRiders, Mergers,and Certificates.
Finishing and documenting 3-BoxAudit for Weekly Systems Update.
Filing ofFBI background report onApplicants.
Assign scheduled appointments to appointed legal officers.
ClericalReception ofScheduled Applicants.
AuditingConfidentialDepartment Files.
Composing&Mailing outappointment letters.
Finalizing Pick list Printouts for Examinations/Cases.
CompletingNFTS Data Systems updates.
PerformingClericaltasks regardingNaturalization ceremonies, including:A. Scheduling, Preparing,and Distribution ofCertificates. B.
Retrieval ofDocumentation fromApplicants.
UsingComputer Terminalto update DHS Automated Systems.
ProvidingAdministrative Support in Processing.
PerformingDaily tasks offiling, recording,copying,and maintaining records.
CompanyName City , State Office Clerk/FederalContractor 07/2005 to 01/2007
PerformingData Entry ofLegalDocumentsand Court Case Dates.
MailClerk and Online MailInformationUpdating
Setting Schedules for Staff or Private Meetings
Filing and Faxing ofConfidentialCourt Documents
Metering outgoing and stamping incomingmail
Reestablishing protocols for officesupplies storageand maintenance
Providing clericalsupport to Officeadministrators &Managers.
Education
Bachelor's ofArts :PoliticalScienceand Law2007 Montclair State University , City , State, US Bachelor's ofArts Degreein PoliticalScience
and LawMontclair State University - Upper Montclair, NJ 1998 to 2007
High SchoolDiploma 1998 Our Lady ofGood CounselHigh School, City , State, US High SchoolDiploma Our Lady ofGood CounselHigh
School- Newark, NJ 1995 to 1998
Skills
Filing, Clerk, Maintenance, Clerical, Office Clerk, ClericalSupport, Data Entry, Faxing, IncomingMail, LegalDocuments, MailClerk, Office
Supplies, Stamping, Administrative Support, Audit, Auditing, Cases, Copying, Documentation, Documenting, Fbi, File, Finishing, Nrc, Scheduling,
Basis, Correspondence, Database, Hr, HumanResources, Peoplesoft, Teaching, Inventory, 60 Wpm, Customer Service, Equipment
Maintenance, Excel, File Management, Materials Management, Microsoft Word, Ms Excel, Ms Outlook, Ms Powerpoint, Office Assistant,
OrganizationalSkills, Outlook, Powerpoint, Proofreading, Receptionist, RetailSales, Scanning, Secretarial, Telephone, Telephone Skills, Typing, Word"""]



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



resume_str = []
def pdf2string(resume_paths):
    for path in resume_paths:
        text = high_level.extract_text(path)
        str_list = text.split()
        str_list = str_list[:]
        string = ' '.join(str_list)
        resume_str.append(string)
    return resume_str

#extract text from doc file function

jd_str = []
def doc2string(jd_paths):
    for path in jd_paths:
        document = Document(path)
        fullText = []
        for para in document.paragraphs:
            fullText.append(para.text)
        string = ' '.join(fullText)
        jd_str.append(string)
    return jd_str

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

resume_strings=preprocess(resume_strings)
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

# extracting skills from resume

annotations_resume = skill_extractor.annotate(resume_strings[0])
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

# using sentence transformer to calculate similarity
# Token Embeddings for resume
resume_model_output, resume_encoded_input = tokenize_and_embed(skills_resume)
resume_embeddings = resume_model_output.last_hidden_state
# print(f"Token embeddings shape: {resume_embeddings.size()}")
# Token Embeddings for JD
jd_model_output , jd_encoded_input = tokenize_and_embed(skills_jd)
jd_embeddings = jd_model_output.last_hidden_state
# print(f"Token embeddings shape: {jd_embeddings.size()}")

# normalize resume embeddings

resume_embeddings = mean_pooling(resume_model_output, resume_encoded_input["attention_mask"])
# Normalize the embeddings
resume_embeddings = F.normalize(resume_embeddings, p=2, dim=1)
# print(f"Resume embeddings shape: {resume_embeddings.size()}")

# normalize jd embeddings

jd_embeddings = mean_pooling(jd_model_output, jd_encoded_input["attention_mask"])
# Normalize the embeddings
jd_embeddings = F.normalize(jd_embeddings, p=2, dim=1)
# print(f"JD embeddings shape: {jd_embeddings.size()}")

#calculate cosine similarity between resume and jdembeddings

resume_embeddings = resume_embeddings.detach().numpy()
jd_embeddings = jd_embeddings.detach().numpy()

scores = np.zeros((resume_embeddings.shape[0], jd_embeddings.shape[0]))

for idx in range(resume_embeddings.shape[0]):
    scores[idx, :] = cosine_similarity([resume_embeddings[idx]], jd_embeddings)[0]


average_cosine_similarity = np.mean(scores)
print("Average Cosine Similarity:", average_cosine_similarity)

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
    sample_results = [
        {
            'id': 1,
            'score': average_cosine_similarity,
            'path': '/path/to/resume1.pdf'
        }
    ]

    # Count the number of results that match the threshold
    count_matches = sum(1 for result in sample_results if result['score'] >= threshold)

    # Construct the response JSON
    response_data = {
        'status': 'success',
        'count': count_matches,
        'metadata': {
            'confidenceScore': threshold
        },
        'results': sample_results
    }

    # Return the response as JSON
    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(debug=True)
