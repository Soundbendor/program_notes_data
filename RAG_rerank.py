import gc
import torch
import os
os.environ["TRANSFORMERS_CACHE"] = "/path/to/.cache/huggingface/models"
os.environ["HF_HOME"] = "/path/to/.cache/huggingface"

# Create a .env file and write your huggingface access token as HF_TOKEN = $$$
from dotenv import load_dotenv
load_dotenv()

# Langchain and transformers imports
from langchain.llms import HuggingFacePipeline
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline, TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

# More imports, mostly for Langchain setup
import time
import json
import os
from datetime import datetime
from transformers import AutoModel, AutoTokenizer

from textwrap import fill
from langchain_core.prompts import PromptTemplate

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

import locale
locale.getpreferredencoding = lambda: "UTF-8"
from langchain.vectorstores import FAISS

from langchain.embeddings import HuggingFaceEmbeddings

from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.globals import set_verbose, set_debug
set_debug(False) # If set True, you will see the Langchain chain generation process's intermediate steps and the similar chunks retrieved in the process.

# Download and cache the Llama 3 70B model and tokenizer
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          cache_dir='/path/to/.cache/huggingface',
                                          token='your_access_token_for_huggingface')

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token='your_access_token_for_huggingface',
    cache_dir='/path/to/.cache/huggingface',
    device_map="auto",
    torch_dtype=torch.float16,
)

# This function loads the required pipeline for Llama 3 with different parameters

def load_llm_pipeline():
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        max_new_tokens = 6000,
        eos_token_id = terminators,
        do_sample = True,
        temperature = 0.03,
        top_p = 0.9,
        repetition_penalty = 1.1,
        return_full_text = False,
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

os.environ["COHERE_API_KEY"] = "your_Cohere_API_key"

def create_text_file(pages, file_name):
    # Function to write all pages' text to a single .txt file
    if not pages:  # Check if the pages list is empty
        print(f"No pages available to save for {file_name}. Skipping file creation.")
        return False  # Indicates that no file was created

    merged_text = ""
    for page in pages:
        if page['text'] is not None:
            page_text = page['text'].replace('\r\n', '')
            merged_text += page_text

    if merged_text == "":
        print("The document is empty. Skipping file creation.")
        return False

    with open(file_name, 'w', encoding='utf-8') as text_file:
        text_file.write(merged_text)
    print(f"Saved text to {file_name}")
    return True  # This will indicate successful file creation

def format_composer_name(composer_name):
    """
    Reformats the composer name from 'Lastname, Firstname' to 'Firstname Lastname'.
    Handles excess whitespace by stripping and normalizing spaces.
    """
    parts = composer_name.split(',')
    if len(parts) == 2:
        first_name = ' '.join(parts[1].split())  # Split by whitespace and rejoin to remove extra spaces
        last_name = ' '.join(parts[0].split())  # Same approach for the last name
        return f"{first_name} {last_name}"
    else:
        return ' '.join(composer_name.split())  # Return normalized original if not in expected format

def format_title(title):
    """
    Converts an ALL CAPS title to Title Case, considering special cases like "X", "V" 'II'.
    """
    words = title.lower().split()
    title_cased_words = []

    # List of words which should remain in uppercase (like roman numerals, abbreviations, etc.)
    always_uppercase = ['II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

    for word in words:
        if word.upper() in always_uppercase:
            title_cased_words.append(word.upper())
        else:
            title_cased_words.append(word.capitalize())

    return ' '.join(title_cased_words)

# Function to extract program notes, a lot of things happen inside it
def extract_program_notes(file_path, composer_name, composition_name, movement = None):

    # To format the prompt to add movement into it
    formatted_movement = f" from the movement '{movement}'" if movement else ""

    # The main system prompt to the model with template required and supported by Llama-3
    prompt = PromptTemplate(
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> Respond to the input request based on your knowledge. Find all the relevant text that is part of a program note for {composition_name} by {composer_name}{formatted_movement}. There can be other program notes as well, but focus mainly on the starting of the program notes for {composition_name} by {composer_name}. To provide you more context, these texts were converted through Optical Character Recognition (OCR). Along with the very important concert program notes, there are some unwanted elements such as advertisements and irrelevant information in between. You need to get rid of such unwanted and irrelevant elements from the text. And sometimes you will find some OCR text conversion errors like invalid characters and misspellings. Please fix those as much as possible. Also, there might be discontinuations due to advertisements, page breaks, and the program note text can be split in different pages which you should identify and merge together. To reemphasize, your primary objective is to accurately extract the whole concert program note text for the {composition_name} by {composer_name} while disregarding unrelated content. Ensure to preserve the narrative of the description from start till the very end without any summarizing, rewriting, and missing out on relevant parts. Remember to filter out any content that are not part of the concert program notes, such as advertisements or generic announcements, ensuring the focus remains on the program note description. Use the following context to help: {context} <|eot_id|><|start_header_id|>user<|end_header_id|>
    Input: {question}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables = ["context", "question"],
    partial_variables = {"composer_name":composer_name, "composition_name":composition_name, "formatted_movement":formatted_movement}
    )

    # To load the document as .txt file
    loader = TextLoader(f"{file_path}",
                        encoding="utf8")
    documents = loader.load()

    # Splits the document into 5000 character chunks with 4000 character overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap= 4000)
    all_splits = text_splitter.split_documents(documents)

    # This is the embedding model to embed the document chunks and store them in the vector database
    model_name = "sentence-transformers/all-mpnet-base-v2"
    cache_folder = '/path/to/.cache/huggingface'
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    
    embeddings = HuggingFaceEmbeddings(
        cache_folder=cache_folder,
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs,
    )

    # FAISS vector database
    db_faiss = FAISS.from_documents(all_splits, embeddings)
    # This is the naive retriever that gets the k=10 most similar chunks
    naive_retriever = db_faiss.as_retriever(search_type="similarity", search_kwargs={ "k" : 10})
    # This is the Cohere Reranker retriever which selects the top 5 most relevant out of the above 10
    compressor = CohereRerank(top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=naive_retriever
    )
    
    # Langchain chain setup
    setup_and_retrieval = RunnableParallel({"question": RunnablePassthrough(), "context": compression_retriever })
    output_parser = StrOutputParser()
    llm = load_llm_pipeline()
    rerank_retrieval_chain = setup_and_retrieval | prompt | llm | output_parser
    
    
    # This is the query that is used in the similarity search
    query = f"Please provide the complete program note for {composition_name} by {composer_name}{formatted_movement}. Do not include any starting sentences such as 'Here is the complete program note for ...'. If you do not find any program note description for the requested music composition, please just respond with: 'Not Available'."
    print(f"Starting extraction for {composition_name} by {composer_name}.")
    result = rerank_retrieval_chain.invoke(query)
    print(f"Done for {composition_name} by {composer_name}.")

    # Delete variables from the GPU (not really effective)
    with torch.no_grad():
        del query
        del compression_retriever
        del naive_retriever
        del embeddings
        del compressor
        del documents
        del rerank_retrieval_chain
        del db_faiss
        del llm

    return result

# If model stops occasionally due to CUDA out of memory or other errors, it is nice to have the resuming point.
def save_last_processed_index(index, file_path='path/to/last_processed_index.txt'):
    with open(file_path, 'w') as f:
        f.write(str(index))

def load_last_processed_index(last_processed_file_path, start_index):
    if os.path.exists(last_processed_file_path):
        with open(last_processed_file_path, 'r') as f:
            return int(f.read().strip())
    return start_index

start_index = 0
# end_index = 13001 # Define end_index if you want to do it for some range of documents
last_processed_file_path = '/path/to/last_processed_index.txt'

# Function that does the bulk of the processing
def process_json_file(json_file_path):
    json_output = []
    # Start one index after the last processed index
    last_processed_index = load_last_processed_index(last_processed_file_path, start_index) + 1
    # Starting point (required for file creation)
    begin_index = last_processed_index
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    for i, program in enumerate(data.get('programs', [])[begin_index:], start = begin_index):

        # Makes sure you start from the correct index
        current_index = i

        # Creating a .txt file for the pages (can be removed after extraction is done)
        txt_file_name = f"/path/to/program_notes_data/Extractions/text_files/program_{current_index}.txt"

        # Check if the document is not empty
        if not create_text_file(program.get('pages', []), txt_file_name):
            continue
        
        # Create a dictionary of concert info
        concerts_info = []
        for concert in program.get('concerts', []):
            # Format the date
            formatted_date = datetime.strptime(concert['Date'], '%Y-%m-%dT%H:%M:%SZ').strftime('%m-%d-%Y')
            concerts_info.append({
                "eventType": concert.get('eventType'),
                "location": concert.get('Location'),
                "venue": concert.get('Venue'),
                "date": formatted_date,
                "time": concert.get('Time')
            })

        # Now, move through each musical work in a given program document
        for work in program.get('works', []):
            # Skip works that are actually 'intermissions'
            if work.get('interval') is not None:
                continue
            
            # Clear cache (important so that your GPU is free from garbage variables)
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()

            # Check if conductor name is in the data and format the name in firstName lastName
            formatted_conductor_name = None
            conductor_name = work.get('conductorName', None)
            if conductor_name is not None:
                formatted_conductor_name = format_composer_name(conductor_name)

            # Get the composer's name and work title
            composer_name = work.get('composerName', None)
            composition_name = work.get('workTitle', None)

            # Skip worktitle if it is a dict (to handle separately)
            if isinstance(composition_name, dict):
                print(f"Skipping this {composition_name} because it is a dict. I will consider these cases later.")
                continue
            
            # Format the composer's name and composition title using the functions
            formatted_composer_name = format_composer_name(composer_name)
            formatted_composition_name = format_title(composition_name)

            # Get the movement and pass it to the function
            movement = work.get('movement', None)
            extracted_note = extract_program_notes(txt_file_name, formatted_composer_name, formatted_composition_name, movement)

            # Extract soloists data
            soloists_info = []
            for soloist in work.get('soloists', []):
                soloists_info.append({
                    "soloistName": soloist.get('soloistName', None),
                    "soloistInstrument": soloist.get('soloistInstrument', None),
                    "soloistRoles": soloist.get('soloistRoles', None)
                })
    
            # Save program-level info
            json_output.append({
                "id": program.get('id'),
                "programID": program.get('programID'),
                "orchestra":program.get('orchestra'),
                "season": program.get('season'),
                "concerts": concerts_info,
                "workTitle": formatted_composition_name,
                "movement": movement,
                "composerName": formatted_composer_name,
                "conductorName": formatted_conductor_name,
                "soloists": soloists_info,
                "ProgramNote": extracted_note
            })
    
            # Optionally remove the .txt file if not needed after extraction
            # os.remove(txt_file_name)

            # Delete unused variable (doesn't help)
            with torch.no_grad():
                del extracted_note

        # Recursively update file name and its contents after each document such that if it stops, nothing is lost
        json_file_name = f'/path/to/program_notes_data/Extractions/json_files/extracted_program_notes_{begin_index}_{current_index}.json'
        with open(json_file_name, 'w') as outfile:
            json.dump(json_output, outfile, indent=4)

        # Save the index into the file
        save_last_processed_index(current_index)

        # Remove previous file if it exists after creation of the new one
        previous_json_file_name = f'/path/to/program_notes_data/Extractions/json_files/extracted_program_notes_{begin_index}_{current_index - 1}.json'
        if os.path.exists(previous_json_file_name):
            os.remove(previous_json_file_name)

json_file_path = '/path/to/program_notes_data/nyphilharmonic_programs.json'  # Update with the path to your JSON file
process_json_file(json_file_path)
