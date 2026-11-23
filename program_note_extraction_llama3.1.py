import gc
import torch
import os
import argparse
os.environ["TRANSFORMERS_CACHE"] = "/path/to/.cache/huggingface/models"
os.environ["HF_HOME"] = "/path/to/.cache/huggingface"
from dotenv import load_dotenv
load_dotenv()

from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

model_name = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline, TextIteratorStreamer

tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          cache_dir='/path/to/.cache/huggingface',
                                          token=os.environ.get("HF_TOKEN"))

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=os.environ.get("HF_TOKEN"),
    cache_dir='/path/to/.cache/huggingface',
    device_map="auto",
    torch_dtype=torch.float16,
    rope_scaling={"factor": 8.0, "type": "dynamic"}
)

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
        max_new_tokens = 8000,
        eos_token_id = terminators,
        do_sample = True,
        temperature = 0.03,
        top_p = 0.9,
        repetition_penalty = 1.1,
        return_full_text = False,
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

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
set_debug(False)

# Not required
os.environ.setdefault("COHERE_API_KEY", "")  # set COHERE_API_KEY in your .env

def parse_args():
    parser = argparse.ArgumentParser(description="Process program notes data with Llama 3.1.")
    parser.add_argument('--start_index', type=int, required=True, help='The start index for processing')
    parser.add_argument('--end_index', type=int, required=True, help='The end index for processing')
    parser.add_argument('--last_processed_file', type=str, default='/path/to/Extractions_llama3.1/last_processed_index.txt',
                        help='File to store the last processed index')
    return parser.parse_args()

# Function to save last processed index
def save_last_processed_index(index, file_path):
    with open(file_path, 'w') as f:
        f.write(str(index))

# Function to load last processed index from the file
def load_last_processed_index(last_processed_file_path, start_index):
    if os.path.exists(last_processed_file_path):
        with open(last_processed_file_path, 'r') as f:
            return int(f.read().strip())
    return start_index

def format_time(seconds):
    """
    Converts time in seconds to a human-readable format (hours:minutes:seconds).
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap= 0)
    all_splits = text_splitter.split_documents(documents)

    # This is the embedding model to embed the document chunks and store them in the vector database
    model_name = "sentence-transformers/all-mpnet-base-v2"
    cache_folder = '/path/to/Huggingface_cache/models'
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
    naive_retriever = db_faiss.as_retriever(search_type="similarity", search_kwargs={ "k" : 1})
    # This is the Cohere Reranker retriever which selects the top 5 most relevant out of the above 10
    # compressor = CohereRerank(top_n=5)
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor,
    #     base_retriever=naive_retriever
    # )
    
    # Langchain chain setup
    setup_and_retrieval = RunnableParallel({"question": RunnablePassthrough(), "context": naive_retriever })
    output_parser = StrOutputParser()
    llm = load_llm_pipeline()
    retrieval_chain = setup_and_retrieval | prompt | llm | output_parser
    
    
    # This is the query that is used in the similarity search
    query = f"Please provide the complete program note for {composition_name} by {composer_name}{formatted_movement}. Do not include any starting sentences such as 'Here is the complete program note for ...'. If you do not find any program note description for the requested music composition, please just respond with: 'Not Available'."
    print(f"Starting extraction for {composition_name} by {composer_name}.")
    result = retrieval_chain.invoke(query)
    print(f"Done for {composition_name} by {composer_name}.")

    with torch.no_grad():
        del query
        # del compression_retriever
        del naive_retriever
        del embeddings
        # del compressor
        del documents
        del retrieval_chain
        del db_faiss
        del llm
    
    return result

def save_last_processed_index(index, file_path):
    with open(file_path, 'w') as f:
        f.write(str(index))

def load_last_processed_index(last_processed_file_path, start_index):
    if os.path.exists(last_processed_file_path):
        with open(last_processed_file_path, 'r') as f:
            return int(f.read().strip())
    return start_index

def process_json_file(json_file_path, start_index, end_index, last_processed_file):
    total_time = 0
    json_output = []
    last_processed_index = load_last_processed_index(last_processed_file, start_index)
    begin_index = last_processed_index + 1

    total_start_time = time.time()
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    for i, program in enumerate(data.get('programs', [])[begin_index:end_index], start = begin_index):
        current_index = i
        count = 0
        start_time = time.time()

        # Creating a .txt file for the pages
        txt_file_name = f"/path/to/Extractions_llama3.1/text_files/program_{current_index}.txt"
        if not create_text_file(program.get('pages', []), txt_file_name):
            continue

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

        for work in program.get('works', []):
            if work.get('interval') is not None:
                continue

            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()
            conductor_name = work.get('conductorName', None)
            if conductor_name is not None:
                formatted_conductor_name = format_composer_name(conductor_name)
            composer_name = work.get('composerName', None)
            composition_name = work.get('workTitle', None)
            if isinstance(composition_name, dict):
                print(f"Skipping this {composition_name} because it is a dict. I will consider these cases later.")
                continue
            formatted_composer_name = format_composer_name(composer_name)
            
            formatted_composition_name = format_title(composition_name)
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

        json_file_name = f'/path/to/Extractions_llama3.1/json_files/extracted_program_notes_{begin_index}_{current_index}.json'
        with open(json_file_name, 'w') as outfile:
            json.dump(json_output, outfile, indent=4)

        save_last_processed_index(current_index, last_processed_file)

        elapsed_time = time.time() - start_time
        print(f"Time taken for extraction of program {current_index}: {format_time(elapsed_time)}")

        # Checkpoint for every 50 programs
        if (current_index - begin_index + 1) % 50 == 0:
            total_elapsed_time = time.time() - total_start_time
            count += 1
            total_time += total_elapsed_time
            print(f"Total time for programs from index {current_index-49} to {current_index} is: {format_time(total_elapsed_time)}")
            total_start_time = time.time()  # Reset total start time for next batch of 50 programs

        previous_json_file_name = f'/path/to/Extractions_llama3.1/json_files/extracted_program_notes_{begin_index}_{current_index - 1}.json'
        if os.path.exists(previous_json_file_name):
            os.remove(previous_json_file_name)
        
    print(f"Total processing time for last {count*50} programs is: {format_time(total_time)}")
        
        # Save the extracted data to a JSON file
        # with open('hpc-share/Extractions/json_files/extracted_program_notes_4001_5000.json', 'w') as outfile:
        #     json.dump(json_output, outfile, indent=4

if __name__ == "__main__":
    json_file_path = '/path/to/program_notes_data/nyphilharmonic_programs.json'  # Update with the path to your JSON file
    args = parse_args()
    process_json_file(json_file_path, 
    start_index=args.start_index, 
    end_index=args.end_index,
    last_processed_file=args.last_processed_file)