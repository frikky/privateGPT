import os
import time
import github 
import tempfile

from constants import CHROMA_SETTINGS

#from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Continue building out sources
## e.g. through scraping and like a googlebot
## Make a site that goes through, scrapes and returns you a db trained on the data
## App questions: Could it work?

chunk_size = 500
chunk_overlap = 50
persist_directory = "db"
model_name = "all-MiniLM-L6-v2"
#os.environ.get('EMBEDDINGS_MODEL_NAME')

def find_files(github_repo, path=""):
    # Image types
    allowed_extensions = [".md", ".MD", ".yml", ".yaml", ".json", ".txt"]
    banned_filetypes = [".png", ".jpg", ".jpeg", ".gif", ".svg", ".PNG", ".JPG", ".JPEG", ".GIF", ".SVG"]
    all_filedata = []

    # Loop all files in repo
    for file in github_repo.get_contents(path):
        # Check if file is directory
        newpath = file.name
        if len(path) > 0:
            newpath = path + "/" + file.name

        # Check if it ends with any of banNed filetypes
        if any(file.name.endswith(x) for x in banned_filetypes):
            continue

        if file.type == "dir":
            # Recurse
            dir_filedata = find_files(github_repo, newpath)
            all_filedata.extend(dir_filedata)
        else:
            # Check if it ends with any of allowed extensions
            if not any(file.name.endswith(x) for x in allowed_extensions):
                continue

            # naming workaround...
            # https://github.com/imartinez/privateGPT/issues/358
            # newpath = newpath.replace(".md", ".txt")

            #print("Processing file: %s" % newpath)
            # Load file and try with different encodings
            try:
                file_content = file.decoded_content.decode("utf-8")
            except: 
                try:
                    file_content = file.decoded_content.decode("latin-1")
                except:
                    try:
                        file_content = file.decoded_content.decode("utf-16")
                    except:
                        print("Could not decode file: %s" % newpath)
                        continue
                       
            print("Content of file %s is len %d" % (newpath, len(file_content)))
            all_filedata.append({
                "filename": file.name,
                "path": newpath,
                "content": file_content
            })

    return all_filedata

def load_files(all_files):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Just a shitty way to load it faster without referencing paths
    all_content = ""
    for file in all_files:
        all_content += file["content"] + "\n\n"

    with tempfile.NamedTemporaryFile(mode='w', delete=True) as temporary_file:
        print("Writing file %s" % file["path"])
        try:
            temporary_file.write(all_content)
            temporary_file.flush()
            loader = TextLoader(temporary_file.name)

            documents = loader.load()
            texts = text_splitter.split_documents(documents)
            print(f"Split into {len(texts)} chunk(s) of text (max. {chunk_size} tokens each)")

            return texts

        except Exception as e:
            print("Could not load file %s: %s" % (file["path"], e))

            return None

# Load github 
def load_github():
    gh = github.Github(os.environ['GITHUB_TOKEN'])

    repo_path = "shuffle/shuffle-docs"
    user_name, repo_name = repo_path.split("/")
    repo = gh.get_user(user_name).get_repo(repo_name)

    all_filedata = find_files(repo)
    print("Got %d files to handle" % len(all_filedata))

    returned_text = load_files(all_filedata)
    if returned_text is None:
        print("Could not load files")
        return None

    while True:
        try:
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            db = Chroma.from_documents(returned_text, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)

            db.persist()
            db = None

            break
        except requests.exceptions.ConnectionError as e:
            print("Timed out. Trying again: %s" % e)

def load_model():
    if not os.path.exists("db"):
        os.makedirs("db")

    modelname = "ggml-gpt4all-j-v1.3-groovy.bin"
    if not os.path.exists("models"):
        os.makedirs("models")
    
    if not os.path.exists("models/%s" % modelname):
        modelurl = "https://gpt4all.io/models/%s" % modelname
        print("Downloading model from url %s" % modelurl)

        request = requests.get(modelurl, stream=True)
        with open("models/%s" % modelname, "wb") as file:
            for chunk in request.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    file.flush()

def main():
    load_model()
    load_github()

if __name__ == "__main__":
    main()
