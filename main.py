import os
import zipfile
import tempfile
import shutil
import stat
import logging
from io import BytesIO
import requests
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'.py', '.js', '.java', '.ts', '.tsx', '.jsx', '.cpp', '.c',
                      '.h', '.hpp', '.go', '.rb', '.php', '.cs', '.swift', '.kt',
                      '.rs', '.scala', '.sh', '.sql', '.html', '.css', '.vue',
                      '.svelte', '.dart', '.r', '.m', '.pl', '.lua'}
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_TOTAL_FILES = 1000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = os.path.abspath('uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

class CodebaseProcessor:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None

    def allowed_file(self, filename):
        return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

    def is_binary_file(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                return b'\0' in f.read(1024)
        except:
            return True

    def extract_zip(self, zip_path, extract_to):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)

    def remove_readonly(self, func, path, _):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    def clone_github_repo(self, repo_url, dest_dir):
        parts = repo_url.rstrip('/').split('/')
        owner, repo = parts[-2], parts[-1]
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"
        r = requests.get(zip_url, timeout=30)
        if r.status_code == 404:
            zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip"
            r = requests.get(zip_url, timeout=30)
        if r.status_code != 200:
            raise Exception(f"Failed to download repo zip: {r.status_code}")
        with zipfile.ZipFile(BytesIO(r.content)) as z:
            z.extractall(dest_dir)

    def collect_code_files(self, directory):
        code_files = []
        count = 0
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                if count >= MAX_TOTAL_FILES:
                    break
                path = os.path.join(root, f)
                size = os.path.getsize(path)
                if self.allowed_file(f) and size <= MAX_FILE_SIZE and not self.is_binary_file(path):
                    code_files.append(path)
                    count += 1
            if count >= MAX_TOTAL_FILES:
                break
        return code_files

    def prepare_documents(self, files):
        docs = []
        for path in files:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if not content.strip():
                    continue
                rel = os.path.relpath(path, app.config['UPLOAD_FOLDER'])
                ext = os.path.splitext(path)[1]
                docs.append(Document(
                    page_content=f"File: {rel}\nType: {ext}\n\n{content}",
                    metadata={'source': rel, 'file_type': ext}
                ))
            except:
                continue
        return docs

    def create_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def store_embeddings(self, docs):
        if not self.embeddings:
            self.create_embeddings()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=['\n\n', '\n', ' ', '']
        )
        chunks = splitter.split_documents(docs)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        return self.vector_store

    def load_llm(self):
        for name in ('mistral', 'llama3.2', 'llama2', 'codellama'):
            try:
                llm = Ollama(model=name, temperature=0.1, top_p=0.9, num_ctx=4096)
                llm.invoke('ping')
                return llm
            except:
                continue
        raise Exception('No Ollama model found')

    def create_qa_chain(self, vs):
        llm = self.load_llm()
        retriever = vs.as_retriever(search_type='similarity', search_kwargs={'k': 6})
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )

processor = CodebaseProcessor()

@app.errorhandler(413)
def handle_large(e):
    return jsonify({'error':'File too large'}), 413

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'healthy'})

@app.route('/upload', methods=['POST'])
def upload_codebase():
    extract_dir = tempfile.mkdtemp(prefix='codebase_')
    try:
        zip_file = request.files.get('zip')
        repo_url = request.form.get('repo_url')
        if zip_file and zip_file.filename.lower().endswith('.zip'):
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(zip_file.filename))
            zip_file.save(save_path)
            processor.extract_zip(save_path, extract_dir)
            os.remove(save_path)
        elif repo_url:
            processor.clone_github_repo(repo_url, extract_dir)
        else:
            return jsonify({'error':'Provide zip or repo_url'}), 400
        code_files = processor.collect_code_files(extract_dir)
        if not code_files:
            return jsonify({'error':'No readable code files'}), 400
        docs = processor.prepare_documents(code_files)
        if not docs:
            return jsonify({'error':'No readable code files'}), 400
        vs = processor.store_embeddings(docs)
        dbp = os.path.join(app.config['UPLOAD_FOLDER'], 'vector_db')
        if os.path.exists(dbp):
            shutil.rmtree(dbp)
        vs.save_local(dbp)
        return jsonify({'status':'success', 'num_files': len(code_files), 'num_documents': len(docs)})
    finally:
        shutil.rmtree(extract_dir, onerror=processor.remove_readonly)

@app.route('/query', methods=['POST'])
def query_codebase():
    data = request.get_json() or {}
    q = data.get('query', '').strip()
    if not q:
        return jsonify({'error':'Query required'}), 400
    dbp = os.path.join(app.config['UPLOAD_FOLDER'], 'vector_db')
    if not os.path.exists(dbp):
        return jsonify({'error':'Upload first'}), 400
    if not processor.embeddings:
        processor.create_embeddings()
    vs = FAISS.load_local(dbp, processor.embeddings, allow_dangerous_deserialization=True)
    chain = processor.create_qa_chain(vs)
    out = chain.invoke({'query': q})
    return jsonify({'result': out['result'], 'source_documents': [
        {'source': d.metadata.get('source', '?'), 'preview': d.page_content[:200] + '...'}
        for d in out.get('source_documents', [])
    ]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
