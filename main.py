import os
import zipfile
import tempfile
import shutil
import stat
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import git
from urllib.parse import urlparse
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'.py', '.js', '.java', '.ts', '.tsx', '.jsx', '.cpp', '.c', '.h', '.hpp', 
                     '.go', '.rb', '.php', '.cs', '.swift', '.kt', '.rs', '.scala', '.sh', '.sql',
                     '.html', '.css', '.vue', '.svelte', '.dart', '.r', '.m', '.pl', '.lua'}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file
MAX_TOTAL_FILES = 1000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

app = Flask(__name__)
CORS(app)

# Create upload folder
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

class CodebaseProcessor:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        
    def allowed_file(self, filename):
        """Check if file extension is allowed"""
        return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS
    
    def is_binary_file(self, filepath):
        """Check if file is binary"""
        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(1024)
                return b'\0' in chunk
        except:
            return True
    
    def extract_zip(self, zip_path, extract_to):
        """Extract zip file safely"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Security check: prevent zip bomb and path traversal
                for member in zip_ref.namelist():
                    if os.path.isabs(member) or ".." in member:
                        raise ValueError(f"Unsafe path in zip: {member}")
                zip_ref.extractall(extract_to)
            logger.info(f"Successfully extracted zip to {extract_to}")
        except Exception as e:
            logger.error(f"Failed to extract zip: {e}")
            raise
    
    def remove_readonly(self, func, path, excinfo):
        """Remove read-only files on Windows"""
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except:
            pass
    
    def validate_github_url(self, url):
        """Validate GitHub repository URL"""
        pattern = r'^https://github\.com/[\w\-\.]+/[\w\-\.]+/?$'
        return re.match(pattern, url) is not None
    
    def clone_github_repo(self, repo_url, dest_dir):
        """Clone GitHub repository"""
        try:
            if not self.validate_github_url(repo_url):
                raise ValueError("Invalid GitHub repository URL")
            
            logger.info(f"Cloning repository: {repo_url}")
            git.Repo.clone_from(repo_url, dest_dir, depth=1)  # Shallow clone
            
            # Remove .git directory to save space
            git_dir = os.path.join(dest_dir, '.git')
            if os.path.exists(git_dir):
                shutil.rmtree(git_dir, ignore_errors=True, onerror=self.remove_readonly)
            
            logger.info("Repository cloned successfully")
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            raise
    
    def collect_code_files(self, directory):
        """Collect code files from directory"""
        code_files = []
        file_count = 0
        
        for root, dirs, files in os.walk(directory):
            # Skip common directories that don't contain source code
            dirs[:] = [d for d in dirs if not d.startswith('.') and 
                      d not in ['node_modules', '__pycache__', 'venv', 'env', 'dist', 'build']]
            
            for file in files:
                if file_count >= MAX_TOTAL_FILES:
                    logger.warning(f"Reached maximum file limit: {MAX_TOTAL_FILES}")
                    break
                    
                filepath = os.path.join(root, file)
                
                # Check file extension and size
                if self.allowed_file(file) and os.path.getsize(filepath) <= MAX_FILE_SIZE:
                    if not self.is_binary_file(filepath):
                        code_files.append(filepath)
                        file_count += 1
            
            if file_count >= MAX_TOTAL_FILES:
                break
        
        logger.info(f"Collected {len(code_files)} code files")
        return code_files
    
    def prepare_documents(self, files):
        """Prepare documents for embedding"""
        docs = []
        
        for filepath in files:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    if content.strip():  # Skip empty files
                        # Create relative path for better readability
                        rel_path = os.path.relpath(filepath)
                        
                        # Add file metadata as context
                        file_ext = os.path.splitext(filepath)[1]
                        doc_content = f"File: {rel_path}\nType: {file_ext}\n\n{content}"
                        
                        docs.append(Document(
                            page_content=doc_content,
                            metadata={
                                'source': rel_path,
                                'file_type': file_ext,
                                'file_size': len(content)
                            }
                        ))
            except Exception as e:
                logger.warning(f"Failed to process file {filepath}: {e}")
                continue
        
        logger.info(f"Prepared {len(docs)} documents")
        return docs
    
    def create_embeddings(self):
        """Create embeddings model"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embeddings model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embeddings model: {e}")
            raise
    
    def store_embeddings(self, docs):
        """Store documents in vector database"""
        try:
            if not self.embeddings:
                self.create_embeddings()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(docs)
            logger.info(f"Split into {len(split_docs)} chunks")
            
            # Create vector store
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
            logger.info("Vector store created successfully")
            
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise
    
    def load_llm(self):
        """Load language model"""
        try:
            # Try different model names in order of preference
            models_to_try = ["llama3.2", "llama2", "codellama", "mistral"]
            
            for model_name in models_to_try:
                try:
                    llm = Ollama(
                        model=model_name,
                        temperature=0.1,
                        top_p=0.9,
                        num_ctx=4096
                    )
                    # Test the model
                    llm.invoke("Hello")
                    logger.info(f"Successfully loaded model: {model_name}")
                    return llm
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {e}")
                    continue
            
            raise Exception("No available Ollama models found. Please install Ollama and pull a model.")
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise
    
    def create_qa_chain(self, vector_store):
        """Create QA chain"""
        try:
            llm = self.load_llm()
            
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}
            )
            
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                verbose=True
            )
            
            logger.info("QA chain created successfully")
            return chain
        except Exception as e:
            logger.error(f"Failed to create QA chain: {e}")
            raise

# Global processor instance
processor = CodebaseProcessor()

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(413)
def handle_large_file(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'version': '1.0.0'})

@app.route('/upload', methods=['POST'])
def upload_codebase():
    try:
        zip_file = request.files.get('zip')
        repo_url = request.form.get('repo_url')
        
        if not zip_file and not repo_url:
            return jsonify({'error': 'Please provide either a ZIP file or GitHub repository URL.'}), 400
        
        # Create temporary directory for extraction
        extract_dir = tempfile.mkdtemp(prefix='codebase_')
        code_files = []
        
        try:
            # Process ZIP file
            if zip_file and zip_file.filename:
                if not zip_file.filename.endswith('.zip'):
                    return jsonify({'error': 'Only ZIP files are supported.'}), 400
                
                filename = secure_filename(zip_file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                zip_file.save(save_path)
                
                processor.extract_zip(save_path, extract_dir)
                code_files.extend(processor.collect_code_files(extract_dir))
                
                # Clean up zip file
                os.remove(save_path)
            
            # Process GitHub repository
            if repo_url:
                repo_extract = tempfile.mkdtemp(prefix='repo_')
                try:
                    processor.clone_github_repo(repo_url, repo_extract)
                    code_files.extend(processor.collect_code_files(repo_extract))
                finally:
                    shutil.rmtree(repo_extract, onerror=processor.remove_readonly)
            
            if not code_files:
                return jsonify({'error': 'No valid code files found in the provided source.'}), 400
            
            # Process documents
            docs = processor.prepare_documents(code_files)
            if not docs:
                return jsonify({'error': 'No readable code files found.'}), 400
            
            # Create vector store
            vector_store = processor.store_embeddings(docs)
            
            # Save vector store
            db_path = os.path.join(app.config['UPLOAD_FOLDER'], "vector_db")
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
            vector_store.save_local(db_path)
            
            return jsonify({
                'status': 'success',
                'message': 'Codebase processed successfully',
                'num_files': len(code_files),
                'num_documents': len(docs)
            })
            
        finally:
            # Clean up extraction directory
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir, onerror=processor.remove_readonly)
                
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query_codebase():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required.'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty.'}), 400
        
        # Load vector store
        db_path = os.path.join(app.config['UPLOAD_FOLDER'], "vector_db")
        if not os.path.exists(db_path):
            return jsonify({'error': 'No codebase uploaded. Please upload a codebase first.'}), 400
        
        try:
            if not processor.embeddings:
                processor.create_embeddings()
            
            vector_store = FAISS.load_local(db_path, processor.embeddings, allow_dangerous_deserialization=True)
            
            # Create QA chain
            qa_chain = processor.create_qa_chain(vector_store)
            
            # Enhanced query with context
            enhanced_query = f"""
            As a code analysis expert, please analyze the following query about the codebase:
            
            Query: {query}
            
            Please provide:
            1. A detailed analysis of the relevant code
            2. Specific recommendations or answers
            3. Code examples if applicable
            4. Best practices or potential improvements
            
            Focus on being practical and actionable in your response.
            """
            
            # Get response
            result = qa_chain.invoke({"query": enhanced_query})
            
            response_data = {
                'result': result['result'],
                'source_documents': [
                    {
                        'source': doc.metadata.get('source', 'Unknown'),
                        'content_preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                    }
                    for doc in result.get('source_documents', [])
                ]
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return jsonify({'error': f'Failed to process query: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Query error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Code Optimizer server...")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    
    # Ensure Ollama is available
    try:
        test_llm = processor.load_llm()
        logger.info("Ollama is available and ready")
    except Exception as e:
        logger.warning(f"Ollama may not be available: {e}")
        logger.info("Please ensure Ollama is installed and running with a model pulled")
    
    app.run(host='0.0.0.0', port=8080, debug=False)