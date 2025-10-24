const ALLOWED_EXTENSIONS = new Set([
  '.py', '.js', '.java', '.ts', '.tsx', '.jsx', '.cpp', '.c',
  '.h', '.hpp', '.go', '.rb', '.php', '.cs', '.swift', '.kt',
  '.rs', '.scala', '.sh', '.sql', '.html', '.css', '.vue',
  '.svelte', '.dart', '.r', '.m', '.pl', '.lua'
]);

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const MAX_TOTAL_FILES = 1000;
const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 200;

let vectorStore = { documents: [], embeddings: [] };
let isProcessed = false;

class OpenAIClient {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.baseURL = 'https://api.openai.com/v1';
  }

  async createEmbedding(text) {
    const response = await fetch(`${this.baseURL}/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify({
        model: 'text-embedding-3-small',
        input: text
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`OpenAI API error: ${error.error?.message || response.statusText}`);
    }

    const data = await response.json();
    return data.data[0].embedding;
  }

  async createEmbeddings(texts) {
    const embeddings = [];
    const batchSize = 100;
    
    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const response = await fetch(`${this.baseURL}/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`
        },
        body: JSON.stringify({
          model: 'text-embedding-3-small',
          input: batch
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(`OpenAI API error: ${error.error?.message || response.statusText}`);
      }

      const data = await response.json();
      embeddings.push(...data.data.map(d => d.embedding));
    }

    return embeddings;
  }

  async chat(messages, temperature = 0.1) {
    const response = await fetch(`${this.baseURL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: messages,
        temperature: temperature,
        max_tokens: 2000
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`OpenAI API error: ${error.error?.message || response.statusText}`);
    }

    const data = await response.json();
    return data.choices[0].message.content;
  }
}

function cosineSimilarity(a, b) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function searchVectorStore(queryEmbedding, k = 6) {
  const similarities = vectorStore.embeddings.map((embedding, index) => ({
    index,
    similarity: cosineSimilarity(queryEmbedding, embedding)
  }));

  similarities.sort((a, b) => b.similarity - a.similarity);

  return similarities.slice(0, k).map(result => ({
    document: vectorStore.documents[result.index],
    similarity: result.similarity
  }));
}

function allowedFile(filename) {
  const ext = filename.substring(filename.lastIndexOf('.')).toLowerCase();
  return ALLOWED_EXTENSIONS.has(ext);
}

function isBinaryFile(content) {
  for (let i = 0; i < Math.min(1024, content.length); i++) {
    if (content.charCodeAt(i) === 0) return true;
  }
  return false;
}

async function extractZipFromFile(file) {
  const JSZip = window.JSZip;
  const zip = await JSZip.loadAsync(file);
  const files = [];

  for (const [path, zipEntry] of Object.entries(zip.files)) {
    if (zipEntry.dir || !allowedFile(path)) continue;
    
    try {
      const content = await zipEntry.async('string');
      if (content.length > MAX_FILE_SIZE) continue;
      if (isBinaryFile(content)) continue;
      if (!content.trim()) continue;

      files.push({ path, content });
      if (files.length >= MAX_TOTAL_FILES) break;
    } catch (e) {
      console.warn(`Failed to read ${path}:`, e);
    }
  }

  return files;
}

async function downloadGitHubRepo(repoUrl) {
  const parts = repoUrl.trim().replace(/\/$/, '').split('/');
  const owner = parts[parts.length - 2];
  const repo = parts[parts.length - 1];

  let zipUrl = `https://github.com/${owner}/${repo}/archive/refs/heads/main.zip`;
  let response = await fetch(zipUrl);

  if (!response.ok) {
    zipUrl = `https://github.com/${owner}/${repo}/archive/refs/heads/master.zip`;
    response = await fetch(zipUrl);
  }

  if (!response.ok) {
    throw new Error(`Failed to download repository: ${response.statusText}`);
  }

  const blob = await response.blob();
  return extractZipFromFile(blob);
}

function prepareDocuments(files) {
  const documents = [];

  for (const { path, content } of files) {
    const ext = path.substring(path.lastIndexOf('.'));
    const docContent = `File: ${path}\nType: ${ext}\n\n${content}`;
    
    documents.push({
      content: docContent,
      metadata: {
        source: path,
        file_type: ext
      }
    });
  }

  return documents;
}

function recursiveSplit(text, separators, sepIndex) {
  if (sepIndex >= separators.length) {
    const chunks = [];
    for (let i = 0; i < text.length; i += CHUNK_SIZE) {
      chunks.push(text.substring(i, i + CHUNK_SIZE));
    }
    return chunks;
  }

  const separator = separators[sepIndex];
  const parts = text.split(separator);
  const chunks = [];
  let currentChunk = '';

  for (const part of parts) {
    if (currentChunk.length + part.length + separator.length <= CHUNK_SIZE) {
      currentChunk += (currentChunk ? separator : '') + part;
    } else {
      if (currentChunk) {
        chunks.push(currentChunk);
      }

      if (part.length > CHUNK_SIZE) {
        const subChunks = recursiveSplit(part, separators, sepIndex + 1);
        chunks.push(...subChunks);
        currentChunk = '';
      } else {
        currentChunk = part;
      }
    }
  }

  if (currentChunk) {
    chunks.push(currentChunk);
  }

  return chunks;
}

function splitDocuments(documents) {
  const chunks = [];
  const separators = ['\n\n', '\n', ' ', ''];

  for (const doc of documents) {
    const content = doc.content;
    const metadata = doc.metadata;

    if (content.length <= CHUNK_SIZE) {
      chunks.push({ content, metadata });
      continue;
    }

    const parts = recursiveSplit(content, separators, 0);

    for (const part of parts) {
      chunks.push({ content: part, metadata });
    }
  }

  return chunks;
}

function setStatus(message, isError = false) {
  const statusEl = document.getElementById('uploadStatus');
  statusEl.textContent = message;
  statusEl.style.color = isError ? '#f85149' : 'var(--muted)';
}

function addMessage(content, isUser = false) {
  const chatWindow = document.getElementById('chatWindow');
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
  
  if (isUser) {
    messageDiv.textContent = content;
  } else {
    messageDiv.innerHTML = marked.parse(content);
  }
  
  chatWindow.appendChild(messageDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function uploadCodebase() {
  const apiKey = document.getElementById('apiKey').value.trim();
  const zipFile = document.getElementById('zipfile').files[0];
  const repoUrl = document.getElementById('repoUrl').value.trim();

  if (!apiKey) {
    setStatus('Please enter your OpenAI API key', true);
    return;
  }

  if (!zipFile && !repoUrl) {
    setStatus('Please provide a ZIP file or GitHub repo URL', true);
    return;
  }

  const uploadBtn = document.getElementById('uploadBtn');
  uploadBtn.disabled = true;
  setStatus('Processing...');

  try {
    const openaiClient = new OpenAIClient(apiKey);

    let files;
    if (zipFile) {
      setStatus('Extracting ZIP file...');
      files = await extractZipFromFile(zipFile);
    } else {
      setStatus('Downloading GitHub repository...');
      files = await downloadGitHubRepo(repoUrl);
    }

    if (files.length === 0) {
      throw new Error('No readable code files found');
    }

    setStatus(`Found ${files.length} files. Preparing documents...`);
    const documents = prepareDocuments(files);

    setStatus(`Splitting ${documents.length} documents into chunks...`);
    const chunks = splitDocuments(documents);

    setStatus(`Creating embeddings for ${chunks.length} chunks (this may take a while)...`);
    const texts = chunks.map(chunk => chunk.content);
    
    const embeddings = [];
    const batchSize = 100;
    
    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      setStatus(`Creating embeddings... ${i + batch.length}/${texts.length}`);
      const batchEmbeddings = await openaiClient.createEmbeddings(batch);
      embeddings.push(...batchEmbeddings);
    }

    setStatus('Storing embeddings...');
    vectorStore.documents = chunks;
    vectorStore.embeddings = embeddings;

    window.openaiClient = openaiClient;
    isProcessed = true;
    document.getElementById('queryBtn').disabled = false;
    setStatus(`Success! Processed ${files.length} files, ${chunks.length} chunks`);
    addMessage('Codebase processed successfully! You can now ask questions about the code.');
  } catch (error) {
    console.error('Upload error:', error);
    setStatus(`Error: ${error.message}`, true);
  } finally {
    uploadBtn.disabled = false;
  }
}

async function submitQuery() {
  const query = document.getElementById('query').value.trim();
  
  if (!query) return;
  if (!isProcessed) {
    addMessage('Please upload and process a codebase first.');
    return;
  }

  const queryBtn = document.getElementById('queryBtn');
  const queryInput = document.getElementById('query');
  
  queryBtn.disabled = true;
  queryInput.value = '';
  
  addMessage(query, true);
  addMessage('Searching and generating answer...');

  try {
    const queryEmbedding = await window.openaiClient.createEmbedding(query);
    const results = searchVectorStore(queryEmbedding, 6);
    const context = results.map(r => r.document.content).join('\n\n[SEPARATOR]\n\n');
    
    const messages = [
      {
        role: 'system',
        content: 'You are a helpful code assistant. Answer questions about the codebase based on the provided context. Be concise and accurate. If you cannot find the answer in the context, say so.'
      },
      {
        role: 'user',
        content: `Context from codebase:\n\n${context}\n\nQuestion: ${query}`
      }
    ];
    
    const answer = await window.openaiClient.chat(messages);
    
    const chatWindow = document.getElementById('chatWindow');
    chatWindow.removeChild(chatWindow.lastChild);
    
    addMessage(answer);
    
    const sourcesHtml = `<details class="sources"><summary>View ${results.length} source(s)</summary>${
      results.map((r, i) => `<div class="source"><strong>${r.document.metadata.source}</strong><br/>${r.document.content.substring(0, 200)}...</div>`).join('')
    }</details>`;
    
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'message bot';
    sourcesDiv.innerHTML = sourcesHtml;
    chatWindow.appendChild(sourcesDiv);
    
    chatWindow.scrollTop = chatWindow.scrollHeight;
  } catch (error) {
    console.error('Query error:', error);
    const chatWindow = document.getElementById('chatWindow');
    if (chatWindow.lastChild && chatWindow.lastChild.textContent.includes('Searching')) {
      chatWindow.removeChild(chatWindow.lastChild);
    }
    addMessage(`Error: ${error.message}`);
  } finally {
    queryBtn.disabled = false;
  }
}
