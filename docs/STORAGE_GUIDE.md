# RAG-Visualizer Storage Directory

This directory (`~/.rag-visualizer/`) contains all persistent data for the RAG-Visualizer application.

## Directory Structure

```
C:\Users\jvorn\.rag-visualizer\
├── chunks/         # Processed text chunks (currently unused in favor of session state)
├── config/         # Application configuration files
├── documents/      # Uploaded raw documents (PDFs, TXTs, etc.)
├── embeddings/     # Cached embeddings (currently unused in favor of session state)
├── indices/        # Vector store indices (currently unused in favor of session state)
└── session/        # Active session state and runtime data
```

## Directory Details

### 📁 `documents/`
**Purpose:** Stores uploaded raw documents

**Contents:**
- Original files uploaded through the Upload step
- Supported formats: PDF, TXT, DOCX, MD, etc.
- Files are sanitized and stored with their original names

**When it's used:**
- Files are saved here when you upload documents in the UI
- Files are loaded from here when you select a document in the sidebar

**Management:**
- You can manually delete files from this directory to remove documents
- Files persist across sessions and app restarts

---

### 📁 `chunks/`
**Purpose:** Originally intended for storing processed text chunks

**Current Status:** Largely unused - chunks are now stored in `session/` instead

**Historical Context:**
- This directory was designed to cache chunked text for performance
- Current implementation stores chunks in session state for better integration

---

### 📁 `embeddings/`
**Purpose:** Originally intended for caching embeddings

**Current Status:** Largely unused - embeddings are now stored in `session/` instead

**Historical Context:**
- Was designed to cache expensive embedding computations
- Current implementation stores embeddings in session state

---

### 📁 `indices/`
**Purpose:** Originally intended for FAISS vector store indices

**Current Status:** Largely unused - vector stores are now saved in `session/` instead

**Historical Context:**
- Designed to persist FAISS indices between sessions
- Current implementation stores indices in the session directory

---

### 📁 `config/`
**Purpose:** Stores application configuration

**Contents:**
- `llm_config.json` - LLM provider settings (API keys, model names, etc.)

**Structure of `llm_config.json`:**
```json
{
  "provider": "OpenAI",
  "model": "gpt-4",
  "api_key": "sk-...",
  "base_url": "",
  "temperature": 0.7,
  "max_tokens": 1024,
  "system_prompt": "..."
}
```

**Security Note:**
- ⚠️ Contains API keys in plain text
- Keep this directory secure and don't share these files

---

### 📁 `session/`
**Purpose:** Stores active session state for persistence across page refreshes

**Contents:**
- `session_state.json` - Main session state metadata
- `reduced_embeddings.npy` - UMAP-reduced 2D embeddings for visualization
- `current_vector_store/` - Active FAISS vector store index

**Structure of `session_state.json`:**
```json
{
  "doc_name": "example.pdf",
  "embedding_model_name": "all-MiniLM-L6-v2",
  "chunking_params": {
    "strategy": "Recursive Character",
    "chunk_size": 500,
    "chunk_overlap": 50
  },
  "chunks": [
    {
      "text": "...",
      "metadata": {},
      "start_index": 0,
      "end_index": 500
    }
  ],
  "embeddings_key": "embeddings_10_all-MiniLM-L6-v2_example.pdf",
  "embeddings_model": "all-MiniLM-L6-v2",
  "has_reduced_embeddings": true,
  "has_vector_store": true
}
```

**Important Notes:**
- ⚠️ **Embedder objects are NOT stored** - They are recreated on-demand from the model name
- This prevents PyTorch meta tensor serialization errors
- Vector stores and embeddings are binary files (numpy arrays and FAISS indices)

**When to Clear:**
- If you encounter errors like "Cannot copy out of meta tensor"
- If the app seems to have stale or corrupted data
- After upgrading to a new version with breaking changes

---

## Clearing Data

### Clear Everything
To completely reset the application:
```bash
# Windows (PowerShell)
Remove-Item -Recurse -Force $env:USERPROFILE\.rag-visualizer

# Windows (Command Prompt)
rmdir /s /q %USERPROFILE%\.rag-visualizer
```

### Clear Just Session State
To fix session-related errors without losing documents:
```bash
# Windows (PowerShell)
Remove-Item -Force $env:USERPROFILE\.rag-visualizer\session\*

# Windows (Command Prompt)
del /q %USERPROFILE%\.rag-visualizer\session\*
```

### Clear Just Config
To reset LLM settings:
```bash
# Windows (PowerShell)
Remove-Item -Force $env:USERPROFILE\.rag-visualizer\config\llm_config.json

# Windows (Command Prompt)
del %USERPROFILE%\.rag-visualizer\config\llm_config.json
```

---

## Troubleshooting

### "Cannot copy out of meta tensor" Error
**Cause:** Old embedder object stored in session state (before fix)

**Solution:**
1. Delete `session/session_state.json`
2. Restart the Streamlit app
3. The migration code will clean up any remaining old objects

### Missing Embeddings After Restart
**Cause:** Session state not properly persisted

**Check:**
- Verify `session/session_state.json` exists
- Check if `session/reduced_embeddings.npy` exists
- Check if `session/current_vector_store/` directory exists

**Solution:** Regenerate embeddings by changing a parameter in the sidebar

### API Key Not Persisting
**Cause:** Config file not being saved

**Check:**
- Verify `config/llm_config.json` exists
- Check file permissions on the config directory

---

## Technical Details

### Storage Location
- **Default Path:** `~/.rag-visualizer/` (cross-platform)
- **Windows:** `C:\Users\<username>\.rag-visualizer\`
- **macOS/Linux:** `/home/<username>/.rag-visualizer/`

### File Formats
- **JSON:** Session state, config, metadata
- **NumPy (.npy):** Reduced embeddings (2D arrays)
- **FAISS Index:** Binary vector store indices
- **Raw Files:** Original uploaded documents (PDF, TXT, etc.)

### Managed By
- **Code Location:** `rag_visualizer/services/storage.py`
- **Session Restoration:** `rag_visualizer/app.py` (lines 34-42)
- **Persistence:** Automatic after generating embeddings

---

## Version History

### Current Version (Post Meta-Tensor Fix)
- Embedder objects are NO longer stored in session state
- They are recreated on-demand from the model name
- Migration code automatically removes old embedder objects
- Reduces serialization errors and improves reliability

### Previous Version (Pre Meta-Tensor Fix)
- Embedder objects were stored in `last_embeddings_result`
- Caused PyTorch meta tensor errors on deserialization
- Required manual session state clearing to fix

---

## For Future Agents

When working with this storage directory:

1. **Never store PyTorch models in session state** - Store only model names and recreate on-demand
2. **Session state is loaded once on app startup** - See `app.py:34-42`
3. **Embeddings are cached by key** - Key format: `embeddings_{chunk_count}_{model_name}_{doc_name}`
4. **Config contains API keys** - Be careful not to log or expose them
5. **Vector stores are binary FAISS indices** - Can only be loaded by FAISS library
6. **Migration code exists** - In `app.py:44-49` and `embeddings.py:145-147` to clean up old embedder objects

---

*Last Updated: 2025-12-28*
*RAG-Visualizer Storage Documentation*
