# Feature & State List for `chunks.py`

## Feature & State List

| # | User Action | UI Change | Session State Variables Affected |
|---|-------------|-----------|----------------------------------|
| **Navigation** ||||
| 1 | Clicks "Go to Upload Step" button (when no document selected) | Page navigates to the upload step via `st.rerun()` | `current_step` → set to `"upload"` |
| **Document Parsing** ||||
| 2 | Clicks "Parse Document" button (first parse) | Spinner appears → Document parsed → Chunks visualization renders | `parsed_text_{doc}_{params_hash}` → set to parsed text<br>`applied_parsing_params` → set to copy of `parsing_params`<br>`chunks` → populated with chunk objects |
| 3 | Clicks "Reparse Document" button (after parsing settings change) | Spinner appears → Document re-parsed with new settings → Chunks re-rendered | `parsed_text_{doc}_{params_hash}` → new key created with new parsed text<br>`applied_parsing_params` → updated to current `parsing_params`<br>`chunks` → regenerated |
| **Passive State Reads (no user action - read on render)** ||||
| 4 | Page loads with document selected | Config panel shows document name, strategy, chunk size, overlap | Reads: `doc_name`, `applied_chunking_params` (fallback: `chunking_params`), `applied_parsing_params`, `parsing_params` |
| 5 | Page loads with parsed text available | Chunk visualization renders with stats (total chunks, avg size, overlap counts) | Reads: `parsed_text_{doc}_{params_hash}`<br>Writes: `chunks` → regenerated from text + params |
| 6 | Page loads with parsing settings changed | Info banner appears: "Parsing settings have changed..." | Compares: `parsing_params` vs `applied_parsing_params` |

---

## Detailed State Variable Reference

| Variable | Type | Purpose | Set By | Read By |
|----------|------|---------|--------|---------|
| `doc_name` | `str` | Currently selected document filename | Sidebar / Upload step | `render_chunks_step()` |
| `chunking_params` | `dict` | Current chunking config (provider, splitter, chunk_size, chunk_overlap) | Sidebar | `render_chunks_step()` |
| `applied_chunking_params` | `dict` | Last applied chunking config | Sidebar (on apply) | `render_chunks_step()` |
| `parsing_params` | `dict` | Current parsing config from sidebar | Sidebar | `render_chunks_step()` |
| `applied_parsing_params` | `dict` | Parsing params used for current parsed text | Parse button handler | `render_chunks_step()` |
| `current_step` | `str` | Active step in the wizard | "Go to Upload" button | Other modules |
| `parsed_text_{doc}_{hash}` | `str` | Cached parsed document text (dynamic key) | Parse button handler | `render_chunks_step()` |
| `chunks` | `list[Chunk]` | Generated chunk objects | `render_chunks_step()` | Other modules (e.g., embedding step) |

---

## Key Conditional Flows

| Condition | Result |
|-----------|--------|
| `doc_name` is None/empty | Shows info message + "Go to Upload Step" button; early return |
| `source_text` is empty AND no persistent cache | Shows "Parse Document" button; early return after button |
| `parsing_params != applied_parsing_params` | Shows "Reparse Document" button + warning info; continues to show existing chunks |
| `source_text` exists | Generates chunks via `get_chunks()`, renders stats & visualization |

---

## External Dependencies

| Import | Usage |
|--------|-------|
| `get_chunks` | Generates chunk objects from parsed text using configured splitter |
| `get_storage_dir`, `load_document` | File system operations for document loading |
| `parse_document` | Parses raw document content into text |
| `get_provider_splitters` | Gets splitter display names for UI |

---

## Persistent Storage

The module persists parsed text to disk at `{storage_dir}/parsed_texts/`:
- Filename format: `{sanitized_doc_name}_{md5_hash}.txt`
- Hash is computed from `parsed_text_{doc_name}_{json(parsing_params)}`
- Loaded on page render if not in session state (lines 153-158)
- Saved after successful parse (line 194)
