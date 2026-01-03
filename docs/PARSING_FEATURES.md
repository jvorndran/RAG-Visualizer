# Document Parsing Features

## Overview

RAG Visualizer provides flexible document parsing controls that allow you to customize how documents are processed before chunking. These controls are available in the sidebar under the RAG Config tab, in the Document Parsing section.

## PDF Parsing Engines

RAG Visualizer supports three PDF parsing engines, each with different capabilities and trade-offs:

### pypdf (Default)

**Description:** The default PDF text extraction engine, suitable for most documents.

**Characteristics:**
- Free and open-source
- Runs locally (no external API calls)
- Fast processing
- Basic text extraction capabilities
- Manual structure preservation options

**Best for:**
- Simple PDF documents with primarily text content
- Users who want fast, local processing
- Documents without complex layouts or tables

**Limitations:**
- Limited table extraction capabilities
- No automatic structure detection
- May struggle with complex layouts or scanned PDFs

### docling (Advanced)

**Description:** IBM's document understanding library with enhanced structure recognition.

**Characteristics:**
- Free and open-source
- Runs locally (no external API calls)
- Native markdown output
- Automatic structure detection
- Good table extraction

**Best for:**
- Complex PDF documents with tables and structured content
- Users who need better structure preservation without API costs
- Documents with mixed content types (text, tables, headings)

**Advantages over pypdf:**
- Superior table extraction
- Automatic heading detection
- Better preservation of document structure
- Native markdown export

### llamaparse (Best)

**Description:** LlamaIndex's specialized PDF parsing service optimized for RAG applications.

**Characteristics:**
- Paid service (requires API credits)
- Cloud-based processing (requires internet connection)
- Best-in-class parsing quality
- Optimized for RAG use cases
- Excellent table and structure extraction

**Best for:**
- Production RAG applications
- Documents requiring highest quality parsing
- Complex PDFs with tables, images, and multi-column layouts
- Users willing to pay for premium quality

**Requirements:**
- LlamaParse API key (obtain from LlamaCloud)
- Active internet connection
- API credits

**Configuration:**
- Enter your API key in the Advanced Parsing Options expander
- API key is stored locally for convenience

## Output Format Options

The output format determines how parsed text is formatted before chunking. This setting applies to all document types.

### Markdown (Recommended)

**Description:** Converts all documents to markdown format for consistency.

**Benefits:**
- Consistent format across all document types
- Preserves document structure (headings, lists, tables)
- Better chunking results due to structural markers
- Improved semantic search quality
- Human-readable format

**How it works:**
- **PDF:** Uses engine-specific markdown export (docling/llamaparse native, pypdf with manual conversion)
- **DOCX:** Converts heading styles to markdown headings, tables to markdown tables
- **Markdown:** Preserves original markdown
- **Text:** Adds minimal markdown formatting

**Recommended for:**
- Most RAG applications
- When you want consistent document representation
- When structure preservation is important

### Original Format

**Description:** Preserves the document's native text format after parsing.

**Characteristics:**
- No format conversion applied
- Raw text extraction from each document type
- Minimal processing
- Format varies by document type

**Best for:**
- When you want unmodified text
- Documents where formatting is not important
- Plain text documents

### Plain Text

**Description:** Strips all formatting to produce clean plain text.

**How it works:**
- Removes all markdown syntax (headings, bold, italic, tables)
- Removes special characters (configurable)
- Normalizes whitespace (configurable)
- Produces simple, unformatted text

**Best for:**
- When you only need text content without structure
- Compatibility with systems that don't handle markdown
- Minimalist text processing

## Text Processing Options

### Normalize Whitespace

**Default:** Enabled

**Description:** Cleans up excessive whitespace in parsed documents.

**What it does:**
- Replaces multiple consecutive spaces with a single space
- Replaces multiple consecutive newlines with double newline
- Removes leading and trailing whitespace from each line

**Benefits:**
- Cleaner, more consistent text
- Reduces token waste from excessive whitespace
- Improves chunking quality by removing formatting artifacts

**Example:**
```
Before:
"This  is    text.\n\n\n\nMore text."

After:
"This is text.\n\nMore text."
```

**Recommended:** Keep enabled for most use cases.

### Remove Special Characters

**Default:** Disabled

**Description:** Removes special characters while preserving basic punctuation.

**What it does:**
- Removes non-alphanumeric characters
- Preserves basic punctuation: `.` `,` `-` `!` `?` `:` `;`
- Preserves newlines and whitespace

**Benefits:**
- Cleaner text for embedding models
- Removes formatting artifacts and special symbols
- May improve search quality in some cases

**Caution:**
- May remove important symbols or notation
- Can affect technical documents with code or formulas
- May alter meaning in some contexts

**Example:**
```
Before:
"Cost: $50 (50% off!) — Special offer™"

After:
"Cost: 50 50 off - Special offer"
```

**Recommended:** Only enable if you're certain special characters are not needed for your use case.

## Advanced Options

Advanced options are available in the "Advanced Parsing Options" expander and are context-sensitive based on the selected PDF parser.

### LlamaParse API Key (llamaparse only)

**When visible:** Only when "llamaparse (Best)" is selected as the PDF parser

**Description:** Your LlamaParse API key for accessing the cloud parsing service.

**How to obtain:**
1. Visit LlamaCloud (cloud.llamaindex.ai)
2. Create an account or sign in
3. Navigate to API Keys section
4. Generate a new API key
5. Copy and paste into this field

**Security note:** API keys are stored locally in plaintext. Do not share your configuration directory with untrusted parties.

### Preserve Structure (pypdf only)

**When visible:** Only when "pypdf (Default)" is selected as the PDF parser

**Default:** Enabled

**Description:** Adds separators between pages and attempts to detect headings.

**What it does:**
- Inserts page break markers between pages
- Uses markdown-style separators (`---`) when output format is markdown
- Helps maintain document structure during chunking

**Note:** This is a manual fallback for pypdf. The docling and llamaparse engines handle structure preservation automatically and do not need this option.

### Extract Tables (pypdf only)

**When visible:** Only when "pypdf (Default)" is selected as the PDF parser

**Default:** Enabled

**Description:** Attempts to extract table content from PDFs.

**What it does:**
- Extracts text from table cells
- Formats as markdown tables when output format is markdown

**Limitations:**
- pypdf has limited table detection capabilities
- Complex tables may not be extracted correctly
- For better table extraction, consider using docling or llamaparse

**Note:** docling and llamaparse have superior table extraction and do not need this option.

## Usage Recommendations

### For Best Results:

1. **Start with defaults:**
   - PDF Parser: pypdf (Default)
   - Output Format: Markdown (Recommended)
   - Normalize Whitespace: Enabled
   - Remove Special Characters: Disabled

2. **If pypdf quality is insufficient:**
   - Upgrade to docling (Advanced) for free improved parsing
   - Consider llamaparse (Best) for production applications

3. **For consistent RAG performance:**
   - Always use "Markdown (Recommended)" output format
   - This ensures consistent document representation across all file types

4. **For technical documents:**
   - Keep "Remove Special Characters" disabled
   - Special characters may be important for code, formulas, or technical notation

5. **For simple text documents:**
   - pypdf or Original Format may be sufficient
   - Markdown conversion adds minimal value for plain text

### Common Workflows:

**Academic Papers:**
- Parser: docling (Advanced) or llamaparse (Best)
- Output: Markdown (Recommended)
- Normalize Whitespace: Enabled
- Preserve tables and structure for citations and data

**Business Documents:**
- Parser: docling (Advanced)
- Output: Markdown (Recommended)
- Normalize Whitespace: Enabled
- Good table extraction for financial data

**Simple Text Files:**
- Parser: pypdf (Default)
- Output: Original Format or Markdown
- Normalize Whitespace: Enabled
- Minimal processing needed

**Code Documentation:**
- Parser: pypdf (Default) or docling (Advanced)
- Output: Markdown (Recommended)
- Remove Special Characters: Disabled
- Preserve code syntax and special characters

## State Management

**Important:** Changing any parsing settings will invalidate previously generated chunks and embeddings. When you modify parsing parameters:

1. All existing chunks are deleted
2. Embeddings are cleared
3. Search results are reset
4. You'll need to regenerate chunks with the new parsing settings

This ensures your RAG pipeline always uses consistent parsing configuration throughout the entire workflow.

## Persistence

All parsing settings are automatically saved to disk and persist across:
- Browser refreshes
- App restarts
- Session changes

Your configuration is stored locally in `~/.rag-visualizer/config/` and will be restored when you restart the application.

## Troubleshooting

**"LlamaParse API key is required" error:**
- Enter your API key in Advanced Parsing Options
- Ensure the API key is valid and has remaining credits

**Poor table extraction with pypdf:**
- Switch to docling (Advanced) or llamaparse (Best)
- These engines have superior table extraction capabilities

**Document structure not preserved:**
- Ensure Output Format is set to "Markdown (Recommended)"
- For pypdf, enable "Preserve Structure" in Advanced options
- Consider upgrading to docling or llamaparse for automatic structure detection

**Excessive whitespace in chunks:**
- Enable "Normalize Whitespace" option
- This is enabled by default and recommended for most use cases

**Important symbols removed from text:**
- Disable "Remove Special Characters" option
- This option is aggressive and may remove important content

## Technical Details

### Supported File Types:
- PDF (`.pdf`) - All three engines available
- DOCX (`.docx`) - Native markdown conversion for headings and tables
- Markdown (`.md`, `.markdown`) - Pass-through or conversion based on output format
- Text (`.txt`) - UTF-8 decoding with fallback encodings

### Processing Pipeline:

1. **Document Loading:** File is loaded from storage
2. **Format Detection:** File extension determines parser
3. **Engine-Specific Parsing:** Selected PDF engine processes content
4. **Format Conversion:** Text converted to selected output format
5. **Text Processing:** Normalize whitespace and remove special characters (if enabled)
6. **Chunking:** Processed text is split into chunks using selected strategy
7. **Embedding:** Chunks are embedded using selected model

### Dependencies:

All parsing libraries are bundled as required dependencies:
- `pypdf>=3.17` - Default PDF parser
- `python-docx>=1.1` - DOCX parser
- `markdown>=3.5` - Markdown parser
- `docling>=1.0` - Advanced PDF parser
- `llama-parse>=0.4` - Premium PDF parser

No additional installation required - all engines are available immediately after package installation.
