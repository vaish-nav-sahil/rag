# DOCRAG — Multimodal Retrieval-Augmented Generation

A learning-stage Multimodal RAG project that explores retrieving information from PDF documents containing both text and images using CLIP, FAISS, and GPT-4.1.

## Overview

Traditional Retrieval-Augmented Generation (RAG) systems primarily focus on text. However, PDF documents often contain charts, diagrams, and other visual information that can be important for answering questions.

**DOCRAG** explores a multimodal approach by extracting text and embedded images from PDFs, generating embeddings for both modalities, retrieving relevant content, and passing the retrieved evidence to a multimodal language model to generate answers.

This repository currently contains a Jupyter Notebook implementing the initial prototype.

## Features

- **PDF text extraction** using PyMuPDF.
- **Text chunking** using LangChain's `RecursiveCharacterTextSplitter`.
- **Image extraction** and conversion to PIL format.
- **Multimodal embeddings** using CLIP for text and images.
- **Vector similarity search** using FAISS.
- **Base64 image storage** for passing retrieved images to the vision-capable LLM.
- **Question answering** using GPT-4.1 through LangChain.
- **Metadata tracking** for document pages and content types.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| PyMuPDF | PDF processing |
| Hugging Face Transformers | Loading CLIP |
| CLIP (`openai/clip-vit-base-patch32`) | Text and image embeddings |
| LangChain | Document processing and LLM integration |
| FAISS | Vector similarity search |
| GPT-4.1 | Multimodal question answering |
| NumPy | Embedding array operations |
| Pillow | Image processing |

## Architecture

```text
              PDF Document
                   |
          +--------+--------+
          |                 |
     Extract Text      Extract Images
          |                 |
     Split into         Convert to
       Chunks            PIL Images
          |                 |
      CLIP Text         CLIP Image
      Embeddings        Embeddings
          |                 |
          +--------+--------+
                   |
          Unified Embedding Space
                   |
              FAISS Index
                   |
             User Question
                   |
          CLIP Query Embedding
                   |
          Retrieve Relevant Text
              and Images
                   |
          Resolve Image References
             using Base64 Store
                   |
                GPT-4.1
                   |
              Generated Answer
```

## How It Works

### 1. PDF Ingestion

The notebook uses PyMuPDF to extract text and embedded images from a PDF document.

### 2. Text Processing

Extracted text is divided into smaller chunks using `RecursiveCharacterTextSplitter`, with a chunk size of 500 characters and an overlap of 100 characters.

### 3. Embedding Generation

The pretrained CLIP model generates 512-dimensional embeddings for text chunks and images. The embeddings are normalized before being stored.

### 4. Vector Storage and Retrieval

FAISS indexes the precomputed embeddings. A user's question is embedded using CLIP, and similarity search retrieves relevant text chunks and image references.

### 5. Multimodal Answer Generation

Retrieved text and associated images are passed to GPT-4.1, allowing it to generate answers using the available textual and visual evidence.

## Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version.
- Jupyter Notebook or JupyterLab.
- An OpenAI API key with access to GPT-4.1.
- Internet access for downloading the pretrained CLIP model and accessing the OpenAI API.

### Installation

Install the required packages:

```bash
pip install pymupdf numpy torch pillow scikit-learn transformers langchain langchain-core langchain-community langchain-openai langchain-text-splitters faiss-cpu python-dotenv
```

### Configure Environment Variables

Create a `.env` file in your project directory:

```env
OPENAI_API_KEY=your_openai_api_key
```

Never commit your actual `.env` file or API key to GitHub.

### Prepare the PDF

Place your PDF in the expected location or update the `pdf_path` variable in the notebook:

```python
pdf_path = "multimodal_sample.pdf"
```

The notebook currently uses this sample filename. You can modify it to point to another PDF.

### Run the Notebook

1. Clone or download this repository.
2. Install the dependencies.
3. Configure your API key.
4. Open the `.ipynb` notebook in Jupyter.
5. Run the cells in order.
6. Ask questions about the processed PDF and inspect the retrieved results.

## Current Status

**Status: Initial working prototype**

The current implementation demonstrates:

- PDF text and embedded-image extraction.
- CLIP embedding generation for both modalities.
- Unified FAISS similarity search.
- Retrieval of text chunks and image references.
- Integration with GPT-4.1 for multimodal question answering.

## Limitations

This project is actively being learned and improved. Current limitations include:

- PDF image extraction may miss charts or diagrams represented as vector graphics.
- Retrieval quality has not yet been evaluated using a formal benchmark.
- CLIP embeddings may not be optimal for every type of text-retrieval task.
- Retrieved image references must be resolved to their stored image data before visual analysis.
- The system may generate unsupported answers if the retrieved evidence is insufficient.
- Production features such as persistent indexing, comprehensive error handling, automated evaluation, and deployment are not yet implemented.

## Future Improvements

- [ ] Render PDF pages to capture vector-based charts and diagrams.
- [ ] Improve retrieval quality and experiment with reranking.
- [ ] Build an evaluation dataset for text and image retrieval.
- [ ] Add tests for unsupported questions and hallucination resistance.
- [ ] Implement persistent FAISS indexes.
- [ ] Improve document and image metadata management.
- [ ] Build a user-friendly interface.
- [ ] Add robust error handling and deployment support.

## Learning Objectives

This project is helping me understand:

- Multimodal embeddings and cross-modal retrieval.
- Retrieval-Augmented Generation architecture.
- Vector databases and similarity search.
- PDF parsing and image processing.
- Connecting retrieved evidence to a vision-capable LLM.

## Disclaimer

DOCRAG is an educational project and is not intended to be production-ready. Retrieval accuracy and answer reliability require further testing and evaluation.

## License

No license has been specified yet. Add an appropriate open-source license if you intend to permit others to reuse, modify, or distribute this project.
