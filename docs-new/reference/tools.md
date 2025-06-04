# Tools Reference

This page provides a comprehensive overview of the tools used in the Semblance AI project, from data curation to deployment and benchmarking.

## Full Toolchain Overview

### 1. Data Collection & Curation

| Tool | Description | URL |
|------|-------------|-----|
| Scrapy | Web scraping framework for user-approved online content | [scrapy.org](https://scrapy.org/) |
| BeautifulSoup | Lightweight library for parsing HTML/XML | [crummy.com/software/BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) |
| yt-dlp | Tool for downloading video/audio from personal media | [github.com/yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp) |
| ffmpeg | Multimedia processing for extracting video/audio data | [ffmpeg.org](https://ffmpeg.org/) |
| pdfminer.six | PDF parsing library for extracting text | [github.com/pdfminer/pdfminer.six](https://github.com/pdfminer/pdfminer.six) |
| PyMuPDF | High-performance PDF and document parsing | [pymupdf.readthedocs.io](https://pymupdf.readthedocs.io/) |
| unstructured | Library for parsing complex documents | [github.com/Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured) |
| pypandoc | Document format conversion | [github.com/JessicaTegner/pypandoc](https://github.com/JessicaTegner/pypandoc) |
| ebooklib | Library for parsing ePub files | [github.com/aerkalov/ebooklib](https://github.com/aerkalov/ebooklib) |
| music21 | Toolkit for analyzing music scores | [web.mit.edu/music21](https://web.mit.edu/music21/) |
| Label Studio | Annotation platform for tagging personal data | [labelstudio.ai](https://labelstudio.ai/) |
| Streamlit | Framework for user-friendly interfaces | [streamlit.io](https://streamlit.io/) |
| pycryptodome | Library for encrypting personal data | [pycryptodome.org](https://www.pycryptodome.org/) |
| Apache Airflow | Workflow orchestration | [airflow.apache.org](https://airflow.apache.org/) |
| Prefect | Python-native orchestration | [prefect.io](https://www.prefect.io/) |

### 2. Data Storage & Management

| Tool | Description | URL |
|------|-------------|-----|
| SQLite | Lightweight database for structured data | [sqlite.org](https://www.sqlite.org/) |
| DuckDB | Embedded database for analytical querying | [duckdb.org](https://duckdb.org/) |
| MinIO | Self-hosted object storage | [min.io](https://min.io/) |
| AWS S3 | Scalable cloud object storage | [aws.amazon.com/s3](https://aws.amazon.com/s3/) |
| Cryptomator | Encrypted local file vaults | [cryptomator.org](https://cryptomator.org/) |
| Weaviate | Vector database for semantic search | [weaviate.io](https://weaviate.io/) |
| Chroma | Lightweight vector storage | [trychroma.com](https://www.trychroma.com/) |
| DVC | Data versioning tool | [dvc.org](https://dvc.org/) |
| Git LFS | Versioning for large media files | [git-lfs.github.com](https://git-lfs.github.com/) |

### 3. Data Preprocessing & Feature Engineering

| Tool | Description | URL |
|------|-------------|-----|
| Pandas | Data wrangling for text and tabular data | [pandas.pydata.org](https://pandas.pydata.org/) |
| Polars | High-performance data wrangling | [pola.rs](https://www.pola.rs/) |
| NumPy | Numerical operations | [numpy.org](https://numpy.org/) |
| scikit-learn | Classic feature engineering | [scikit-learn.org](https://scikit-learn.org/) |
| spaCy | NLP library for processing texts | [spacy.io](https://spacy.io/) |
| Transformers | Advanced NLP and embeddings | [huggingface.co](https://huggingface.co/) |
| sentence-transformers | Generating embeddings | [sbert.net](https://www.sbert.net/) |
| librosa | Audio preprocessing | [librosa.org](https://librosa.org/) |
| torchaudio | Audio feature extraction | [pytorch.org/audio](https://pytorch.org/audio/) |
| OpenCV | Vision preprocessing | [opencv.org](https://opencv.org/) |
| PIL | Image manipulation | [pillow.readthedocs.io](https://pillow.readthedocs.io/) |
| Featuretools | Automated feature extraction | [featuretools.com](https://www.featuretools.com/) |

### 4. Model Development

| Tool | Description | URL |
|------|-------------|-----|
| PyTorch | Core framework for LLM development | [pytorch.org](https://pytorch.org/) |
| Transformers | Prebuilt architectures | [huggingface.co](https://huggingface.co/) |
| peft | Efficient fine-tuning | [github.com/huggingface/peft](https://github.com/huggingface/peft) |
| OpenLLM | Specialized tools for personalized LLMs | [github.com/bentoml/OpenLLM](https://github.com/bentoml/OpenLLM) |
| LLaMA.cpp | Efficient local inference | [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) |
| llama-cpp-python | Python bindings for LLaMA.cpp | [github.com/abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python) |
| NanoGPT | Lightweight LLM prototyping | [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) |
| trl | Reinforcement learning for alignment | [github.com/huggingface/trl](https://github.com/huggingface/trl) |
| FastAI | High-level API for prototyping | [fast.ai](https://www.fast.ai/) |

### 5. Model Training Infrastructure

| Tool | Description | URL |
|------|-------------|-----|
| Weights & Biases | Experiment tracking | [wandb.ai](https://wandb.ai/) |
| MLflow | Lightweight experiment tracking | [mlflow.org](https://mlflow.org/) |
| Accelerate | Distributed training on local GPUs | [huggingface.co/docs/accelerate](https://huggingface.co/docs/accelerate) |
| DeepSpeed | Optimizing large-scale training | [deepspeed.ai](https://www.deepspeed.ai/) |
| Docker | Containerization | [docker.com](https://www.docker.com/) |
| Podman | Lightweight containerization | [podman.io](https://podman.io/) |
| optimum | NVIDIA GPU optimization | [huggingface.co/docs/optimum](https://huggingface.co/docs/optimum) |
| Colab Pro | Cloud-based prototyping | [colab.google](https://colab.google/) |
| RunPod | GPU rental for training | [runpod.io](https://www.runpod.io/) |

### 6. Evaluation & Benchmarking

| Tool | Description | URL |
|------|-------------|-----|
| lm-eval-harness | LLM benchmarking | [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) |
| MT-Bench | Conversational evaluation | [github.com/lmsys/lm-sys](https://github.com/lmsys/lm-sys) |
| ROUGE | Text similarity metrics | [github.com/pltrdy/rouge](https://github.com/pltrdy/rouge) |
| BERTScore | Semantic similarity | [github.com/Tiiiger/bert_score](https://github.com/Tiiiger/bert_score) |
| scikit-learn | General metric calculations | [scikit-learn.org](https://scikit-learn.org/) |
| Great Expectations | Data quality testing | [greatexpectations.io](https://greatexpectations.io/) |
| Pandera | Schema validation | [pandera.readthedocs.io](https://pandera.readthedocs.io/) |
| pytest | Custom tests for alignment | [pytest.org](https://pytest.org/) |
| SHAP | Model interpretability | [shap.readthedocs.io](https://shap.readthedocs.io/) |

### 7. Deployment & Serving

| Tool | Description | URL |
|------|-------------|-----|
| FastAPI | Local API for model serving | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) |
| Gradio | Interactive UI for model interaction | [gradio.app](https://gradio.app/) |
| Panel | Dashboard for visualization | [panel.holoviz.org](https://panel.holoviz.org/) |
| vLLM | Efficient LLM inference serving | [vllm.ai](https://vllm.ai/) |
| TGI | Text generation inference | [github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) |
| ONNX | Optimized model format | [onnx.ai](https://onnx.ai/) |
| Nginx | Reverse proxy for secure serving | [nginx.org](https://nginx.org/) |
| Modal | Serverless deployment | [modal.com](https://modal.com/) |
| Keycloak | User authentication | [keycloak.org](https://www.keycloak.org/) |

### 8. RAG-Specific Tools

| Tool | Description | URL |
|------|-------------|-----|
| LlamaIndex | RAG orchestration | [llamaindex.ai](https://www.llamaindex.ai/) |
| LangChain | Flexible RAG pipelines | [langchain.com](https://www.langchain.com/) |
| Chroma | Lightweight vector storage | [trychroma.com](https://www.trychroma.com/) |
| FAISS | Local vector retrieval | [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss) |
| RAGAS | RAG performance evaluation | [github.com/explosion/ragas](https://github.com/explosion/ragas) |

## Deployment Strategies

### Local Deployment
- Designed for personal hardware (laptop, GPU workstation)
- Prioritizes privacy and cost-efficiency
- Ideal for initial prototyping and personal use with 4GB+ dataset
- Examples: LLaMA.cpp, Ollama, local vector stores

### Cloud Deployment
- Scalable, resource-intensive tools
- Suitable for large datasets or training larger models (1B+ parameters)
- Involves costs and privacy considerations
- Examples: AWS SageMaker, Azure ML, Google Vertex AI

### Hybrid Deployment
- Tools that can run both locally and in the cloud
- Offers flexibility for transitioning between environments
- Suitable for gradual scaling
- Examples: Hugging Face tools, MLflow, Docker containers 