# Toolchain for Semblance AI

### Full Toolchain (from Data Curation to Benchmarking)

#### 1. Data Collection & Curation

| Tool           | Description                                                                 | URL                                                                 |
|----------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------|
| Scrapy         | Web scraping framework for user-approved online content (e.g., personal blogs) | [https://scrapy.org/](https://scrapy.org/)                          |
| BeautifulSoup  | Lightweight library for parsing HTML/XML for specific data extraction       | [https://www.crummy.com/software/BeautifulSoup/](https://www.crummy.com/software/BeautifulSoup/) |
| yt-dlp         | Tool for downloading video/audio from personal media (e.g., YouTube playlists) | [https://github.com/yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp) |
| ffmpeg         | Multimedia processing for extracting video/audio data                       | [https://ffmpeg.org/](https://ffmpeg.org/)                          |
| pdfminer.six   | PDF parsing library for extracting text from documents (e.g., personal papers) | [https://github.com/pdfminer/pdfminer.six](https://github.com/pdfminer/pdfminer.six) |
| PyMuPDF        | High-performance PDF and document parsing library                           | [https://pymupdf.readthedocs.io/](https://pymupdf.readthedocs.io/)  |
| unstructured   | Library for parsing complex documents (e.g., mixed text/image files)        | [https://github.com/Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured) |
| pypandoc       | Document format conversion (e.g., Markdown to text)                         | [https://github.com/JessicaTegner/pypandoc](https://github.com/JessicaTegner/pypandoc) |
| ebooklib       | Library for parsing ePub files from personal book collections               | [https://github.com/aerkalov/ebooklib](https://github.com/aerkalov/ebooklib) |
| music21        | Toolkit for analyzing music scores or metadata for musical preferences       | [https://web.mit.edu/music21/](https://web.mit.edu/music21/)        |
| Label Studio   | Annotation platform for tagging personal data (e.g., ethical stances)       | [https://labelstudio.ai/](https://labelstudio.ai/)                  |
| Streamlit      | Framework for building user-friendly interfaces for manual data upload/tagging | [https://streamlit.io/](https://streamlit.io/)                      |
| pycryptodome   | Library for encrypting personal data during collection                     | [https://www.pycryptodome.org/](https://www.pycryptodome.org/)      |
| Apache Airflow | Workflow orchestration for complex data collection pipelines                | [https://airflow.apache.org/](https://airflow.apache.org/)          |
| Prefect        | Python-native orchestration for smaller data workflows                      | [https://www.prefect.io/](https://www.prefect.io/)                  |

#### 2. Data Storage & Management

| Tool          | Description                                                                 | URL                                                                 |
|---------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------|
| SQLite        | Lightweight database for structured personal data                           | [https://www.sqlite.org/](https://www.sqlite.org/)                  |
| DuckDB        | Embedded database for analytical querying of personal datasets              | [https://duckdb.org/](https://duckdb.org/)                          |
| MinIO         | Self-hosted object storage for multimedia data                             | [https://min.io/](https://min.io/)                                  |
| AWS S3        | Scalable cloud object storage for large datasets                           | [https://aws.amazon.com/s3/](https://aws.amazon.com/s3/)            |
| Cryptomator   | Encrypted local file vaults for sensitive data                             | [https://cryptomator.org/](https://cryptomator.org/)                |
| Weaviate      | Vector database for semantic search of personal data                       | [https://weaviate.io/](https://weaviate.io/)                        |
| Chroma        | Lightweight vector storage for embeddings                                  | [https://www.trychroma.com/](https://www.trychroma.com/)            |
| DVC           | Data versioning tool for reproducibility                                   | [https://dvc.org/](https://dvc.org/)                                |
| Git LFS       | Versioning for large media files (e.g., videos)                            | [https://git-lfs.github.com/](https://git-lfs.github.com/)          |

#### 3. Data Preprocessing & Feature Engineering

| Tool                  | Description                                                                 | URL                                                                 |
|-----------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------|
| Pandas                | Data wrangling for text and tabular data                                   | [https://pandas.pydata.org/](https://pandas.pydata.org/)            |
| Polars                | High-performance data wrangling library                                    | [https://www.pola.rs/](https://www.pola.rs/)                        |
| NumPy                 | Numerical operations for feature extraction                                | [https://numpy.org/](https://numpy.org/)                            |
| scikit-learn          | Classic feature engineering (e.g., TF-IDF)                                 | [https://scikit-learn.org/](https://scikit-learn.org/)              |
| spaCy                 | NLP library for processing personal writings and texts                     | [https://spacy.io/](https://spacy.io/)                              |
| Hugging Face Transformers | Advanced NLP and embeddings for text processing                         | [https://huggingface.co/](https://huggingface.co/)                  |
| sentence-transformers | Generating embeddings for knowledge-augmented generation                   | [https://www.sbert.net/](https://www.sbert.net/)                    |
| librosa               | Audio preprocessing for music or voice data                                | [https://librosa.org/](https://librosa.org/)                        |
| torchaudio            | Audio feature extraction for speech or music                               | [https://pytorch.org/audio/](https://pytorch.org/audio/)            |
| OpenCV                | Vision preprocessing for personal images or art                            | [https://opencv.org/](https://opencv.org/)                          |
| PIL                   | Image manipulation for preprocessing                                       | [https://pillow.readthedocs.io/](https://pillow.readthedocs.io/)    |
| Featuretools          | Automated feature extraction from metadata                                 | [https://www.featuretools.com/](https://www.featuretools.com/)      |

#### 4. Model Development

| Tool            | Description                                                                 | URL                                                                 |
|-----------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------|
| PyTorch         | Core framework for LLM development                                         | [https://pytorch.org/](https://pytorch.org/)                        |
| Hugging Face Transformers | Prebuilt architectures for fine-expanded model tuning                     | [https://huggingface.co/](https://huggingface.co/)                  |
| peft            | Efficient fine-tuning with LoRA/adapters                                   | [https://github.com/huggingface/peft](https://github.com/huggingface/peft) |
| OpenLLM         | Specialized tools for personalized LLMs                                    | [https://github.com/bentoml/OpenLLM](https://github.com/bentoml/OpenLLM) |
| LLaMA.cpp       | Efficient local inference for small models                                 | [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) |
| llama-cpp-python | Python bindings for LLaMA.cpp                                            | [https://github.com/abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python) |
| NanoGPT         | Lightweight LLM prototyping framework                                      | [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) |
| trl             | Reinforcement learning for value alignment                                 | [https://github.com/huggingface/trl](https://github.com/huggingface/trl) |
| FastAI          | High-level API for quick model prototyping                                 | [https://www.fast.ai/](https://www.fast.ai/)                        |

#### 5. Model Training Infrastructure

| Tool                  | Description                                                                 | URL                                                                 |
|-----------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------|
| Weights & Biases      | Experiment tracking for training runs                                      | [https://wandb.ai/](https://wandb.ai/)                              |
| MLflow                | Lightweight experiment tracking                                            | [https://mlflow.org/](https://mlflow.org/)                          |
| HuggingFace Accelerate | Distributed training on local GPUs                                        | [https://huggingface.co/docs/accelerate](https://huggingface.co/docs/accelerate) |
| DeepSpeed             | Optimizing large-scale training                                            | [https://www.deepspeed.ai/](https://www.deepspeed.ai/)              |
| Docker                | Containerization for reproducible environments                             | [https://www.docker.com/](https://www.docker.com/)                  |
| Podman                | Lightweight containerization alternative                                   | [https://podman.io/](https://podman.io/)                            |
| optimum               | NVIDIA GPU optimization for local training                                 | [https://huggingface.co/docs/optimum](https://huggingface.co/docs/optimum) |
| Colab Pro             | Cloud-based prototyping for small models                                   | [https://colab.google/](https://colab.google/)                      |
| RunPod                | GPU rental for large-scale training                                        | [https://www.runpod.io/](https://www.runpod.io/)                    |

#### 6. Evaluation & Benchmarking

| Tool              | Description                                                                 | URL                                                                 |
|-------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------|
| lm-eval-harness   | LLM benchmarking for reasoning and coherence                               | [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) |
| MT-Bench          | Conversational evaluation for LLMs                                         | [https://github.com/lmsys/lm-sys](https://github.com/lmsys/lm-sys)  |
| ROUGE             | Text similarity metrics for style alignment                                | [https://github.com/pltrdy/rouge](https://github.com/pltrdy/rouge)  |
| BERTScore         | Semantic similarity for personal data                                      | [https://github.com/Tiiiger/bert_score](https://github.com/Tiiiger/bert_score) |
| scikit-learn      | General metric calculations                                               | [https://scikit-learn.org/](https://scikit-learn.org/)              |
| Great Expectations | Data quality testing                                                     | [https://greatexpectations.io/](https://greatexpectations.io/)      |
| Pandera           | Schema validation for datasets                                            | [https://pandera.readthedocs.io/](https://pandera.readthedocs.io/)  |
| pytest            | Custom tests for alignment with beliefs                                    | [https://pytest.org/](https://pytest.org/)                          |
| SHAP              | Model interpretability for data influence                                  | [https://shap.readthedocs.io/](https://shap.readthedocs.io/)        |

#### 7. Deployment & Serving

| Tool         | Description                                                                 | URL                                                                 |
|--------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------|
| FastAPI      | Local API for model serving                                                | [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)      |
| Gradio       | Interactive UI for model interaction                                       | [https://gradio.app/](https://gradio.app/)                          |
| Panel        | Dashboard for data/model visualization                                     | [https://panel.holoviz.org/](https://panel.holoviz.org/)            |
| vLLM         | Efficient LLM inference serving                                           | [https://vllm.ai/](https://vllm.ai/)                                |
| TGI          | Text generation inference for LLMs                                        | [https://github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) |
| ONNX         | Optimized model format for deployment                                      | [https://onnx.ai/](https://onnx.ai/)                                |
| Nginx        | Reverse proxy for secure serving                                           | [https://nginx.org/](https://nginx.org/)                            |
| Modal        | Serverless deployment for scalability                                      | [https://modal.com/](https://modal.com/)                            |
| Keycloak     | User authentication for shared access                                      | [https://www.keycloak.org/](https://www.keycloak.org/)              |

#### 8. Retrieval-Augmented Generation (RAG) Specific

| Tool       | Description                                                                 | URL                                                                 |
|------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------|
| LlamaIndex | RAG orchestration for personal data                                        | [https://www.llamaindex.ai/](https://www.llamaindex.ai/)            |
| LangChain  | Flexible RAG pipelines                                                    | [https://www.langchain.com/](https://www.langchain.com/)            |
| Chroma     | Lightweight vector storage for retrieval                                   | [https://www.trychroma.com/](https://www.trychroma.com/)            |
| FAISS      | Local vector retrieval for embeddings                                     | [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss) |
| RAGAS      | Evaluation of RAG performance                                             | [https://github.com/explosion/ragas](https://github.com/explosion/ragas) |

---

### Notes on Deployment Strategies
- **Local**: Tools designed for personal hardware (e.g., laptop, GPU workstation) to prioritize privacy and cost-efficiency. Ideal for initial prototyping and personal use with your 4GB+ dataset.
- **Cloud**: Scalable, resource-intensive tools for large datasets or training larger models (e.g., 1B parameters). Useful for future scaling but may involve costs and privacy considerations.
- **Hybrid**: Tools that can run locally or in the cloud, offering flexibility. Suitable for transitioning from local experiments to cloud-based scaling or for tools with optional cloud integrations (e.g., Hugging Face).