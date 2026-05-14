# Multimodal Video-RAG System

## 📌 Project Introduction
This project implements an advanced data processing pipeline designed to convert videos from the **MMBench-Video dataset** into a rich, searchable plain-text format. 

Currently, **283 out of 608 videos** have been successfully processed and converted.

This pre-processed textual data serves as the foundation for a state-of-the-art **Multimodal Video-RAG (Retrieval-Augmented Generation)** system. By transforming dense video pixels and audio waves into semantic text, the system allows Large Language Models to accurately search, reason over, and answer questions about video content without requiring expensive native multimodal inference at runtime.

---

## 🧠 RAG Technique & Code Workflow Analysis
The project relies on a two-phase architecture: an **Offline Video Encoding Pipeline** and a **Runtime RAG API**. 

### 1. Offline Video Encoding Pipeline (`video_encoder.py`)
This pipeline converts raw `.mp4` videos into aligned semantic text documents.
* **Video Parsing & Modality Extraction:** The pipeline uses `PySceneDetect` to semantically chunk the video into distinct scenes based on visual content changes. For each scene, the audio track and video frames are isolated.
* **Audio Transcription:** The isolated audio for each scene is processed using **OpenAI's Whisper** model (`small` variant) to generate highly accurate text transcriptions.
* **Keyframe Sampling Strategies:** To avoid processing redundant frames, the system calculates the Mean Squared Error (MSE) between frames within a scene. It dynamically selects the top $K$ most visually distinct frames (keyframes) to comprehensively represent the scene's visual flow.
* **Visual-to-Text Conversion (Vision AI):** A dual-model approach is used for visual understanding. First, **BLIP** (`base_coco`) generates a foundational caption for each keyframe. Next, a **LLaVA** model (running serverlessly via Modal) iteratively integrates the new frame's caption with the preceding context to synthesize a seamless, logical narrative of the entire scene.
* **Alignment & Chunking:** The transcribed audio text and the synthesized visual narrative are aligned temporally by scene. This combined data is structured into "Video Documents" containing start/end times and saved to a standardized `result_document.json`.

### 2. Runtime RAG Agent (`chat.py`)
* **Vector Embedding & Database:** Upon API startup, the JSON knowledge base is loaded. Each segment's combined audio/visual text is embedded using **HuggingFace Sentence Transformers** (`all-MiniLM-L6-v2`). The vectors are stored in an in-memory **Qdrant Vector Database** via LangChain.
* **Retrieval & Generation:** When a user queries the system, the question is embedded and queried against Qdrant to retrieve the top-3 most relevant video segments. These segments (containing both visual and audio descriptions) are injected into the context window of a local **Mistral** model (via Ollama) to generate an accurate, grounded answer.

---

## ⚙️ Environment Configuration

The application is split into two distinct dependency environments to optimize deployment size:

### 1. Offline Machine Learning Pipeline (Video Encoding)
* **Hardware:** GPU/CUDA is **highly recommended** (NVIDIA GPU).
* **Core Libraries:** `torch`, `torchvision`, `openai-whisper`, `salesforce-lavis` (BLIP), `opencv-python`, `scenedetect`, `modal` (serverless compute).
* **Dependencies File:** `requirements.txt`

### 2. RAG API Runtime (Dockerized)
* **Hardware:** CPU is sufficient (GPU optional for faster embeddings/LLM generation).
* **Python Version:** 3.11.9
* **Core Libraries:** `fastapi`, `uvicorn`, `langchain`, `qdrant-client`, `ollama`, `sentence-transformers`, `sqlalchemy`.
* **Dependencies File:** `requirements-api.txt`

---

## 🚀 Installation & Docker Usage Guide

### Local Development Setup
1. Clone the repository and navigate to the project root.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the API dependencies:
   ```bash
   pip install -r requirements-api.txt
   ```

### Docker Deployment (Recommended)
The production environment uses a multi-stage Docker build to keep the API lightweight. It utilizes host networking to seamlessly connect to a native Ollama instance running on the host machine.

**1. Configure Environment Variables**
Create or modify the `.env` file in the root directory:
```env
# Connect to native Ollama on the host
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=all-MiniLM-L6-v2
DOCUMENT_PATH=result_document.json
```

**2. Build and Run the Container**
Ensure you have Docker and Docker Compose installed. Execute the following commands:
```bash
# Build the Docker image
docker compose build rag-api

# Start the service in detached mode
docker compose up -d
```

**3. Volume Mounting Explained**
In the `docker-compose.yaml`, the data is mounted automatically:
* **Knowledge Base:** `- ./result_document.json:/app/data/result_document.json:ro`
  *(Mounts the processed MMBench-Video data as read-only).*
* **HuggingFace Cache:** `- hf_cache:/app/.hf_cache`
  *(Prevents the embedding model from re-downloading on every container restart).*
* **Database:** `- chatbot_db:/app/db`
  *(Persists the SQLite conversation history).*

You can monitor the startup process (which includes indexing the vector database) using:
```bash
docker logs -f video-rag-api
```
The API will be accessible at `http://localhost:8000`.

---

## 🦙 Local LLM Setup (Ollama & Mistral)

The generation phase of this Video-RAG system relies on a local Large Language Model (LLM) to synthesize the final answers based on the retrieved video segments. We use **Ollama** to run the **Mistral** model locally, ensuring privacy and reducing latency.

**1. Install Ollama**
Follow the official instructions to install Ollama for your OS:
[https://ollama.com/download](https://ollama.com/download)

**2. Pull and Run the Mistral Model**
Once Ollama is installed, run the following command in your terminal. This will automatically download the Mistral model and start the Ollama service:
```bash
ollama run mistral
```

**3. Service Information**
By default, the Ollama service will run locally and listen on:
* **`http://localhost:11434`**

*(Note: The Docker configuration via `network_mode: host` allows the API container to seamlessly connect to this native Ollama instance).*

---

## 🧪 API Usage & Testing

Once the Docker container is running and the startup process (including vector database indexing) is complete, you can interact with the system via the REST API.

* **Default Base URL:** `http://localhost:8000`
* **Swagger UI Documentation & Testing:** Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) in your browser. This interactive interface allows you to view all available endpoints, check expected request/response schemas, and test queries directly.

**Sample cURL Request:**
To query the RAG system via the terminal:
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "According to the video, at what speed does the Earth spin?"
}'
```

---

## 🧹 Stopping Docker & Resource Cleanup

To safely stop and clean up the environment when you are done:

**1. Stop Running Containers**
```bash
docker compose down
```
*(This stops and removes the containers but preserves your SQLite database and HuggingFace cache volumes).*

**2. Complete Cleanup (Optional)**
If you want to completely free up disk space and remove all associated data (including the database and downloaded embedding models):
```bash
# Remove containers AND named volumes
docker compose down -v

# Remove the built Docker image
docker rmi video-rag-api:1.0.0

# Prune unused Docker networks and dangling images
docker system prune -f
```

---

## 📂 Project Structure Overview

```text
muti_model/
├── Dockerfile                  # Multi-stage build instructions for the runtime API
├── docker-compose.yaml         # Container orchestration and volume management
├── .env                        # Environment variable configurations
├── requirements.txt            # Full dependency list (including heavy ML models)
├── requirements-api.txt        # Lightweight dependencies for the Docker runtime
├── entrypoint.sh               # Startup script handling health checks and Uvicorn launch
├── main.py                     # FastAPI application entrypoint and route registration
├── chat.py                     # Core RAG Agent logic (Qdrant DB, LangChain, Ollama)
├── video_encoder.py            # Offline ML pipeline for scene detection, Whisper, and BLIP
├── result_document.json        # Pre-processed knowledge base (Video-to-Text documents)
├── ollama_modal.py             # Serverless deployment configuration for Ollama via Modal
└── app/                        # Modularized backend application directory
    ├── api/                    # FastAPI route definitions
    ├── core/                   # Application settings and error definitions
    ├── models/                 # SQLAlchemy database schemas
    └── services/               # Business logic, including the AI Service threadpool wrapper
```
