# exaPipeline

**exaPipeline** is a production-ready, containerized system designed to transform domain-specific PDF documents into structured datasets for fine-tuning large language models (LLMs), currently optimized for **Qwen3:4B**.  

It automates the entire pipeline from document ingestion to training data generation, ensuring high-quality outputs with minimal manual intervention.

---

## Features

- **Microservices Architecture**: Scalable, fault-tolerant design with separate components for each processing stage.
- **AI-Powered Processing**: Leverages OCR and LLM technologies for accurate document analysis.
- **Real-Time Monitoring**: Track progress and system health through intuitive dashboards.
- **Full Automation**: Transform PDFs into JSONL training data automatically.
- **Quality Assurance**: Validation and hallucination detection ensure high-quality data.
- **Scalable Architecture**: Supports large-scale processing with containerized microservices.
- **User-Friendly Dashboard**: No-code Streamlit dashboard for non-technical users.
- **Multiple Export Formats**: Supports SFT, RLAIF, RLHF, and Qwen3 chat formats.

---

## Deployment

### Prerequisites

- Docker & Docker Compose  
- Git  
- **exaOCR server** (PDF → Markdown)  
- **vLLM server** running Qwen3:4B  

### Environment Configuration

Create `.env` files for your services:

#### exaPipeline/.env
```bash
REDIS_URL=redis://<redis-host-ip>:6379/0
OCR_API_URL=http://<exaocr-ip>:45001
VLLM_API_URL=http://<vllm-ip>:45000
```

#### exaPipelineDashboard/.env
```bash
BACKEND_API_URL=http://<backend-api-ip>:8000
```

> ⚠️ **Important:** Never hard-code IPs. Always use `.env` files for configuration.

---

### Deployment Steps

1. **Clone the Repositories**
```bash
git clone https://github.com/ikantkode/exaOCR.git
git clone https://github.com/ikantkode/exaPipeline.git
git clone https://github.com/ikantkode/exaPipelineDashboard.git
```

2. **Configure Environment Files**  
   Edit `.env` files with your actual IPs or `localhost` for single-machine setups.

3. **Deploy exaOCR**
```bash
cd exaOCR
docker compose up -d
```

4. **Deploy exaPipeline**
```bash
cd exaPipeline
docker compose up -d
```

5. **Deploy Dashboard**
```bash
cd exaPipelineDashboard
docker compose up -d
```

6. **Verify Services**
- Dashboard: [http://localhost:8501](http://localhost:8501)  
- API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)  
- Monitoring: [http://localhost:5555](http://localhost:5555)

---

## Usage

1. Upload PDFs (domain-specific documents).  
2. Monitor processing status in real-time.  
3. Review annotations and synthetic data.  
4. Export datasets in multiple training formats.

---

## vLLM Setup

Deploy **vLLM with Qwen3:4B**:
```bash
docker run --gpus all -p 45000:80 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-4B
```
Set your `.env`:
```bash
VLLM_API_URL=http://<vllm-ip>:45000
```

> Make sure vLLM is running before starting the pipeline.

---

## Troubleshooting

- **Connection Errors:** Verify all IPs in `.env` files, check network connectivity.  
- **No Validated Data:** Check validation worker logs; ensure filenames follow expected formats.  
- **vLLM Errors:** Ensure server is running; restart with `docker restart vllm-container`.  
- **Dashboard Issues:** Check logs via `docker logs exapipelinedashboard-streamlit-1`.

---

## Repositories

- [exaOCR](https://github.com/ikantkode/exaOCR)  
- [exaPipeline](https://github.com/ikantkode/exaPipeline)  
- [exaPipelineDashboard](https://github.com/ikantkode/exaPipelineDashboard)  

---

## License

[MIT](LICENSE)

---

© 2025 **exaPipeline** | Production-Ready Document-to-LLM Training System
