# Whisper Diarization REST API

REST API wrapper for whisper-diarization with async job queue processing.

## Overview

This API provides:
- **Async job queue** - Submit audio files and poll for results
- **Deepgram-compatible response format** - Drop-in replacement for Deepgram API
- **GPU-accelerated processing** - Uses CUDA for fast transcription and diarization
- **Sequential job processing** - One job at a time to maximize GPU efficiency

## Quick Start

### Docker (Recommended)

```bash
# Build and run
docker compose up --build

# API available at http://localhost:8001
```

### Local Development

```bash
# Install dependencies
pip install -c constraints.txt -r requirements.txt

# Run API server
uvicorn api:app --host 0.0.0.0 --port 8001
```

## API Endpoints

### Submit Job

```
POST /jobs
Content-Type: multipart/form-data
```

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `file` | file | (required) | Audio file (wav, mp3, webm, m4a, etc.) |
| `whisper_model` | string | `medium.en` | Whisper model name |
| `language` | string | `null` | Language code or null for auto-detect |
| `stemming` | boolean | `true` | Separate vocals from music |
| `suppress_numerals` | boolean | `false` | Convert digits to written text |
| `batch_size` | integer | `8` | Batch size for inference |

**Available Whisper Models:**
| Model | Parameters | VRAM | Speed | Accuracy |
|-------|------------|------|-------|----------|
| `tiny` / `tiny.en` | 39M | ~1GB | Fastest | Lower |
| `base` / `base.en` | 74M | ~1GB | Fast | Low |
| `small` / `small.en` | 244M | ~2GB | Medium | Good |
| `medium` / `medium.en` | 769M | ~5GB | Slow | Better |
| `large-v2` | 1550M | ~10GB | Slowest | Best |
| `large-v3` | 1550M | ~10GB | Slowest | Best |

**Response:** `201 Created`
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "position": 0
}
```

**Example:**
```bash
curl -X POST http://localhost:8001/jobs \
  -F "file=@recording.webm" \
  -F "whisper_model=medium.en" \
  -F "stemming=true"
```

---

### Get Job Status

```
GET /jobs/{job_id}
```

**Response:**
```json
// Queued
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "position": 2
}

// Processing
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": "transcribing"
}

// Completed
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed"
}

// Failed
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "error": "Error message"
}
```

**Progress Stages:**
1. `separating_vocals` - Extracting vocals from audio (if stemming enabled)
2. `transcribing` - Running Whisper ASR
3. `aligning` - Forced alignment with CTC
4. `diarizing` - Speaker identification with NeMo
5. `post_processing` - Punctuation and formatting
6. `generating_output` - Building response
7. `completed` - Done

---

### Get Job Result

```
GET /jobs/{job_id}/result
```

**Response:** Deepgram-compatible format
```json
{
  "metadata": {
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "model_info": {
      "name": "medium.en"
    },
    "duration": 1742.5
  },
  "results": {
    "channels": [
      {
        "alternatives": [
          {
            "transcript": "Hello, how are you today?...",
            "confidence": 0.95,
            "words": [
              {
                "word": "Hello,",
                "start": 0.5,
                "end": 0.8,
                "confidence": 0.95,
                "speaker": 0,
                "speaker_confidence": 0.85,
                "punctuated_word": "Hello,"
              }
            ]
          }
        ]
      }
    ],
    "utterances": [
      {
        "start": 0.5,
        "end": 3.2,
        "confidence": 0.95,
        "channel": 0,
        "transcript": "Hello, how are you today?",
        "words": [...],
        "speaker": 0,
        "id": "550e8400-e29b-41d4-a716-446655440001"
      }
    ]
  }
}
```

---

### Delete Job

```
DELETE /jobs/{job_id}
```

Cancel a queued job or cleanup a completed job.

**Response:**
```json
{
  "message": "Job deleted"
}
```

---

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "queued_jobs": 2,
  "processing_jobs": 1
}
```

---

## Usage Examples

### Python Client

```python
import requests
import time

API_URL = "http://localhost:8001"

# Submit job
with open("recording.webm", "rb") as f:
    response = requests.post(
        f"{API_URL}/jobs",
        files={"file": f},
        data={"whisper_model": "medium.en"}
    )
job_id = response.json()["job_id"]

# Poll for completion
while True:
    status = requests.get(f"{API_URL}/jobs/{job_id}").json()
    print(f"Status: {status['status']}, Progress: {status.get('progress', 'N/A')}")

    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        raise Exception(status["error"])

    time.sleep(5)

# Get result
result = requests.get(f"{API_URL}/jobs/{job_id}/result").json()

# Access transcript
transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
utterances = result["results"]["utterances"]

for u in utterances:
    print(f"Speaker {u['speaker']}: {u['transcript']}")
```

### JavaScript/Node.js Client

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const API_URL = 'http://localhost:8001';

async function transcribe(audioPath) {
  // Submit job
  const form = new FormData();
  form.append('file', fs.createReadStream(audioPath));
  form.append('whisper_model', 'medium.en');

  const submitRes = await axios.post(`${API_URL}/jobs`, form, {
    headers: form.getHeaders()
  });
  const jobId = submitRes.data.job_id;

  // Poll for completion
  while (true) {
    const statusRes = await axios.get(`${API_URL}/jobs/${jobId}`);
    const status = statusRes.data;

    console.log(`Status: ${status.status}, Progress: ${status.progress || 'N/A'}`);

    if (status.status === 'completed') break;
    if (status.status === 'failed') throw new Error(status.error);

    await new Promise(r => setTimeout(r, 5000));
  }

  // Get result
  const resultRes = await axios.get(`${API_URL}/jobs/${jobId}/result`);
  return resultRes.data;
}
```

### cURL

```bash
# Submit job
JOB_ID=$(curl -s -X POST http://localhost:8001/jobs \
  -F "file=@recording.webm" | jq -r '.job_id')

# Poll status
curl http://localhost:8001/jobs/$JOB_ID

# Get result (when completed)
curl http://localhost:8001/jobs/$JOB_ID/result > result.json
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UPLOAD_DIR` | `/tmp/diarization_uploads` | Temporary file storage |
| `JOB_EXPIRY_SECONDS` | `3600` | Time before completed jobs are cleaned up |

### Docker Compose

```yaml
services:
  whisper-diarization:
    build: .
    container_name: whisper-diarization
    ports:
      - "8001:8001"
    volumes:
      - ./data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

---

## Performance

Tested on RTX 5090 (24GB VRAM):

| Audio Length | Processing Time | Model |
|--------------|-----------------|-------|
| 5 min | ~30 sec | medium.en |
| 30 min | ~3-4 min | medium.en |
| 60 min | ~7-8 min | medium.en |

Processing time includes:
- Vocal separation (demucs)
- Transcription (faster-whisper)
- Forced alignment (ctc-forced-aligner)
- Speaker diarization (NeMo)
- Post-processing (punctuation)

---

## Limitations

- **Single worker** - Processes one job at a time (GPU constraint)
- **In-memory queue** - Jobs don't survive container restarts
- **No authentication** - Add your own auth layer for production
- **Overlapping speakers** - Not yet supported

See [BACKLOG.md](BACKLOG.md) for planned improvements.
