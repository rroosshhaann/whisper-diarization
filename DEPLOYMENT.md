# Cloud Deployment Guide

Deploy whisper-diarization to the cloud for production use.

## Deployment Options Comparison

| Option | Cold Start | Cost (10hr/mo) | Auto-scale | Setup |
|--------|------------|----------------|------------|-------|
| **GCE Spot + Trigger** | ~60 sec | ~$2 | On-demand | Medium |
| **GCE Always-On** | 0 | ~$80 | No | Easy |
| **Cloud Run (GPU)** | 2-5 min | ~$36 | Yes | Easy |
| **Local (your machine)** | 0 | $0 | No | Done |

**Recommendation:** GCE Spot with Cloud Function trigger for intermittent use.

---

## Option 1: GCE Spot with Auto Start/Stop (Recommended)

Best for intermittent use. VM starts on-demand and shuts down when idle.

### Architecture

```
Your App
    │
    ▼
Cloud Function (trigger)
    │ "Is VM running?"
    │ No → Start VM
    │ Yes → Return IP
    ▼
GCE Spot VM (GPU)
    │
    ▼
Whisper Diarization API
    │
    ▼ (30 min idle)
Auto-shutdown
```

### Cost Breakdown

| Component | Price | Monthly (10hr use) |
|-----------|-------|-------------------|
| GCE Spot T4 | $0.11/hr | $1.10 |
| Boot disk (50GB) | $0.04/GB | $2.00 |
| Cloud Function | $0.0000004/call | ~$0.01 |
| **Total** | | **~$3.11** |

### Setup

#### Step 1: Create GCE VM

```bash
# Set variables
export PROJECT_ID="your-project-id"
export ZONE="australia-southeast1-b"
export INSTANCE_NAME="whisper-diarization"

# Create VM with GPU
gcloud compute instances create $INSTANCE_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=cos-stable \
  --image-project=cos-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-ssd \
  --tags=whisper-api \
  --metadata-from-file=startup-script=startup.sh
```

#### Step 2: Create Startup Script

Save as `startup.sh`:

```bash
#!/bin/bash

# Install NVIDIA driver (Container-Optimized OS)
cos-extensions install gpu

# Pull and run container
docker-credential-gcr configure-docker
docker pull gcr.io/$PROJECT_ID/whisper-diarization:latest

docker run -d \
  --name whisper-api \
  --restart=unless-stopped \
  --gpus all \
  -p 8001:8001 \
  gcr.io/$PROJECT_ID/whisper-diarization:latest

# Setup idle shutdown (30 min)
cat > /tmp/idle-shutdown.sh << 'EOF'
#!/bin/bash
IDLE_MINUTES=30
LAST_ACTIVITY_FILE="/tmp/last_api_activity"

# Check API health endpoint for recent activity
if curl -s http://localhost:8001/health | grep -q '"processing_jobs": 0'; then
  if [ -f "$LAST_ACTIVITY_FILE" ]; then
    LAST_ACTIVITY=$(cat "$LAST_ACTIVITY_FILE")
    NOW=$(date +%s)
    IDLE_TIME=$(( (NOW - LAST_ACTIVITY) / 60 ))

    if [ $IDLE_TIME -ge $IDLE_MINUTES ]; then
      echo "Idle for $IDLE_TIME minutes, shutting down..."
      shutdown -h now
    fi
  fi
else
  # Activity detected, update timestamp
  date +%s > "$LAST_ACTIVITY_FILE"
fi
EOF

chmod +x /tmp/idle-shutdown.sh

# Run idle check every 5 minutes
echo "*/5 * * * * /tmp/idle-shutdown.sh" | crontab -
```

#### Step 3: Open Firewall

```bash
gcloud compute firewall-rules create allow-whisper-api \
  --project=$PROJECT_ID \
  --allow=tcp:8001 \
  --target-tags=whisper-api \
  --source-ranges=0.0.0.0/0
```

#### Step 4: Push Docker Image to GCR

```bash
# Authenticate
gcloud auth configure-docker

# Build and push
docker build -t gcr.io/$PROJECT_ID/whisper-diarization:latest .
docker push gcr.io/$PROJECT_ID/whisper-diarization:latest
```

#### Step 5: Create Cloud Function (Trigger)

Create `main.py`:

```python
import functions_framework
from google.cloud import compute_v1
import time

PROJECT_ID = "your-project-id"
ZONE = "australia-southeast1-b"
INSTANCE_NAME = "whisper-diarization"

@functions_framework.http
def start_vm(request):
    """Start the VM if stopped, return its IP."""

    instance_client = compute_v1.InstancesClient()

    # Get instance status
    instance = instance_client.get(
        project=PROJECT_ID,
        zone=ZONE,
        instance=INSTANCE_NAME
    )

    status = instance.status

    if status == "TERMINATED" or status == "STOPPED":
        # Start the instance
        operation = instance_client.start(
            project=PROJECT_ID,
            zone=ZONE,
            instance=INSTANCE_NAME
        )

        # Wait for instance to start (up to 2 min)
        for _ in range(24):
            time.sleep(5)
            instance = instance_client.get(
                project=PROJECT_ID,
                zone=ZONE,
                instance=INSTANCE_NAME
            )
            if instance.status == "RUNNING":
                break

        # Wait additional time for container to start
        time.sleep(30)

    # Get external IP
    instance = instance_client.get(
        project=PROJECT_ID,
        zone=ZONE,
        instance=INSTANCE_NAME
    )

    external_ip = None
    for interface in instance.network_interfaces:
        for access_config in interface.access_configs:
            if access_config.nat_i_p:
                external_ip = access_config.nat_i_p
                break

    if not external_ip:
        return {"error": "No external IP found"}, 500

    return {
        "status": "running",
        "api_url": f"http://{external_ip}:8001",
        "was_started": status in ("TERMINATED", "STOPPED")
    }
```

Create `requirements.txt`:

```
functions-framework==3.*
google-cloud-compute==1.*
```

Deploy:

```bash
gcloud functions deploy whisper-trigger \
  --gen2 \
  --runtime=python311 \
  --region=australia-southeast1 \
  --source=. \
  --entry-point=start_vm \
  --trigger-http \
  --allow-unauthenticated \
  --timeout=180s
```

### Usage

```python
import requests
import time

TRIGGER_URL = "https://australia-southeast1-YOUR_PROJECT.cloudfunctions.net/whisper-trigger"

# Step 1: Wake up VM (or get IP if already running)
response = requests.get(TRIGGER_URL)
api_url = response.json()["api_url"]

# Step 2: Submit job
with open("recording.webm", "rb") as f:
    job = requests.post(f"{api_url}/jobs", files={"file": f}).json()

# Step 3: Poll for result
while True:
    status = requests.get(f"{api_url}/jobs/{job['job_id']}").json()
    if status["status"] == "completed":
        break
    time.sleep(5)

# Step 4: Get result
result = requests.get(f"{api_url}/jobs/{job['job_id']}/result").json()
```

---

## Option 2: GCE Always-On

Simplest setup. VM runs 24/7.

### Setup

Same as Option 1, but without:
- Spot provisioning (remove `--provisioning-model=SPOT`)
- Idle shutdown script

### Cost

~$80/month for n1-standard-4 + T4 GPU in australia-southeast1.

---

## Option 3: Cloud Run with GPU

True serverless with auto-scaling.

### Limitations

- **Cold start: 2-5 minutes** (large image, GPU allocation)
- Only available in limited regions (us-central1, europe-west4)
- **Not available in Australia** as of Dec 2024

### Setup

```bash
# Deploy to Cloud Run
gcloud run deploy whisper-diarization \
  --image gcr.io/$PROJECT_ID/whisper-diarization:latest \
  --region us-central1 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --memory 16Gi \
  --cpu 4 \
  --timeout 600 \
  --min-instances 0 \
  --max-instances 1 \
  --allow-unauthenticated
```

---

## Security Recommendations

### 1. Add Authentication

The API has no built-in auth. Options:

**API Key Header:**
```python
# In api.py
API_KEY = os.environ.get("API_KEY", "")

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return JSONResponse(status_code=401, content={"error": "Invalid API key"})
    return await call_next(request)
```

**Cloud IAP (Identity-Aware Proxy):**
```bash
# Enable IAP on the firewall rule
gcloud compute backend-services update whisper-backend \
  --iap=enabled
```

### 2. Use HTTPS

Put behind a load balancer with SSL:

```bash
# Create SSL cert
gcloud compute ssl-certificates create whisper-cert \
  --domains=diarize.yourdomain.com

# Create load balancer
gcloud compute url-maps create whisper-lb \
  --default-service whisper-backend
```

### 3. Restrict Source IPs

```bash
gcloud compute firewall-rules update allow-whisper-api \
  --source-ranges=YOUR_APP_IP/32
```

---

## Monitoring

### Cloud Monitoring

```bash
# Create uptime check
gcloud monitoring uptime-check-configs create whisper-health \
  --display-name="Whisper API Health" \
  --http-check-path="/health" \
  --http-check-port=8001
```

### Logging

View container logs:

```bash
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- \
  docker logs -f whisper-api
```

---

## Troubleshooting

### VM won't start

```bash
# Check quota
gcloud compute regions describe australia-southeast1 \
  --format="table(quotas.metric,quotas.limit,quotas.usage)"

# GPU quota is often 0 by default - request increase
```

### Container won't start

```bash
# SSH into VM
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE

# Check docker
docker ps -a
docker logs whisper-api
```

### GPU not detected

```bash
# On Container-Optimized OS
cos-extensions list
cos-extensions install gpu

# Verify
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

## Cost Optimization Tips

1. **Use Spot VMs** - 60-70% discount, may be preempted
2. **T4 over L4** - T4 is cheaper and sufficient for this workload
3. **Idle shutdown** - Don't pay when not using
4. **Committed use** - 1-year commit for ~57% discount if always-on
5. **Off-peak scheduling** - Schedule VM for business hours only

---

## Region Availability

| Region | GPU Types | Spot Available |
|--------|-----------|----------------|
| australia-southeast1 (Sydney) | T4, L4, A100 | Yes |
| australia-southeast2 (Melbourne) | T4 | Yes |
| us-central1 | All | Yes |
| europe-west4 | All | Yes |
