# Feature Backlog

## WD-001: Concurrent Processing Based on VRAM

**Priority:** Medium
**Status:** Not Started
**Requested:** 2025-12-10

### Description
Enable concurrent job processing based on available GPU VRAM. Currently processes one job at a time, but systems with more VRAM (e.g., 24GB RTX 5090) could handle multiple concurrent jobs.

### Implementation Ideas
1. **Detect available VRAM** at startup using `torch.cuda.get_device_properties()`
2. **Estimate memory per job** (~4-6GB peak for diarization pipeline)
3. **Spawn multiple worker threads** based on: `max_workers = available_vram // memory_per_job`
4. **Dynamic throttling** - monitor VRAM usage and pause new jobs if memory pressure is high

### Considerations
- Each job loads models sequentially (Whisper → Alignment → Diarization)
- Models are deleted after each stage with `torch.cuda.empty_cache()`
- True concurrent processing may need separate CUDA streams or processes
- Consider using `torch.cuda.memory_reserved()` for monitoring

### Rough Estimate
| VRAM | Max Concurrent Jobs |
|------|---------------------|
| 24GB | 3-4 |
| 16GB | 2 |
| 8GB  | 1 (current) |

### References
- `diarize_parallel.py` in original repo shows parallel Whisper + Diarization approach

---

## WD-002: Persistent Job Queue (Redis/SQLite)

**Priority:** Low
**Status:** Not Started
**Requested:** -

### Description
Replace in-memory job queue with persistent storage so jobs survive container restarts.

### Implementation Ideas
- Use Redis for job queue (production)
- Or SQLite for simpler deployments
- Store job metadata, status, and results

### Considerations
- Current in-memory queue is fine for single-user/dev use
- Only needed if running as always-on service with critical workloads

---

## WD-003: Webhook Callbacks

**Priority:** Low
**Status:** Not Started
**Requested:** -

### Description
Instead of polling, allow clients to provide a webhook URL that gets called when job completes.

### Implementation Ideas
```python
POST /jobs
{
  "file": ...,
  "webhook_url": "https://myapp.com/diarization-complete"
}
```

### Considerations
- Useful for fire-and-forget integrations
- Need retry logic for failed webhook calls

---

## WD-004: Model Preloading / Warm Start

**Priority:** Medium
**Status:** Not Started
**Requested:** -

### Description
Keep models loaded in memory between jobs to reduce startup latency. First job currently downloads/loads all models.

### Implementation Ideas
- Load Whisper, Alignment, and Diarization models at server startup
- Keep in memory (increases baseline VRAM usage)
- Trade-off: faster processing vs higher idle memory

### Considerations
- Would use ~6-8GB VRAM at idle
- First job would be much faster (skip model loading)
- May conflict with concurrent processing goal (WD-001)

---

## WD-005: GCE Auto Start/Stop Deployment

**Priority:** Medium
**Status:** Documented
**Requested:** 2025-12-10

### Description
Deploy to Google Cloud with automatic VM start/stop for cost-effective intermittent use. VM starts when request arrives, processes jobs, then shuts down after idle period.

### Architecture
```
Request → Cloud Function → Start VM (if stopped) → Return IP
                                    ↓
                           GCE Spot VM (GPU)
                                    ↓
                           Whisper Diarization API
                                    ↓
                           Auto-shutdown (30 min idle)
```

### Implementation
See [DEPLOYMENT.md](DEPLOYMENT.md) for full setup guide.

**Components:**
1. GCE Spot VM with T4/L4 GPU in australia-southeast1
2. Cloud Function trigger to start VM on-demand
3. Idle shutdown script (cron-based)
4. Docker image in Google Container Registry

### Cost Estimate
| Usage | Monthly Cost |
|-------|--------------|
| 10 hours | ~$3 |
| 40 hours | ~$8 |
| Always-on | ~$80 |

### Considerations
- Cold start ~60 seconds (VM boot + container start)
- Spot VMs may be preempted during high demand
- No Australian Cloud Run GPU support yet

---

## Template

```markdown
## WD-XXX: Feature Name

**Priority:** High/Medium/Low
**Status:** Not Started / In Progress / Done
**Requested:** YYYY-MM-DD

### Description
What the feature does.

### Implementation Ideas
How to build it.

### Considerations
Trade-offs and edge cases.
```
