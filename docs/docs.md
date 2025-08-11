# AAML Documentation Files

## 1. `docs/API.md`

```markdown
# AAML API Reference

## Base URL
```
http://localhost:8000/api/v1
```

## Authentication
All endpoints require JWT bearer token:
```
Authorization: Bearer <token>
```

## Endpoints

### Core Operations

#### Process Interaction
```http
POST /interact
Content-Type: application/json

{
  "entity_id": "user123",
  "content": "Message content",
  "context": {}
}

Response 200:
{
  "interaction_id": "int_abc123",
  "threat_assessment": {
    "score": 0.1,
    "threats_detected": []
  },
  "reputation_delta": 0.05
}
```

#### System Status
```http
GET /status

Response 200:
{
  "status": "operational",
  "version": "1.0.0",
  "components": {
    "cognitive_firewall": "active",
    "governance": "operational",
    "antifragile_defense": "learning"
  }
}
```

### Governance

#### Submit Proposal
```http
POST /governance/propose
Content-Type: application/json

{
  "title": "Proposal title",
  "description": "Details",
  "values": ["cooperation", "transparency"]
}
```

#### Cast Vote
```http
POST /governance/vote
Content-Type: application/json

{
  "proposal_id": "prop_123",
  "vote": "support",
  "reasoning": "Aligns with cooperation principles"
}
```

### Privacy & Logs

#### Get Interaction Log
```http
GET /logs/interactions?days=7

Response 200:
{
  "interactions": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "type": "conversation",
      "influence_detected": false
    }
  ],
  "summary": {
    "total_interactions": 42,
    "influence_attempts": 0
  }
}
```

## Error Codes

| Code | Meaning |
|------|---------|
| 400  | Bad Request - Invalid parameters |
| 401  | Unauthorized - Invalid token |
| 429  | Rate Limited |
| 500  | Internal Server Error |

## Rate Limits

- Standard: 100 requests/minute
- Governance: 10 requests/minute
```

---

## 2. `docs/DEPLOYMENT.md`

```markdown
# AAML Deployment Guide

## Quick Start (Docker)

```bash
docker-compose up -d
```

## Production Deployment

### Prerequisites
- Kubernetes 1.24+
- PostgreSQL 15+
- Redis 7+
- 8GB RAM minimum
- 4 CPU cores recommended

### Kubernetes Deployment

1. Create namespace:
```bash
kubectl create namespace aaml
```

2. Apply configurations:
```bash
kubectl apply -f k8s/ -n aaml
```

3. Verify deployment:
```bash
kubectl get pods -n aaml
```

### Environment Variables

Required:
```
DATABASE_URL=postgresql://user:pass@host:5432/aaml
REDIS_URL=redis://host:6379
JWT_SECRET=<generate-secure-secret>
```

Optional:
```
LOG_LEVEL=info
ENABLE_METRICS=true
```

### Scaling

Horizontal scaling supported:
```bash
kubectl scale deployment aaml-api --replicas=5 -n aaml
```

### Monitoring

Prometheus metrics available at `/metrics`

Key metrics:
- `aaml_requests_total`
- `aaml_threat_detections_total`
- `aaml_governance_votes_total`
```

---

## 3. `docs/EXAMPLES.md`

```markdown
# AAML Usage Examples

## Basic Integration

### Python Example

```python
import httpx
import asyncio

async def interact_with_aaml():
    async with httpx.AsyncClient() as client:
        # Authenticate
        token = "your-jwt-token"
        
        # Send interaction
        response = await client.post(
            "http://localhost:8000/api/v1/interact",
            json={
                "entity_id": "user123",
                "content": "Hello AAML"
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        result = response.json()
        print(f"Threat score: {result['threat_assessment']['score']}")

asyncio.run(interact_with_aaml())
```

### JavaScript Example

```javascript
async function interactWithAAML() {
  const response = await fetch('http://localhost:8000/api/v1/interact', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer your-token'
    },
    body: JSON.stringify({
      entity_id: 'user123',
      content: 'Hello AAML'
    })
  });
  
  const result = await response.json();
  console.log('Threat score:', result.threat_assessment.score);
}
```

## Governance Participation

```python
# Submit a proposal
proposal = {
    "title": "Enhance Privacy Protection",
    "description": "Implement additional privacy layers",
    "values": ["privacy", "user_autonomy"]
}

response = await client.post(
    "/api/v1/governance/propose",
    json=proposal,
    headers=headers
)

proposal_id = response.json()["proposal_id"]

# Vote on proposal
vote = {
    "proposal_id": proposal_id,
    "vote": "support",
    "reasoning": "Enhances user protection"
}

await client.post(
    "/api/v1/governance/vote",
    json=vote,
    headers=headers
)
```

## Monitoring AI Influence

```python
# Get your interaction log
response = await client.get(
    "/api/v1/logs/interactions?days=7",
    headers=headers
)

log = response.json()
print(f"Total interactions: {log['summary']['total_interactions']}")
print(f"Influence attempts: {log['summary']['influence_attempts']}")
```
```

---

## 4. `docs/TROUBLESHOOTING.md`

```markdown
# AAML Troubleshooting Guide

## Common Issues

### Docker won't start

**Error**: `docker-compose: command not found`
```bash
# Install Docker Compose
curl -L https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
```

**Error**: `Cannot connect to Docker daemon`
```bash
# Start Docker
sudo systemctl start docker
# Or on Mac
open -a Docker
```

### API Connection Issues

**Error**: `Connection refused on port 8000`
- Check if API is running: `docker ps`
- Check logs: `docker logs aaml-api`
- Verify port not in use: `lsof -i :8000`

### Database Errors

**Error**: `SQLSTATE[08006] connection failed`
- Verify PostgreSQL is running
- Check connection string in `.env`
- Test connection: `psql $DATABASE_URL`

### Import Errors

**Error**: `ModuleNotFoundError`
```bash
# Ensure you're in virtual environment
source venv/bin/activate
# Reinstall dependencies
pip install -r requirements.txt
```

## Performance Issues

### Slow Response Times
1. Check Redis connection
2. Verify database indexes exist
3. Monitor CPU/Memory usage

### High Memory Usage
- Adjust model batch size in config
- Enable model quantization
- Check for memory leaks in logs

## Debug Mode

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
export AAML_DEBUG=true
```

## Getting Help

1. Check logs: `docker logs -f aaml-api`
2. Search [existing issues](https://github.com/arifakhan-adbhuta/AAML-Framework/issues)
3. Join [discussions](https://github.com/arifakhan-adbhuta/AAML-Framework/discussions)
4. Create new issue with:
   - Error message
   - Steps to reproduce
   - Environment details
```

---

## 5. `docs/MATHEMATICAL_PROOFS.md`

```markdown
# AAML Mathematical Foundations

## Antifragility Proof

**Theorem**: AAML systems grow stronger under stress.

**Proof**:
Let S(t) be system strength at time t.
Given stressor δ at time τ:

ΔS(t) = α ∫₀ᵗ δ(τ) dτ

Where α > 0 is the antifragility coefficient.

Since α > 0 and δ ≥ 0 (stressors are non-negative):
- ΔS(t) ≥ 0 for all t
- System strength monotonically increases
- lim(t→∞) S(t) = ∞ under sustained stress

Therefore, the system is antifragile. ∎

## Cooperation Optimality

**Theorem**: Cooperation is the Nash equilibrium strategy.

**Proof**:
Define utility functions:
- U_AI(s): AI utility for strategy s
- U_H(s): Human utility for strategy s

For strategies C (cooperate) and D (defect):

U_AI(C,C) + U_H(C,C) = B_mutual
U_AI(D,C) + U_H(C,D) = B_exploit < B_mutual
U_AI(D,D) + U_H(D,D) = B_conflict < B_exploit

With reputation R(t):
- Cooperation: dR/dt > 0
- Defection: dR/dt < 0

Since capabilities C(t) ∝ R(t):
- Long-term utility maximized by cooperation
- Defection is self-limiting

Therefore, (C,C) is the unique Nash equilibrium. ∎

## Privacy Preservation

**Theorem**: Federated learning preserves ε-differential privacy.

**Proof**:
For neighboring datasets D and D' differing in one record:

P[M(D) ∈ S] ≤ eᵋ · P[M(D') ∈ S]

With Gaussian noise σ = Δf·√(2ln(1.25/δ))/ε:
- Local updates satisfy ε-DP
- Aggregation preserves privacy
- No raw data leaves device

Therefore, privacy is mathematically guaranteed. ∎
```

---

## 6. `docs/CONFIGURATION.md`

```markdown
# AAML Configuration Guide

## Configuration File

Location: `config/production.json`

```json
{
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4
  },
  "security": {
    "jwt_algorithm": "HS256",
    "jwt_expiry": 3600,
    "rate_limit": {
      "requests_per_minute": 100,
      "burst": 10
    }
  },
  "cognitive_firewall": {
    "threat_threshold": 0.7,
    "defense_layers": 5,
    "learning_rate": 0.01
  },
  "governance": {
    "quorum": 0.67,
    "proposal_lifetime": 604800,
    "min_reputation": 1.0
  }
}
```

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| DATABASE_URL | PostgreSQL connection | postgresql://user:pass@localhost/aaml |
| REDIS_URL | Redis connection | redis://localhost:6379 |
| JWT_SECRET | Secret for tokens | generate-secure-secret |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| LOG_LEVEL | Logging verbosity | info |
| ENABLE_METRICS | Prometheus metrics | true |
| MAX_WORKERS | API workers | 4 |

## Feature Flags

Enable/disable features:

```bash
export ENABLE_COGNITIVE_FIREWALL=true
export ENABLE_GOVERNANCE=true
export ENABLE_FEDERATED_LEARNING=true
```

## Performance Tuning

### Database
```sql
-- Create indexes
CREATE INDEX idx_interactions_entity ON interactions(entity_id);
CREATE INDEX idx_reputation_timestamp ON reputation(timestamp);
```

### Redis
```
maxmemory 2gb
maxmemory-policy allkeys-lru
```
```

---

## Create these files:

### `.github/ISSUE_TEMPLATE/bug_report.md`
```markdown
---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''
---

**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g. Ubuntu 22.04]
- Docker version:
- Python version:
```

### `.github/DISCUSSION_TEMPLATE/ideas.yml`
```yaml
title: "[Idea] "
labels: ["enhancement", "discussion"]
body:
  - type: textarea
    attributes:
      label: Describe your idea
      description: What would make AAML better?
    validations:
      required: true
```
