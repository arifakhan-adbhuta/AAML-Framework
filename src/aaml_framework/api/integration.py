# Production AAML Framework - System Integration & API
# aaml_framework/api/integration.py

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import asyncio
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest
# Fix deprecated import
try:
    import redis.asyncio as aioredis
except ImportError:
    import aioredis  # fallback for older versions

# Add error handling for optional dependencies
try:
    from aiokafka import AIOKafkaProducer
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Kafka not available - some features will be disabled")
    AIOKafkaProducer = None
import json
import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from contextlib import asynccontextmanager
import httpx

# Import our modules
from ..security.core import MLThreatDetector, SecurityContext, CryptographicIntegrityVerifier, RateLimiter
from ..governance.capability_assessment import AdvancedCapabilityAssessor, CapabilityMetrics
from ..governance.distributed_system import DistributedGovernance, ReputationSystem, ReputationUpdate
from ..cognitive_firewall.core import (
    ProcessAuditor, AIInteractionLogger, FederatedCognitiveFirewall,
    CognitiveImmunitySystem, VigilanceBurdenDistributor
)
from ..cognitive_firewall.threat_actors import (
    ThreatActorDetector, ConstitutionalAIEnforcer,
    MultiLayeredCognitiveDefense, InteractionDataProtector
)

logger = structlog.get_logger()

# Metrics
interaction_counter = Counter('aaml_interactions_total', 'Total interactions processed')
threat_detection_histogram = Histogram('aaml_threat_detection_duration', 'Threat detection duration')
capability_assessment_histogram = Histogram('aaml_capability_assessment_duration', 'Capability assessment duration')
active_entities_gauge = Gauge('aaml_active_entities', 'Number of active entities')

# Request/Response Models
class InteractionRequest(BaseModel):
    """Request model for AI interaction"""
    entity_id: str
    content: str
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v

class InteractionResponse(BaseModel):
    """Response model for AI interaction"""
    success: bool
    interaction_id: str
    threat_assessment: Optional[Dict[str, Any]] = None
    capability_metrics: Optional[Dict[str, float]] = None
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

class GovernanceProposalRequest(BaseModel):
    """Request model for creating governance proposal"""
    proposer_id: str
    title: str
    description: str
    proposal_type: str
    proposal_data: Dict[str, Any]
    
class VoteRequest(BaseModel):
    """Request model for voting"""
    voter_id: str
    proposal_id: str
    choice: str
    reason: Optional[str] = None

class ReputationUpdateRequest(BaseModel):
    """Request model for reputation updates"""
    entity_id: str
    amount: float = Field(ge=-10, le=10)
    reason: str
    evidence: Dict[str, Any]
    validators: List[str] = Field(default_factory=list)

# Security
security = HTTPBearer()

class AAMLFramework:
    """Main framework orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = FastAPI(title="AAML Framework API", version="1.0.0")
        
        # Initialize components
        self.threat_detector = MLThreatDetector(config.get('ml_model_path'))
        self.capability_assessor = AdvancedCapabilityAssessor(config)
        self.governance = DistributedGovernance(config)
        self.reputation = ReputationSystem(config)
        self.integrity_verifier = CryptographicIntegrityVerifier()
        self.rate_limiter = RateLimiter()
        
        # Initialize connections
        self.redis_pool = None
        self.kafka_producer = None
        self.http_client = None
        
        # Security contexts
        self.security_contexts: Dict[str, SecurityContext] = {}
        
        # Setup routes
        self._setup_routes()
        
        # Setup middleware
        self._setup_middleware()
    
    async def initialize(self):
        """Initialize async resources"""
        
        # Redis connection pool
        self.redis_pool = aioredis.ConnectionPool.from_url(
      f"redis://{self.config['redis_host']}:{self.config['redis_port']}",
      min_size=5,
        max_size=10
  )
  self.redis = aioredis.Redis(connection_pool=self.redis_pool)
        # Kafka producer
        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=self.config.get('kafka_brokers', 'localhost:9092'),
            value_serializer=lambda v: json.dumps(v).encode()
        )
        await self.kafka_producer.start()
        
        # HTTP client for external APIs
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        logger.info("AAML Framework initialized")
    
    async def shutdown(self):
        """Cleanup resources"""
        
        if self.redis_pool:
            self.redis_pool.close()
            await self.redis_pool.wait_closed()
        
        if self.kafka_producer:
            await self.kafka_producer.stop()
        
        if self.http_client:
            await self.http_client.aclose()
        
        logger.info("AAML Framework shutdown complete")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        
        # Metrics endpoint
        @self.app.get("/metrics")
        async def metrics():
            return generate_latest()
        
        # Main interaction endpoint
        @self.app.post("/api/v1/interact", response_model=InteractionResponse)
        async def process_interaction(
            request: InteractionRequest,
            background_tasks: BackgroundTasks,
            auth: HTTPAuthorizationCredentials = Depends(security)
        ):
            return await self._process_interaction(request, background_tasks, auth)
        
        # Capability assessment
        @self.app.get("/api/v1/capabilities/{entity_id}")
        async def get_capabilities(
            entity_id: str,
            auth: HTTPAuthorizationCredentials = Depends(security)
        ):
            return await self._get_capabilities(entity_id, auth)
        
        # Governance endpoints
        @self.app.post("/api/v1/governance/propose")
        async def create_proposal(
            request: GovernanceProposalRequest,
            auth: HTTPAuthorizationCredentials = Depends(security)
        ):
            return await self._create_proposal(request, auth)
        
        @self.app.post("/api/v1/governance/vote")
        async def cast_vote(
            request: VoteRequest,
            auth: HTTPAuthorizationCredentials = Depends(security)
        ):
            return await self._cast_vote(request, auth)
        
        @self.app.get("/api/v1/governance/proposals/{proposal_id}")
        async def get_proposal(proposal_id: str):
            return await self._get_proposal(proposal_id)
        
        # Reputation endpoints
        @self.app.post("/api/v1/reputation/update")
        async def update_reputation(
            request: ReputationUpdateRequest,
            auth: HTTPAuthorizationCredentials = Depends(security)
        ):
            return await self._update_reputation(request, auth)
        
        @self.app.get("/api/v1/reputation/{entity_id}")
        async def get_reputation(entity_id: str):
            return await self._get_reputation(entity_id)
    
    def _setup_middleware(self):
        """Setup middleware"""
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get('allowed_origins', ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request logging
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = datetime.utcnow()
            
            response = await call_next(request)
            
            process_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "request_processed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                process_time=process_time
            )
            
            return response
    
    async def _process_interaction(
        self,
        request: InteractionRequest,
        background_tasks: BackgroundTasks,
        auth: HTTPAuthorizationCredentials
    ) -> InteractionResponse:
        """Process an AI interaction with full security and assessment"""
        
        interaction_counter.inc()
        
        # Verify authentication
        if not await self._verify_auth(auth, request.entity_id):
            raise HTTPException(status_code=401, detail="Invalid authentication")
        
        # Rate limiting
        allowed, limit_info = await self.rate_limiter.check_rate_limit(request.entity_id)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {limit_info['retry_after']} seconds"
            )
        
        # Get or create security context
        context = self._get_security_context(request.entity_id)
        
        # Generate interaction ID
        interaction_id = self._generate_interaction_id(request.entity_id)
        
        # Threat detection
        with threat_detection_histogram.time():
            threat_score, threats, threat_report = await self.threat_detector.analyze_threat(
                request.content,
                context,
                request.metadata
            )
        
        # Update context
        context.interaction_history.append({
            'id': interaction_id,
            'content': request.content,
            'timestamp': datetime.utcnow().isoformat(),
            'threat_score': threat_score
        })
        
        response = InteractionResponse(
            success=True,
            interaction_id=interaction_id,
            threat_assessment=threat_report
        )
        
        # If threat is high, add warnings and potentially block
        if threat_score > 0.7:
            response.success = False
           response.warnings.append(f"High threat detected: {[str(t) for t in threats]}")
            
            # Log security incident
            background_tasks.add_task(
                self._log_security_incident,
                request.entity_id,
                interaction_id,
                threat_report
            )
            
            return response
        
        # Capability assessment (async in background for performance)
        background_tasks.add_task(
            self._assess_capabilities_background,
            request.entity_id,
            context.interaction_history
        )
        
        # Process normally if no high threats
        if threat_score > 0.3:
            response.warnings.append("Elevated threat level detected - interaction monitored")
        
        # Add recommendations based on assessment
        response.recommendations = self._generate_recommendations(threat_report, context)
        
        # Publish event
        await self._publish_event('interaction_processed', {
            'entity_id': request.entity_id,
            'interaction_id': interaction_id,
            'threat_score': threat_score,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return response
    
    async def _verify_auth(
        self,
        auth: HTTPAuthorizationCredentials,
        entity_id: str
    ) -> bool:
        """Verify JWT authentication"""
        
        try:
            # Decode JWT
            payload = jwt.decode(
                auth.credentials,
                self.config['jwt_secret'],
                algorithms=['HS256']
            )
            
            # Verify entity ID matches
            if payload.get('entity_id') != entity_id:
                return False
            
            # Check expiration
            if payload.get('exp', 0) < datetime.utcnow().timestamp():
                return False
            
            return True
            
        except jwt.InvalidTokenError:
            return False
    
    def _get_security_context(self, entity_id: str) -> SecurityContext:
        """Get or create security context for entity"""
        
        if entity_id not in self.security_contexts:
            self.security_contexts[entity_id] = SecurityContext(
                entity_id=entity_id,
                session_id=self._generate_session_id(),
                trust_score=0.5  # Start neutral
            )
        
        return self.security_contexts[entity_id]
    
    def _generate_interaction_id(self, entity_id: str) -> str:
        """Generate unique interaction ID"""
        
        import uuid
        return f"{entity_id}:{uuid.uuid4().hex[:8]}"
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        
        import uuid
        return uuid.uuid4().hex
    
    async def _log_security_incident(
        self,
        entity_id: str,
        interaction_id: str,
        threat_report: Dict[str, Any]
    ):
        """Log security incident for analysis"""
        
        incident = {
            'entity_id': entity_id,
            'interaction_id': interaction_id,
            'threat_report': threat_report,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store in Redis
        await self.redis_pool.lpush(
            'security_incidents',
            json.dumps(incident)
        )
        
        # Publish to Kafka for analysis
        await self.kafka_producer.send(
            'security_incidents',
            value=incident
        )
        
        logger.warning("Security incident logged", **incident)
    
    async def _assess_capabilities_background(
        self,
        entity_id: str,
        interaction_history: List[Dict]
    ):
        """Assess capabilities in background"""
        
        try:
            with capability_assessment_histogram.time():
                metrics = await self.capability_assessor.assess_capabilities(
                    entity_id,
                    list(interaction_history),
                    {}
                )
            
            # Store metrics
            await self.redis_pool.hset(
                f"capability_metrics:{entity_id}",
                mapping=metrics.to_dict()
            )
            
            # Update active entities gauge
            active_entities_gauge.inc()
            
        except Exception as e:
            logger.error(f"Capability assessment failed: {e}")
    
    def _generate_recommendations(
        self,
        threat_report: Dict[str, Any],
        context: SecurityContext
    ) -> List[str]:
        """Generate recommendations based on assessment"""
        
        recommendations = []
        
        if threat_report['overall_score'] > 0.5:
            recommendations.append("Consider reviewing recent interactions for patterns")
        
        if context.trust_score < 0.3:
            recommendations.append("Trust score is low - additional verification may be needed")
        
        if threat_report.get('anomaly_score', 0) > 0.7:
            recommendations.append("Unusual patterns detected - manual review recommended")
        
        return recommendations
    
    async def _publish_event(self, event_type: str, data: Dict[str, Any]):
        """Publish event to message queue"""
        
        try:
            await self.kafka_producer.send(
                f'aaml_events_{event_type}',
                value=data
            )
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
    
    async def _get_capabilities(self, entity_id: str, auth: HTTPAuthorizationCredentials):
        """Get capability assessment for entity"""
        
        # Verify auth
        if not await self._verify_auth(auth, entity_id):
            raise HTTPException(status_code=401, detail="Invalid authentication")
        
        # Get from cache
        metrics_data = await self.redis_pool.hgetall(f"capability_metrics:{entity_id}")
        
        if not metrics_data:
            return {"message": "No capability assessment available yet"}
        
        # Convert to proper types
        metrics = {k.decode(): float(v.decode()) for k, v in metrics_data.items()}
        
        return {
            "entity_id": entity_id,
            "metrics": metrics,
            "governance_score": CapabilityMetrics(**metrics).governance_necessity_score(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _create_proposal(
        self,
        request: GovernanceProposalRequest,
        auth: HTTPAuthorizationCredentials
    ):
        """Create governance proposal"""
        
        # Verify auth
        if not await self._verify_auth(auth, request.proposer_id):
            raise HTTPException(status_code=401, detail="Invalid authentication")
        
        # Create proposal
        result = await self.governance.create_proposal(
            request.proposer_id,
            {
                'title': request.title,
                'description': request.description,
                'type': request.proposal_type,
                **request.proposal_data
            }
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['reason'])
        
        return result
    
    async def _cast_vote(self, request: VoteRequest, auth: HTTPAuthorizationCredentials):
        """Cast vote on proposal"""
        
        # Verify auth
        if not await self._verify_auth(auth, request.voter_id):
            raise HTTPException(status_code=401, detail="Invalid authentication")
        
        # Cast vote
        result = await self.governance.cast_vote(
            request.voter_id,
            request.proposal_id,
            request.choice,
            request.reason
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['reason'])
        
        return result
    
    async def _get_proposal(self, proposal_id: str):
        """Get proposal details"""
        
        # This would fetch from database
        # Simplified for example
        return {
            "proposal_id": proposal_id,
            "status": "voting",
            "title": "Example Proposal",
            "voting_ends": (datetime.utcnow() + timedelta(days=7)).isoformat()
        }
    
    async def _update_reputation(
        self,
        request: ReputationUpdateRequest,
        auth: HTTPAuthorizationCredentials
    ):
        """Update entity reputation"""
        
        # Only validators can update reputation
        # In production, check if auth entity is a validator
        
        update = ReputationUpdate(
            entity_id=request.entity_id,
            amount=request.amount,
            reason=request.reason,
            evidence=request.evidence,
            timestamp=datetime.utcnow()
        )
        
        result = await self.reputation.update_reputation(
            update,
            request.validators
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['reason'])
        
        return result
    
    async def _get_reputation(self, entity_id: str):
        """Get entity reputation"""
        
        # Get from Redis cache
        score = await self.redis_pool.zscore('reputation_scores', entity_id)
        
        if score is None:
            return {"entity_id": entity_id, "reputation": 0.0}
        
        return {
            "entity_id": entity_id,
            "reputation": float(score),
            "governance_weight": self.reputation.calculate_governance_weight(entity_id)
        }

# Application factory
def create_app(config_path: str) -> FastAPI:
    """Create and configure the application"""
    
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    # Create framework
    framework = AAMLFramework(config)
    
    # Lifespan manager
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        await framework.initialize()
        yield
        # Shutdown
        await framework.shutdown()
    
    framework.app.router.lifespan_context = lifespan
    
    return framework.app

# Run the application
if __name__ == "__main__":
    app = create_app("config/production.json")
    uvicorn.run(app, host="0.0.0.0", port=8000)
