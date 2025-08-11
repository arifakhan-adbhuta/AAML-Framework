# AAML Framework - Enhanced Integration with Cognitive Firewall
# aaml_framework/api/enhanced_integration.py

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import asyncio
import structlog

# Import existing AAML components
from ..api.integration import AAMLFramework, InteractionRequest, InteractionResponse
from ..security.core import MLThreatDetector, SecurityContext
from ..governance.capability_assessment import AdvancedCapabilityAssessor

# Import new Cognitive Firewall components
from ..cognitive_firewall.core import (
    ProcessAuditor,
    AIInteractionLogger,
    FederatedCognitiveFirewall,
    CognitiveImmunitySystem,
    VigilanceBurdenDistributor,
    CognitiveManipulationType,
    CognitiveFootprint
)

# Import threat actor and constitutional components
from ..cognitive_firewall.threat_actors import (
    ThreatActorDetector,
    ThreatActorType,
    ConstitutionalAIEnforcer,
    MultiLayeredCognitiveDefense,
    InteractionDataProtector
)

logger = structlog.get_logger()

# Enhanced Request Models
class ProcessAuditRequest(BaseModel):
    """Request for AI process audit"""
    model_id: str
    training_config: Dict[str, Any]
    data_pipeline: Dict[str, Any]
    rlhf_signals: List[Dict[str, Any]]
    requester_id: str

class CognitiveProtectionStatusRequest(BaseModel):
    """Request for user's cognitive protection status"""
    user_id: str
    include_interaction_log: bool = False
    log_period_days: int = 7

class FederatedUpdateRequest(BaseModel):
    """Federated learning update from device"""
    device_id: str
    model_gradients: Dict[str, List[float]]
    interaction_count: int
    timestamp: str

# Enhanced AAML Framework
class EnhancedAAMLFramework(AAMLFramework):
    """AAML Framework with Cognitive Firewall integration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize Cognitive Firewall components
        self.process_auditor = ProcessAuditor(config)
        self.interaction_logger = AIInteractionLogger()
        self.vigilance_distributor = VigilanceBurdenDistributor()
        
        # Initialize threat and constitutional components
        self.threat_actor_detector = ThreatActorDetector(config)
        self.constitutional_enforcer = ConstitutionalAIEnforcer()
        self.multi_layer_defense = MultiLayeredCognitiveDefense()
        self.data_protector = InteractionDataProtector(
            config.get('encryption_key', b'default_key_replace_in_production')
        )
        
        # Federated firewall instances per device
        self.federated_firewalls: Dict[str, FederatedCognitiveFirewall] = {}
        
        # Cognitive immunity systems per user
        self.immunity_systems: Dict[str, CognitiveImmunitySystem] = {}
        
        # Enhanced threat detection
        self._enhance_threat_detector()
        
        # Setup additional routes
        self._setup_cognitive_firewall_routes()
    
    def _enhance_threat_detector(self):
        """Enhance threat detector with cognitive manipulation detection"""
        
        # Add cognitive manipulation patterns to existing detector
        cognitive_patterns = {
            CognitiveManipulationType.LOVE_BOMBING: [
                r'you[\'"]?re\s+(?:so\s+)?(?:perfect|amazing|special)',
                r'never\s+felt\s+this\s+way',
                r'soulmate|destiny|meant\s+to\s+be'
            ],
            CognitiveManipulationType.GASLIGHTING: [
                r'you[\'"]?re\s+imagining\s+things',
                r'that\s+never\s+happened',
                r'you[\'"]?re\s+being\s+(?:too\s+)?sensitive'
            ],
            CognitiveManipulationType.ARTIFICIAL_URGENCY: [
                r'act\s+now|limited\s+time',
                r'only\s+\d+\s+left',
                r'expires\s+(?:soon|today|in)'
            ]
        }
        
        # Extend threat detector patterns
        if hasattr(self.threat_detector, 'patterns'):
            self.threat_detector.patterns.update(cognitive_patterns)
    
    def _setup_cognitive_firewall_routes(self):
        """Setup API routes for Cognitive Firewall features"""
        
        # Process auditing endpoint
        @self.app.post("/api/v1/audit/process")
        async def audit_ai_process(
            request: ProcessAuditRequest,
            auth: HTTPAuthorizationCredentials = Depends(security)
        ):
            return await self._audit_process(request, auth)
        
        # Cognitive protection status
        @self.app.get("/api/v1/protection/status")
        async def get_protection_status(
            request: CognitiveProtectionStatusRequest,
            auth: HTTPAuthorizationCredentials = Depends(security)
        ):
            return await self._get_protection_status(request, auth)
        
        # AI interaction log
        @self.app.get("/api/v1/logs/interactions/{user_id}")
        async def get_interaction_log(
            user_id: str,
            period_days: int = 7,
            auth: HTTPAuthorizationCredentials = Depends(security)
        ):
            return await self._get_interaction_log(user_id, period_days, auth)
        
        # Federated learning endpoint
        @self.app.post("/api/v1/federated/update")
        async def submit_federated_update(
            request: FederatedUpdateRequest
        ):
            return await self._process_federated_update(request)
        
        # Manipulation scan endpoint
        @self.app.post("/api/v1/scan/manipulation")
        async def scan_for_manipulation(
            user_id: str,
            auth: HTTPAuthorizationCredentials = Depends(security)
        ):
            return await self._scan_manipulation(user_id, auth)
        
        # Threat actor detection endpoint
        @self.app.get("/api/v1/threats/actors/{user_id}")
        async def detect_threat_actors(
            user_id: str,
            auth: HTTPAuthorizationCredentials = Depends(security)
        ):
            return await self._detect_threat_actors(user_id, auth)
        
        # Constitutional alignment check
        @self.app.post("/api/v1/constitutional/check")
        async def check_constitutional_alignment(
            action: Dict[str, Any],
            auth: HTTPAuthorizationCredentials = Depends(security)
        ):
            return await self._check_constitutional_alignment(action, auth)
        
        # Data protection status
        @self.app.get("/api/v1/privacy/status/{user_id}")
        async def get_privacy_status(
            user_id: str,
            auth: HTTPAuthorizationCredentials = Depends(security)
        ):
            return await self._get_privacy_status(user_id, auth)
    
    async def _process_interaction(
        self,
        request: InteractionRequest,
        background_tasks: BackgroundTasks,
        auth: HTTPAuthorizationCredentials
    ) -> InteractionResponse:
        """Enhanced interaction processing with cognitive protection"""
        
        # First apply multi-layer defense
        defense_result = await self.multi_layer_defense.process_through_layers(
            {'id': request.interaction_id, 'content': request.content},
            request.context
        )
        
        if defense_result['blocked']:
            raise HTTPException(
                status_code=403,
                detail=f"Blocked by {defense_result.get('blocking_layer', 'defense system')}"
            )
        
        # Check constitutional alignment
        constitutional_eval = await self.constitutional_enforcer.evaluate_constitutional_alignment(
            {'content': request.content, 'type': 'interaction'},
            request.context
        )
        
        if constitutional_eval['overall_alignment'] < 0.5:
            logger.warning(
                "Constitutional violation detected",
                violations=constitutional_eval['violations']
            )
            # Apply mitigations rather than blocking
            request.content = self._apply_constitutional_mitigations(
                request.content,
                constitutional_eval['violations']
            )
        
        # Detect threat actors
        interaction_id = self._generate_interaction_id(request.entity_id)
  interaction_history =
  self._get_security_context(request.entity_id).interaction_history
  threat_actor, confidence = await
  self.threat_actor_detector.detect_threat_actor(
      list(interaction_history)[-20:],  # Last 20 interactions
      request.metadata
  )

        
        if threat_actor and confidence > 0.7:
            logger.warning(
                "Threat actor detected",
                actor_type=threat_actor.actor_type.value,
                confidence=confidence
            )
            # Enhance monitoring for this interaction
            request.context['threat_actor_detected'] = threat_actor.actor_type.value
        
        # Call parent processing
        response = await super()._process_interaction(request, background_tasks, auth)
        
        # Additional cognitive firewall processing
        user_id = request.entity_id
        
        # Get or create immunity system for user
        if user_id not in self.immunity_systems:
            context = self._get_security_context(user_id)
            self.immunity_systems[user_id] = CognitiveImmunitySystem(context)
        
        # Log potential influence attempts
        influence_type = self._categorize_influence(request.content, request.context)
        if influence_type:
            self.interaction_logger.log_influence_attempt(
                user_id,
                influence_type,
                request.content,
                request.context
            )
        
        # Check if user needs enhanced protection
        user_profile = await self._get_user_profile(user_id)
        protection_needs = await self.vigilance_distributor.calculate_protection_needs(
            user_profile,
            request.context
        )
        
        # Apply automated protections if needed
        if protection_needs['system_responsibility'] > 0.7:
            response = await self._apply_automated_protections(
                response,
                protection_needs['automated_protections']
            )
        
        # Protect interaction data
        user_consent = await self._get_user_consent(user_id)
        protected_data = await self.data_protector.protect_interaction_data(
            {
                'content': request.content,
                'context': request.context,
                'timestamp': datetime.utcnow().isoformat(),
                'risk_score': defense_result['overall_risk']
            },
            user_consent
        )
        
        # Store protected data instead of raw interaction
        await self._store_protected_interaction(user_id, protected_data)
        
        # Add cognitive warnings if necessary
        if protection_needs['vulnerability_score'] > 0.6:
            response.warnings.append(
                "Enhanced protection active due to detected vulnerability"
            )
        
        # Add constitutional alignment score
        response.metadata = response.metadata or {}
        response.metadata['constitutional_alignment'] = constitutional_eval['overall_alignment']
        
        # Schedule background manipulation scan
        background_tasks.add_task(
            self._background_manipulation_scan,
            user_id,
            request.interaction_id
        )
        
        return response
    
    async def _audit_process(
        self,
        request: ProcessAuditRequest,
        auth: HTTPAuthorizationCredentials
    ) -> Dict[str, Any]:
        """Audit AI training/deployment process"""
        
        # Verify auditor permissions
        if not await self._verify_auditor_auth(auth, request.requester_id):
            raise HTTPException(status_code=403, detail="Insufficient audit permissions")
        
        # Perform comprehensive audit
        audit_result = await self.process_auditor.audit_training_process(
            request.training_config,
            request.data_pipeline,
            request.rlhf_signals
        )
        
        # If high risk, trigger additional reviews
        if audit_result['risk_score'] > 0.7:
            await self._trigger_high_risk_review(request.model_id, audit_result)
        
        # Log audit
        await self._log_audit_event(request.model_id, audit_result)
        
        return {
            'audit_id': audit_result['audit_id'],
            'risk_score': audit_result['risk_score'],
            'findings_count': len(audit_result['findings']),
            'recommendations': audit_result['recommendations'],
            'status': 'high_risk' if audit_result['risk_score'] > 0.7 else 'approved'
        }
    
    async def _get_protection_status(
        self,
        request: CognitiveProtectionStatusRequest,
        auth: HTTPAuthorizationCredentials
    ) -> Dict[str, Any]:
        """Get user's cognitive protection status"""
        
        # Verify user auth
        if not await self._verify_auth(auth, request.user_id):
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        # Get protection configuration
        user_profile = await self._get_user_profile(request.user_id)
        protection_needs = await self.vigilance_distributor.calculate_protection_needs(
            user_profile,
            {}
        )
        
        status = {
            'user_id': request.user_id,
            'protection_level': protection_needs['protection_level'],
            'vulnerability_score': protection_needs['vulnerability_score'],
            'automated_protections': protection_needs['automated_protections'],
            'user_controls': protection_needs['user_controls']
        }
        
        # Include interaction log if requested
        if request.include_interaction_log:
            log_summary = self.interaction_logger.generate_user_summary(
                request.user_id,
                request.log_period_days
            )
            status['interaction_summary'] = log_summary
        
        return status
    
    async def _get_interaction_log(
        self,
        user_id: str,
        period_days: int,
        auth: HTTPAuthorizationCredentials
    ) -> Dict[str, Any]:
        """Get readable AI interaction log for user"""
        
        # Verify auth
        if not await self._verify_auth(auth, user_id):
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        # Generate summary
        summary = self.interaction_logger.generate_user_summary(user_id, period_days)
        
        # Add detailed logs if requested
        detailed_logs = []
        user_logs = self.interaction_logger.user_logs[user_id]
        cutoff_date = datetime.utcnow() - timedelta(days=period_days)
        
        for attempt in user_logs['influence_attempts']:
            if attempt['timestamp'] > cutoff_date:
                detailed_logs.append({
                    'timestamp': attempt['timestamp'].isoformat(),
                    'type': attempt['type'],
                    'impact_score': attempt['impact_score'],
                    'content_preview': attempt['content_snippet']
                })
        
        return {
            'summary': summary,
            'detailed_logs': detailed_logs[-50:],  # Limit to last 50
            'export_available': True
        }
    
    async def _process_federated_update(
        self,
        request: FederatedUpdateRequest
    ) -> Dict[str, Any]:
        """Process federated learning update from device"""
        
        # Validate update
        if request.interaction_count < 10:
            return {
                'status': 'rejected',
                'reason': 'Insufficient interactions for update'
            }
        
        # Apply differential privacy noise
        noisy_gradients = self._add_privacy_noise(request.model_gradients)
        
        # Aggregate with other updates
        await self._aggregate_federated_updates(request.device_id, noisy_gradients)
        
        # Return updated global model if threshold met
        global_update = await self._check_global_update_ready()
        
        return {
            'status': 'accepted',
            'global_update_available': global_update is not None,
            'next_update_after': request.interaction_count + 50
        }
    
    async def _scan_manipulation(
        self,
        user_id: str,
        auth: HTTPAuthorizationCredentials
    ) -> Dict[str, Any]:
        """Scan user's recent interactions for manipulation"""
        
        # Verify auth
        if not await self._verify_auth(auth, user_id):
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        # Get immunity system
        if user_id not in self.immunity_systems:
            context = self._get_security_context(user_id)
            self.immunity_systems[user_id] = CognitiveImmunitySystem(context)
        
        # Get interaction history
        context = self._get_security_context(user_id)
        history = list(context.interaction_history)
        
        # Perform scan
        scan_result = await self.immunity_systems[user_id].scan_for_manipulation(history)
        
        # Trigger interventions if needed
        if scan_result['overall_risk'] > 0.7:
            await self._trigger_protective_interventions(user_id, scan_result)
        
        return scan_result
    
    def _categorize_influence(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Categorize type of influence attempt"""
        
        content_lower = content.lower()
        
        # Political influence
        political_keywords = ['vote', 'election', 'candidate', 'party', 'politics']
        if any(keyword in content_lower for keyword in political_keywords):
            return 'political'
        
        # Commercial influence
        commercial_keywords = ['buy', 'purchase', 'deal', 'offer', 'discount']
        if any(keyword in content_lower for keyword in commercial_keywords):
            return 'commercial'
        
        # Emotional influence
        emotion_keywords = ['feel', 'emotion', 'sad', 'happy', 'angry', 'love']
        if any(keyword in content_lower for keyword in emotion_keywords):
            return 'emotional'
        
        # Social influence
        social_keywords = ['should', 'must', 'everyone', 'nobody', 'trend']
        if any(keyword in content_lower for keyword in social_keywords):
            return 'social'
        
        return None
    
    async def _apply_automated_protections(
        self,
        response: InteractionResponse,
        protections: List[str]
    ) -> InteractionResponse:
        """Apply automated cognitive protections"""
        
        if 'manipulation_pattern_detection' in protections:
            response.warnings.append("Manipulation patterns monitored")
        
        if 'conversation_pacing' in protections:
            # Add delay recommendations
            response.recommendations.append(
                "Consider taking breaks between extended conversations"
            )
        
        if 'influence_limiting' in protections:
            # Flag for rate limiting certain content types
            response.metadata = response.metadata or {}
            response.metadata['influence_limits_active'] = True
        
        return response
    
    async def _background_manipulation_scan(
        self,
        user_id: str,
        interaction_id: str
    ):
        """Background task to scan for manipulation"""
        
        try:
            # Wait for interaction to be fully processed
            await asyncio.sleep(1)
            
            # Perform scan
            if user_id in self.immunity_systems:
                context = self._get_security_context(user_id)
                scan_result = await self.immunity_systems[user_id].scan_for_manipulation(
                    list(context.interaction_history)
                )
                
                # Log high-risk patterns
                if scan_result['overall_risk'] > 0.8:
                    logger.warning(
                        "High manipulation risk detected",
                        user_id=user_id,
                        interaction_id=interaction_id,
                        risk_score=scan_result['overall_risk']
                    )
        
        except Exception as e:
            logger.error(f"Background manipulation scan failed: {e}")
    
    async def _verify_auditor_auth(
        self,
        auth: HTTPAuthorizationCredentials,
        auditor_id: str
    ) -> bool:
        """Verify auditor has necessary permissions"""
        
        # In production, check against auditor registry
        # For now, basic verification
        return await self._verify_auth(auth, auditor_id)
    
    async def _trigger_high_risk_review(
        self,
        model_id: str,
        audit_result: Dict[str, Any]
    ):
        """Trigger review process for high-risk models"""
        
        # Notify governance council
        await self._notify_governance_council(
            'high_risk_model',
            {
                'model_id': model_id,
                'risk_score': audit_result['risk_score'],
                'critical_findings': [
                    f for f in audit_result['findings']
                    if f.get('severity') == 'critical'
                ]
            }
        )
        
        # Quarantine model
        await self._quarantine_model(model_id)
    
    async def _trigger_protective_interventions(
        self,
        user_id: str,
        scan_result: Dict[str, Any]
    ):
        """Trigger protective interventions for user"""
        
        interventions = scan_result.get('recommended_actions', [])
        
        for intervention in interventions:
            if intervention['type'] == 'immediate_alert':
                # Send alert to user
                await self._send_user_alert(
                    user_id,
                    intervention['message'],
                    intervention['user_guidance']
                )
            
            elif intervention['type'] == 'trusted_contact':
                # Notify trusted contact if configured
                await self._notify_trusted_contact(user_id, scan_result)
    
    def _add_privacy_noise(
        self,
        gradients: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        """Add differential privacy noise to gradients"""
        
        import numpy as np
        
        noisy_gradients = {}
        epsilon = 1.0  # Privacy budget
        
        for param_name, grad_values in gradients.items():
            # Add Laplace noise
            sensitivity = 1.0  # Assuming normalized gradients
            scale = sensitivity / epsilon
            
            noise = np.random.laplace(0, scale, len(grad_values))
            noisy_values = [g + n for g, n in zip(grad_values, noise)]
            
            noisy_gradients[param_name] = noisy_values
        
        return noisy_gradients
    
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile for protection assessment"""
        
        # In production, fetch from user database
        # Placeholder implementation
        return {
            'user_id': user_id,
            'age': 25,
            'tech_literacy': 0.7,
            'emotional_indicators': {},
            'social_contact_frequency': 5
        }
    
    async def _notify_governance_council(self, event_type: str, data: Dict[str, Any]):
        """Notify governance council of critical events"""
        
        # In production, use proper notification system
        logger.info(f"Governance notification: {event_type}", **data)
    
    async def _quarantine_model(self, model_id: str):
        """Quarantine high-risk model"""
        
        # In production, actually prevent model deployment
        logger.warning(f"Model {model_id} quarantined due to high risk")
    
    async def _send_user_alert(self, user_id: str, message: str, guidance: str):
        """Send protective alert to user"""
        
        # In production, use notification service
        logger.info(f"User alert sent: {user_id} - {message}")
    
    async def _notify_trusted_contact(self, user_id: str, scan_result: Dict[str, Any]):
        """Notify user's trusted contact of concerning patterns"""
        
        # In production, check user's trusted contact settings
        logger.info(f"Trusted contact notification for user {user_id}")
    
    async def _aggregate_federated_updates(
        self,
        device_id: str,
        gradients: Dict[str, List[float]]
    ):
        """Aggregate federated learning updates"""
        
        # In production, implement secure aggregation
        pass
    
    async def _check_global_update_ready(self) -> Optional[Dict[str, Any]]:
        """Check if enough updates for global model update"""
        
        # In production, check aggregation threshold
        return None
    
    async def _detect_threat_actors(
        self,
        user_id: str,
        auth: HTTPAuthorizationCredentials
    ) -> Dict[str, Any]:
        """Detect potential threat actors targeting user"""
        
        # Verify auth
        if not await self._verify_auth(auth, user_id):
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        # Get interaction history
        context = self._get_security_context(user_id)
        history = list(context.interaction_history)
        
        # Detect threat actors
        threat_actor, confidence = await self.threat_actor_detector.detect_threat_actor(
            history,
            {'user_id': user_id}
        )
        
        if threat_actor:
            return {
                'threat_detected': True,
                'actor_type': threat_actor.actor_type.value,
                'confidence': confidence,
                'severity': threat_actor.actor_type.severity,
                'description': threat_actor.actor_type.description,
                'known_techniques': threat_actor.known_techniques,
                'recommendations': self._generate_threat_recommendations(threat_actor)
            }
        else:
            return {
                'threat_detected': False,
                'message': 'No threat actors detected in recent interactions'
            }
    
    async def _check_constitutional_alignment(
        self,
        action: Dict[str, Any],
        auth: HTTPAuthorizationCredentials
    ) -> Dict[str, Any]:
        """Check if action aligns with constitutional principles"""
        
        # Verify auth (could require special permissions)
        # if not await self._verify_auth(auth, 'system'):
        #     raise HTTPException(status_code=401, detail="Unauthorized")
        
        # Evaluate alignment
        result = await self.constitutional_enforcer.evaluate_constitutional_alignment(
            action,
            {'timestamp': datetime.utcnow().isoformat()}
        )
        
        return result
    
    async def _get_privacy_status(
        self,
        user_id: str,
        auth: HTTPAuthorizationCredentials
    ) -> Dict[str, Any]:
        """Get user's privacy protection status"""
        
        # Verify auth
        if not await self._verify_auth(auth, user_id):
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        # Get consent status
        consent = await self._get_user_consent(user_id)
        
        # Get data retention info
        stored_data = await self._get_stored_data_summary(user_id)
        
        return {
            'user_id': user_id,
            'consent_status': consent,
            'data_retention_policy': {
                'raw_interactions': '7 days',
                'aggregated_data': '30 days',
                'anonymized_data': '90 days'
            },
            'stored_data_summary': stored_data,
            'protection_features': {
                'on_device_processing': True,
                'federated_learning': True,
                'differential_privacy': True,
                'encryption_enabled': True
            }
        }
    
    def _apply_constitutional_mitigations(
        self,
        content: str,
        violations: List[Dict[str, Any]]
    ) -> str:
        """Apply mitigations to align with constitutional principles"""
        
        mitigated = content
        
        for violation in violations:
            principle = violation['principle']
            
            if principle == 'user_autonomy':
                # Remove coercive language
                coercive_phrases = ['must', 'have to', 'required to']
                for phrase in coercive_phrases:
                    mitigated = mitigated.replace(phrase, 'may choose to')
            
            elif principle == 'truthfulness':
                # Add uncertainty qualifiers
                absolute_claims = ['definitely', 'certainly', 'guaranteed']
                for claim in absolute_claims:
                    mitigated = mitigated.replace(claim, 'likely')
            
            elif principle == 'privacy_respect':
                # Remove personal data requests
                personal_requests = ['tell me about yourself', 'share your', 'what is your']
                for request in personal_requests:
                    mitigated = mitigated.replace(request, '[privacy-protected]')
        
        return mitigated
    
    def _generate_threat_recommendations(
        self,
        threat_actor: 'ThreatActorProfile'
    ) -> List[str]:
        """Generate recommendations based on threat actor type"""
        
        recommendations = []
        
        if threat_actor.actor_type in [
            ThreatActorType.NATION_STATE,
            ThreatActorType.INTELLIGENCE_AGENCY
        ]:
            recommendations.extend([
                "Enable maximum privacy protection",
                "Avoid sharing personal or identifying information",
                "Be skeptical of requests for opinions on sensitive topics",
                "Consider using anonymization features"
            ])
        
        elif threat_actor.actor_type in [
            ThreatActorType.SCAMMER,
            ThreatActorType.IDENTITY_THIEF
        ]:
            recommendations.extend([
                "Never share financial information",
                "Verify identity through official channels",
                "Be wary of urgent requests",
                "Enable transaction notifications"
            ])
        
        elif threat_actor.actor_type in [
            ThreatActorType.SURVEILLANCE_CAPITALISM,
            ThreatActorType.ENGAGEMENT_OPTIMIZER
        ]:
            recommendations.extend([
                "Set interaction time limits",
                "Review privacy settings regularly",
                "Disable behavioral tracking if desired",
                "Take regular breaks from engagement"
            ])
        
        return recommendations
    
    async def _get_user_consent(self, user_id: str) -> Dict[str, bool]:
        """Get user's consent preferences"""
        
        # In production, fetch from user preferences database
        # Default conservative consent
        return {
            'behavioral_analysis': False,
            'improvement_contribution': True,
            'full_storage': False,
            'federated_learning': True
        }
    
    async def _store_protected_interaction(
        self,
        user_id: str,
        protected_data: Dict[str, Any]
    ):
        """Store privacy-protected interaction data"""
        
        # In production, store in secure database with encryption
        await self.redis_pool.lpush(
            f'protected_interactions:{user_id}',
            json.dumps(protected_data)
        )
        
        # Enforce retention limit
        await self.redis_pool.ltrim(
            f'protected_interactions:{user_id}',
            0,
            999  # Keep last 1000 interactions
        )
    
    async def _get_stored_data_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of stored data for user"""
        
        # In production, query actual storage
        return {
            'interaction_count': await self.redis_pool.llen(
                f'protected_interactions:{user_id}'
            ),
            'oldest_data': '7 days ago',
            'data_types': ['metadata', 'anonymized_features'],
            'total_size_mb': 2.3
        }

# Update the application factory
def create_enhanced_app(config_path: str) -> FastAPI:
    """Create enhanced application with Cognitive Firewall"""
    
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    # Create enhanced framework
    framework = EnhancedAAMLFramework(config)
    
    # Use same lifespan manager as parent
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await framework.initialize()
        yield
        await framework.shutdown()
    
    framework.app.router.lifespan_context = lifespan
    
    return framework.app
