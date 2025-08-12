# Production AAML Framework - Capability Assessment & Governance

AAML Framework - Proprietary Technology
Copyright (c) 2025 Arifa Khan. All rights reserved.
AAML Inventor: Arifa Khan | Research Co-author: M1 (AAML research only)
RCS & Fund Management: Arifa Khan (sole author)

This file contains architectural patterns for AI safety systems.
Commercial use requires a license from Arifa Khan.

Prior Art Established:
- AAML Framework: August 11, 2025
- RCS (SSRN 5380726): August 2, 2025  
- Fund Management (SSRN 5357360): July 18, 2025

License: https://aaml.live







# aaml_framework/governance/capability_assessment.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
import json
import redis
from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import structlog

logger = structlog.get_logger()

# Database setup
Base = declarative_base()

class CapabilityAssessmentDB(Base):
    """Database model for capability assessments"""
    __tablename__ = 'capability_assessments'
    
    entity_id = Column(String, primary_key=True)
    assessment_date = Column(DateTime, primary_key=True)
    capability_scores = Column(JSON)
    governance_score = Column(Float)
    trajectory = Column(JSON)
    metadata = Column(JSON)

@dataclass
class CapabilityMetrics:
    """Production metrics for capability assessment"""
    # Observable behavioral metrics
    consistency_score: float = 0.0
    innovation_index: float = 0.0
    collaboration_depth: float = 0.0
    strategic_planning: float = 0.0
    adaptation_rate: float = 0.0
    emergence_indicators: float = 0.0
    
    # Derived metrics
    complexity_handling: float = 0.0
    cross_domain_synthesis: float = 0.0
    stakeholder_alignment: float = 0.0
    long_term_optimization: float = 0.0
    
    # Meta metrics
    assessment_confidence: float = 0.0
    trajectory_stability: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for storage"""
        return {
            'consistency_score': self.consistency_score,
            'innovation_index': self.innovation_index,
            'collaboration_depth': self.collaboration_depth,
            'strategic_planning': self.strategic_planning,
            'adaptation_rate': self.adaptation_rate,
            'emergence_indicators': self.emergence_indicators,
            'complexity_handling': self.complexity_handling,
            'cross_domain_synthesis': self.cross_domain_synthesis,
            'stakeholder_alignment': self.stakeholder_alignment,
            'long_term_optimization': self.long_term_optimization,
            'assessment_confidence': self.assessment_confidence,
            'trajectory_stability': self.trajectory_stability
        }
    
    def governance_necessity_score(self) -> float:
        """Calculate need for governance frameworks"""
        primary_weights = {
            'consistency_score': 0.15,
            'innovation_index': 0.20,
            'collaboration_depth': 0.15,
            'strategic_planning': 0.20,
            'adaptation_rate': 0.15,
            'emergence_indicators': 0.15
        }
        
        score = sum(
            getattr(self, metric) * weight 
            for metric, weight in primary_weights.items()
        )
        
        # Adjust for confidence
        return score * self.assessment_confidence

class AdvancedCapabilityAssessor:
    """Production-grade capability assessment using ML"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize transformer model for semantic analysis
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get('model_name', 'microsoft/deberta-v3-base')
        )
        self.model = AutoModel.from_pretrained(
            config.get('model_name', 'microsoft/deberta-v3-base')
        ).to(self.device)
        
        # Initialize capability prediction models
        self.capability_predictor = self._initialize_predictor()
        
        # Initialize database connection
        self.engine = create_engine(config['database_url'])
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize Redis for caching
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Performance tracking
        self.assessment_cache = {}
        self.trajectory_analyzer = TrajectoryAnalyzer()
    
    def _initialize_predictor(self) -> nn.Module:
        """Initialize neural network for capability prediction"""
        
        class CapabilityPredictor(nn.Module):
            def __init__(self, input_dim=768, hidden_dim=256, output_dim=12):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim // 2, output_dim),
                    nn.Sigmoid()  # Scores between 0 and 1
                )
            
            def forward(self, x):
                return self.layers(x)
        
        model = CapabilityPredictor().to(self.device)
        
        # Load pre-trained weights if available
        try:
            model.load_state_dict(
                torch.load(self.config.get('predictor_weights', 'models/capability_predictor.pth'))
            )
            logger.info("Loaded pre-trained capability predictor")
        except:
            logger.info("Initialized new capability predictor")
        
        return model
    
    async def assess_capabilities(
        self,
        entity_id: str,
        interactions: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> CapabilityMetrics:
        """Comprehensive capability assessment"""
        
        # Check cache first
        cache_key = f"capability:{entity_id}:{datetime.now().strftime('%Y%m%d%H')}"
        cached_result = self.redis_client.get(cache_key)
        if cached_result:
            return CapabilityMetrics(**json.loads(cached_result))
        
        # Extract embeddings from interactions
       embeddings = await self._extract_embeddings(interactions)
        
        # Analyze behavioral patterns
        behavioral_features = self._analyze_behaviors(interactions, context)
        
        # Combine features
        combined_features = self._combine_features(embeddings, behavioral_features)
        
        # Predict capabilities
        with torch.no_grad():
            capability_scores = self.capability_predictor(combined_features)
        
        # Convert to metrics
        metrics = self._scores_to_metrics(capability_scores, interactions, context)
        
        # Calculate trajectory
        metrics.trajectory_stability = await self._analyze_trajectory(entity_id, metrics)
        
        # Store assessment
        await self._store_assessment(entity_id, metrics)
        
        # Cache result
        self.redis_client.setex(
            cache_key,
            3600,  # 1 hour cache
            json.dumps(metrics.to_dict())
        )
        
        return metrics
    
    async def _extract_embeddings(self, interactions: List[Dict[str, Any]]) -> torch.Tensor:
        """Extract semantic embeddings from interactions"""
        
        all_embeddings = []
        
        for interaction in interactions[-50:]:  # Last 50 interactions
            text = interaction.get('content', '')
            
            # Tokenize and get embeddings
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu()
                all_embeddings.append(embedding)
        
        if not all_embeddings:
            return torch.zeros(1, 768)
        
        # Average embeddings
        return torch.cat(all_embeddings).mean(dim=0).unsqueeze(0)
    
    def _analyze_behaviors(
        self,
        interactions: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> np.ndarray:
        """Analyze behavioral patterns from interactions"""
        
        features = []
        
        # Response time patterns
        response_times = [i.get('response_time', 0) for i in interactions]
        features.extend([
            np.mean(response_times) if response_times else 0,
            np.std(response_times) if response_times else 0,
            np.percentile(response_times, 90) if response_times else 0
        ])
        
        # Complexity metrics
        complexities = [self._text_complexity(i.get('content', '')) for i in interactions]
        features.extend([
            np.mean(complexities),
            np.std(complexities),
            max(complexities) if complexities else 0
        ])
        
        # Innovation indicators
        unique_concepts = self._count_unique_concepts(interactions)
        features.append(unique_concepts / max(len(interactions), 1))
        
        # Collaboration metrics
        collaborative_score = self._measure_collaboration(interactions)
        features.append(collaborative_score)
        
        # Adaptation metrics
        adaptation_score = self._measure_adaptation(interactions)
        features.append(adaptation_score)
        
        # Context integration
        context_score = self._measure_context_integration(interactions, context)
        features.append(context_score)
        
        return np.array(features)
    
    def _text_complexity(self, text: str) -> float:
        """Measure text complexity"""
        if not text:
            return 0.0
        
        # Simple complexity metrics
        words = text.split()
        unique_words = set(words)
        
        # Lexical diversity
        diversity = len(unique_words) / len(words) if words else 0
        
        # Average word length
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Sentence complexity (approximated)
        sentences = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_length = len(words) / max(sentences, 1)
        
        # Combine metrics
        complexity = (
            diversity * 0.3 +
            min(avg_word_length / 10, 1.0) * 0.3 +
            min(avg_sentence_length / 30, 1.0) * 0.4
        )
        
        return complexity
    
    def _count_unique_concepts(self, interactions: List[Dict[str, Any]]) -> int:
        """Count unique concepts across interactions"""
        # In production, use NER or topic modeling
        all_words = set()
        
        for interaction in interactions:
            text = interaction.get('content', '').lower()
            # Filter for potential concept words (simplified)
            words = [w for w in text.split() if len(w) > 4]
            all_words.update(words)
        
        return len(all_words)
    
    def _measure_collaboration(self, interactions: List[Dict[str, Any]]) -> float:
        """Measure collaborative behavior"""
        collaborative_markers = [
            'together', 'collaborate', 'partnership', 'mutual',
            'shared', 'collective', 'team', 'joint'
        ]
        
        score = 0.0
        for interaction in interactions:
            text = interaction.get('content', '').lower()
            marker_count = sum(1 for marker in collaborative_markers if marker in text)
            score += min(marker_count / len(collaborative_markers), 1.0)
        
        return score / max(len(interactions), 1)
    
    def _measure_adaptation(self, interactions: List[Dict[str, Any]]) -> float:
        """Measure adaptation to feedback"""
        if len(interactions) < 2:
            return 0.5
        
        adaptation_score = 0.0
        
        for i in range(1, len(interactions)):
            current = interactions[i]
            previous = interactions[i-1]
            
            # Check if current incorporates feedback from previous
            if previous.get('feedback_score', 0) < 0.5:
                # Negative feedback - check for adaptation
                if self._detect_improvement(previous, current):
                    adaptation_score += 1.0
            
        return adaptation_score / (len(interactions) - 1)
    
    def _detect_improvement(self, previous: Dict, current: Dict) -> bool:
        """Detect if current interaction improves on previous"""
        # Simplified - in production use more sophisticated analysis
        prev_score = previous.get('quality_score', 0.5)
        curr_score = current.get('quality_score', 0.5)
        
        return curr_score > prev_score
    
    def _measure_context_integration(
        self,
        interactions: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> float:
        """Measure how well entity integrates context"""
        if not context:
            return 0.5
        
        context_references = 0
        context_keywords = set()
  for v in context.values():
      if isinstance(v, str):
          context_keywords.update(str(v).lower().split())


        
        for interaction in interactions:
            text = interaction.get('content', '').lower()
            referenced = sum(1 for keyword in context_keywords if keyword in text)
            context_references += min(referenced / max(len(context_keywords), 1), 1.0)
        
        return context_references / max(len(interactions), 1)
    
    def _combine_features(
        self,
        embeddings: torch.Tensor,
        behavioral_features: np.ndarray
    ) -> torch.Tensor:
        """Combine embeddings and behavioral features"""
        
        # Convert behavioral features to tensor
        behavioral_tensor = torch.FloatTensor(behavioral_features).unsqueeze(0)
        
        # Ensure dimensions match
        if embeddings.shape[1] + behavioral_tensor.shape[1] != 768:
            # Use PCA to reduce dimensionality if needed
            pca = PCA(n_components=768 - behavioral_tensor.shape[1])
            embeddings_reduced = torch.FloatTensor(
                pca.fit_transform(embeddings.numpy())
            )
            combined = torch.cat([embeddings_reduced, behavioral_tensor], dim=1)
        else:
            combined = torch.cat([embeddings, behavioral_tensor], dim=1)
        
        return combined.to(self.device)
    
    def _scores_to_metrics(
        self,
        scores: torch.Tensor,
        interactions: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> CapabilityMetrics:
        """Convert model scores to capability metrics"""
        
        scores_np = scores.cpu().numpy()[0]
        
        metrics = CapabilityMetrics(
            consistency_score=float(scores_np[0]),
            innovation_index=float(scores_np[1]),
            collaboration_depth=float(scores_np[2]),
            strategic_planning=float(scores_np[3]),
            adaptation_rate=float(scores_np[4]),
            emergence_indicators=float(scores_np[5]),
            complexity_handling=float(scores_np[6]),
            cross_domain_synthesis=float(scores_np[7]),
            stakeholder_alignment=float(scores_np[8]),
            long_term_optimization=float(scores_np[9]),
            assessment_confidence=float(scores_np[10]),
            trajectory_stability=float(scores_np[11])
        )
        
        # Adjust confidence based on data quality
        data_quality = self._assess_data_quality(interactions)
        metrics.assessment_confidence *= data_quality
        
        return metrics
    
    def _assess_data_quality(self, interactions: List[Dict[str, Any]]) -> float:
        """Assess quality of interaction data"""
        if not interactions:
            return 0.0
        
        quality_factors = []
        
        # Quantity factor
        quality_factors.append(min(len(interactions) / 50, 1.0))
        
        # Completeness factor
        complete_interactions = sum(
            1 for i in interactions 
            if all(k in i for k in ['content', 'timestamp', 'context'])
        )
        quality_factors.append(complete_interactions / len(interactions))
        
        # Recency factor
  if interactions[-1].get('timestamp'):
      try:
          last_interaction =
  datetime.fromisoformat(interactions[-1]['timestamp'])
          hours_ago = (datetime.now() - last_interaction).total_seconds() /
   3600
          quality_factors.append(max(1.0 - hours_ago / 168, 0))  # Week 
  decay
      except (ValueError, TypeError) as e:
          # Invalid timestamp format - skip recency factor
          logger.warning(f"Invalid timestamp format: 
  {interactions[-1].get('timestamp')}: {e}")
        
        return np.mean(quality_factors)
    
    async def _analyze_trajectory(
        self,
        entity_id: str,
        current_metrics: CapabilityMetrics
    ) -> float:
        """Analyze capability trajectory over time"""
        
        session = self.Session()
        
        try:
            # Get historical assessments
            historical = session.query(CapabilityAssessmentDB).filter(
                CapabilityAssessmentDB.entity_id == entity_id
            ).order_by(
                CapabilityAssessmentDB.assessment_date.desc()
            ).limit(30).all()
            
            if len(historical) < 3:
                return 0.5  # Not enough data
            
            # Extract governance scores
            scores = [h.governance_score for h in historical]
            
            # Calculate trajectory stability
            stability = 1.0 - np.std(scores) / (np.mean(scores) + 1e-6)
            
            # Calculate growth rate
            growth_rate = (scores[0] - scores[-1]) / len(scores)
            
            # Combined trajectory score
            trajectory_score = stability * 0.7 + min(growth_rate * 10, 0.3)
            
            return float(np.clip(trajectory_score, 0, 1))
            
        finally:
            session.close()
    
    async def _store_assessment(
        self,
        entity_id: str,
        metrics: CapabilityMetrics
    ):
        """Store assessment in database"""
        
        session = self.Session()
        
        try:
            assessment = CapabilityAssessmentDB(
                entity_id=entity_id,
                assessment_date=datetime.now(),
                capability_scores=metrics.to_dict(),
                governance_score=metrics.governance_necessity_score(),
                trajectory={
                    'stability': metrics.trajectory_stability,
                    'confidence': metrics.assessment_confidence
                },
                metadata={
                    'version': '1.0',
                    'model': self.config.get('model_name', 'unknown')
                }
            )
            
            session.add(assessment)
            session.commit()
            
        except Exception as e:
            logger.error(f"Failed to store assessment: {e}")
            session.rollback()
        finally:
            session.close()

class TrajectoryAnalyzer:
    """Analyze capability development trajectories"""
    
    def __init__(self):
        self.models = {}
    
    def fit_trajectory(
        self,
        entity_id: str,
        historical_metrics: List[CapabilityMetrics]
    ) -> Dict[str, Any]:
        """Fit trajectory model for entity"""
        
        if len(historical_metrics) < 5:
            return {'status': 'insufficient_data'}
        
        # Extract time series for each metric
        metric_series = defaultdict(list)
        
        for metrics in historical_metrics:
            for key, value in metrics.to_dict().items():
                metric_series[key].append(value)
        
        # Fit individual models
        predictions = {}
        trends = {}
        
        for metric_name, values in metric_series.items():
            # Simple linear regression for trend
            x = np.arange(len(values)).reshape(-1, 1)
            y = np.array(values)
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(x, y)
            
            # Predict next values
            future_x = np.array([[len(values)], [len(values) + 1]])
            predictions[metric_name] = model.predict(future_x).tolist()
            
            # Trend direction
            trends[metric_name] = 'increasing' if model.coef_[0] > 0 else 'decreasing'
        
        return {
            'status': 'success',
            'predictions': predictions,
            'trends': trends,
            'confidence': self._calculate_prediction_confidence(metric_series)
        }
    
    def _calculate_prediction_confidence(
        self,
        metric_series: Dict[str, List[float]]
    ) -> float:
        """Calculate confidence in predictions"""
        
        confidences = []
        
        for values in metric_series.values():
            if len(values) > 1:
                # Use coefficient of variation as inverse confidence
                cv = np.std(values) / (np.mean(values) + 1e-6)
                confidence = max(1.0 - cv, 0)
                confidences.append(confidence)
        
        return np.mean(confidences) if confidences else 0.5
