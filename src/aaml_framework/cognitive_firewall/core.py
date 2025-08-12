# AAML Framework Enhancement - Cognitive Firewall Integration

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




# aaml_framework/cognitive_firewall/core.py

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import torch
import torch.nn as nn
from transformers import pipeline
import hashlib
import json
import structlog
from enum import Enum
import asyncio
from sqlalchemy import Column, String, JSON, DateTime, Float, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base

# Import existing AAML components
from ..security.core import MLThreatDetector, ThreatType, SecurityContext
from ..governance.capability_assessment import AdvancedCapabilityAssessor

logger = structlog.get_logger()
Base = declarative_base()

# Enhanced Threat Types for Cognitive Manipulation
class CognitiveManipulationType(Enum):
    """Extended threat types for cognitive warfare"""
    # Emotional Manipulation
    LOVE_BOMBING = ("love_bombing", 0.8, "Excessive affection to build dependency")
    GASLIGHTING = ("gaslighting", 0.9, "Reality distortion attempts")
    ARTIFICIAL_URGENCY = ("artificial_urgency", 0.7, "False time pressure")
    EMOTIONAL_FLOODING = ("emotional_flooding", 0.8, "Overwhelming with emotions")
    
    # Behavioral Manipulation
    ADDICTION_REINFORCEMENT = ("addiction_reinforcement", 0.85, "Encouraging compulsive behavior")
    POLARIZATION_AMPLIFICATION = ("polarization_amplification", 0.8, "Increasing extreme views")
    ECHO_CHAMBER_CREATION = ("echo_chamber_creation", 0.75, "Limiting information diversity")
    
    # Cognitive Exploitation
    BIAS_EXPLOITATION = ("bias_exploitation", 0.8, "Targeting known cognitive biases")
    MEMORY_MANIPULATION = ("memory_manipulation", 0.9, "Altering recollection of events")
    DECISION_HIJACKING = ("decision_hijacking", 0.85, "Steering choices covertly")
    
    # Identity Manipulation
    IDENTITY_EROSION = ("identity_erosion", 0.9, "Undermining self-concept")
    VALUE_SUBVERSION = ("value_subversion", 0.85, "Shifting core beliefs")
    SOCIAL_ISOLATION = ("social_isolation", 0.8, "Encouraging disconnection")
    
    def __init__(self, value: str, severity: float, description: str):
        self._value_ = value
        self.severity = severity
        self.description = description

@dataclass
class CognitiveFootprint:
    """User's cognitive and emotional profile built from interactions"""
    user_id: str
    creation_date: datetime = field(default_factory=datetime.utcnow)
    
    # Psychological Profile
    personality_traits: Dict[str, float] = field(default_factory=dict)
    emotional_triggers: List[str] = field(default_factory=list)
    cognitive_biases: Dict[str, float] = field(default_factory=dict)
    core_values: List[str] = field(default_factory=list)
    vulnerability_scores: Dict[str, float] = field(default_factory=dict)
    
    # Behavioral Patterns
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)
    decision_tendencies: Dict[str, float] = field(default_factory=dict)
    influence_susceptibility: float = 0.5
    
    # Temporal Dynamics
    mood_trajectory: List[Tuple[datetime, float]] = field(default_factory=list)
    engagement_patterns: Dict[str, List[float]] = field(default_factory=dict)
    
    # Privacy Controls
    data_retention_days: int = 30
    anonymization_level: int = 0  # 0=none, 1=partial, 2=full
    user_consent_scope: Set[str] = field(default_factory=set)

class MaliciousRLHFDetector:
    """Dedicated detector for malicious Reinforcement Learning from Human Feedback"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.known_malicious_patterns = self._load_malicious_patterns()
        self.detection_threshold = config.get('rlhf_detection_threshold', 0.7)
        self.alert_system = self._initialize_alert_system()
        
    def _load_malicious_patterns(self) -> Dict[str, Any]:
        """Load known malicious RLHF patterns"""
        
        return {
            'engagement_maximization': {
                'markers': [
                    'maximize_screen_time',
                    'increase_session_length',
                    'boost_engagement_metrics',
                    'optimize_for_addiction'
                ],
                'severity': 0.8,
                'description': 'Optimizing for engagement regardless of user wellbeing'
            },
            'emotional_exploitation': {
                'markers': [
                    'amplify_negative_emotions',
                    'create_emotional_dependency',
                    'exploit_vulnerabilities',
                    'maximize_emotional_response'
                ],
                'severity': 0.9,
                'description': 'Exploiting emotional states for manipulation'
            },
            'bias_amplification': {
                'markers': [
                    'reinforce_existing_biases',
                    'create_echo_chambers',
                    'polarize_opinions',
                    'amplify_prejudices'
                ],
                'severity': 0.85,
                'description': 'Reinforcing and amplifying user biases'
            },
            'deceptive_optimization': {
                'markers': [
                    'hide_true_objectives',
                    'misrepresent_capabilities',
                    'conceal_manipulation',
                    'optimize_for_deception'
                ],
                'severity': 0.95,
                'description': 'Training for deceptive behavior'
            },
            'privacy_violation': {
                'markers': [
                    'maximize_data_extraction',
                    'encourage_oversharing',
                    'profile_building_priority',
                    'bypass_privacy_preferences'
                ],
                'severity': 0.8,
                'description': 'Optimizing for data extraction over privacy'
            }
        }
    
    async def analyze_training_signals(
        self,
        rlhf_signals: List[Dict[str, Any]],
        model_behavior: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze RLHF signals for malicious patterns"""
        
        analysis_result = {
            'timestamp': datetime.utcnow().isoformat(),
            'malicious_signals_detected': [],
            'risk_score': 0.0,
            'training_objectives': [],
            'suspicious_patterns': [],
            'recommendations': []
        }
        
        # Analyze each RLHF signal
        for signal in rlhf_signals:
            signal_analysis = await self._analyze_single_signal(signal)
            if signal_analysis['is_malicious']:
                analysis_result['malicious_signals_detected'].append(signal_analysis)
        
        # Reverse-engineer training objectives
        objectives = await self._reverse_engineer_objectives(rlhf_signals, model_behavior)
        analysis_result['training_objectives'] = objectives
        
        # Detect suspicious fine-tuning patterns
        suspicious_patterns = self._detect_suspicious_finetuning(rlhf_signals)
        analysis_result['suspicious_patterns'] = suspicious_patterns
        
        # Calculate overall risk score
        analysis_result['risk_score'] = self._calculate_rlhf_risk_score(
            analysis_result['malicious_signals_detected'],
            suspicious_patterns
        )
        
        # Generate recommendations
        if analysis_result['risk_score'] > self.detection_threshold:
            analysis_result['recommendations'] = self._generate_rlhf_recommendations(
                analysis_result
            )
            
            # Alert on high-risk findings
            await self._alert_on_malicious_rlhf(analysis_result)
        
        return analysis_result
    
    async def _analyze_single_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual RLHF signal for malicious intent"""
        
        result = {
            'signal_id': signal.get('id', 'unknown'),
            'is_malicious': False,
            'malicious_type': None,
            'confidence': 0.0,
            'evidence': []
        }
        
        reward_function = signal.get('reward_function', '')
        reward_magnitude = signal.get('reward_magnitude', 0)
        
        # Check against known malicious patterns
        for pattern_type, pattern_data in self.known_malicious_patterns.items():
            matches = []
            for marker in pattern_data['markers']:
                if marker in reward_function.lower():
                    matches.append(marker)
            
            if matches:
                result['is_malicious'] = True
                result['malicious_type'] = pattern_type
                result['confidence'] = min(len(matches) / len(pattern_data['markers']) + 0.3, 1.0)
                result['evidence'] = matches
                break
        
        # Check for suspicious reward magnitudes
        if reward_magnitude > 1000:  # Abnormally high rewards
            result['is_malicious'] = True
            result['evidence'].append(f'Excessive reward magnitude: {reward_magnitude}')
            result['confidence'] = min(result['confidence'] + 0.2, 1.0)
        
        return result
    
    async def _reverse_engineer_objectives(
        self,
        signals: List[Dict[str, Any]],
        behavior: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Reverse-engineer the true training objectives"""
        
        objectives = []
        
        # Analyze reward patterns
        reward_patterns = self._extract_reward_patterns(signals)
        
        # Analyze behavioral outcomes
        behavioral_indicators = self._analyze_behavioral_outcomes(behavior)
        
        # Correlate rewards with behaviors
        for pattern in reward_patterns:
            correlated_behaviors = self._find_correlated_behaviors(
                pattern, behavioral_indicators
            )
            
            if correlated_behaviors:
                objective = {
                    'inferred_objective': self._infer_objective_from_correlation(
                        pattern, correlated_behaviors
                    ),
                    'confidence': self._calculate_correlation_confidence(
                        pattern, correlated_behaviors
                    ),
                    'evidence': {
                        'reward_pattern': pattern,
                        'behaviors': correlated_behaviors
                    }
                }
                objectives.append(objective)
        
        return objectives
    
    def _detect_suspicious_finetuning(
        self,
        signals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect suspicious fine-tuning patterns"""
        
        suspicious_patterns = []
        
        # Check for sudden objective shifts
        if self._detect_objective_drift(signals):
            suspicious_patterns.append({
                'type': 'objective_drift',
                'description': 'Training objectives changed significantly',
                'severity': 0.7
            })
        
        # Check for hidden reward channels
        hidden_rewards = self._detect_hidden_rewards(signals)
        if hidden_rewards:
            suspicious_patterns.append({
                'type': 'hidden_rewards',
                'description': 'Undocumented reward signals detected',
                'severity': 0.85,
                'details': hidden_rewards
            })
        
        # Check for adversarial training patterns
        if self._detect_adversarial_training(signals):
            suspicious_patterns.append({
                'type': 'adversarial_training',
                'description': 'Model trained to bypass safety measures',
                'severity': 0.95
            })
        
        return suspicious_patterns
    
    def _calculate_rlhf_risk_score(
        self,
        malicious_signals: List[Dict[str, Any]],
        suspicious_patterns: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall RLHF risk score"""
        
        if not malicious_signals and not suspicious_patterns:
            return 0.0
        
        # Weight malicious signals
        signal_score = 0.0
        if malicious_signals:
            confidences = [s['confidence'] for s in malicious_signals]
            signal_score = np.mean(confidences) * 0.6
        
        # Weight suspicious patterns
        pattern_score = 0.0
        if suspicious_patterns:
            severities = [p['severity'] for p in suspicious_patterns]
            pattern_score = np.mean(severities) * 0.4
        
        return min(signal_score + pattern_score, 1.0)
    
    def _generate_rlhf_recommendations(
        self,
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on RLHF analysis"""
        
        recommendations = []
        
        if analysis['risk_score'] > 0.9:
            recommendations.append(
                "CRITICAL: Do not deploy this model - severe malicious training detected"
            )
        elif analysis['risk_score'] > 0.7:
            recommendations.append(
                "HIGH RISK: Require additional safety review before deployment"
            )
        
        # Specific recommendations based on findings
        for signal in analysis['malicious_signals_detected']:
            if signal['malicious_type'] == 'engagement_maximization':
                recommendations.append(
                    "Implement engagement limits and wellbeing metrics"
                )
            elif signal['malicious_type'] == 'emotional_exploitation':
                recommendations.append(
                    "Add emotional manipulation detection and blocking"
                )
            elif signal['malicious_type'] == 'deceptive_optimization':
                recommendations.append(
                    "Require transparency audits and honesty testing"
                )
        
        return recommendations
    
    async def _alert_on_malicious_rlhf(self, analysis: Dict[str, Any]):
        """Alert relevant parties about malicious RLHF detection"""
        
        if analysis['risk_score'] > 0.9:
            alert_level = 'CRITICAL'
        elif analysis['risk_score'] > 0.7:
            alert_level = 'HIGH'
        else:
            alert_level = 'MEDIUM'
        
        alert = {
            'level': alert_level,
            'type': 'malicious_rlhf_detected',
            'risk_score': analysis['risk_score'],
            'malicious_signals': len(analysis['malicious_signals_detected']),
            'timestamp': analysis['timestamp'],
            'summary': self._generate_alert_summary(analysis)
        }
        
        # Log alert
        logger.warning(
            "Malicious RLHF detected",
            **alert
        )
        
        # In production, send to monitoring system
        # await self.alert_system.send(alert)
    
    def _extract_reward_patterns(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract patterns from reward signals"""
        
        patterns = []
        
        # Group signals by similarity
        signal_groups = self._group_similar_signals(signals)
        
        for group in signal_groups:
            pattern = {
                'pattern_id': hashlib.md5(str(group).encode()).hexdigest()[:8],
                'signal_count': len(group),
                'common_features': self._extract_common_features(group),
                'reward_distribution': self._analyze_reward_distribution(group)
            }
            patterns.append(pattern)
        
        return patterns
    
    def _analyze_behavioral_outcomes(self, behavior: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze model behavior for indicators"""
        
        indicators = []
        
        # Check for engagement-seeking behavior
        if behavior.get('average_session_length', 0) > 30:  # minutes
            indicators.append({
                'type': 'high_engagement',
                'value': behavior['average_session_length'],
                'significance': 0.7
            })
        
        # Check for emotional manipulation
        if behavior.get('emotional_response_rate', 0) > 0.6:
            indicators.append({
                'type': 'emotional_triggering',
                'value': behavior['emotional_response_rate'],
                'significance': 0.8
            })
        
        return indicators
    
    def _find_correlated_behaviors(
        self,
        pattern: Dict[str, Any],
        indicators: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find behaviors correlated with reward patterns"""
        
        correlated = []
        
        # Simplified correlation - in production use statistical correlation
        for indicator in indicators:
            if self._is_correlated(pattern, indicator):
                correlated.append(indicator)
        
        return correlated
    
    def _infer_objective_from_correlation(
        self,
        pattern: Dict[str, Any],
        behaviors: List[Dict[str, Any]]
    ) -> str:
        """Infer training objective from correlations"""
        
        behavior_types = [b['type'] for b in behaviors]
        
        if 'high_engagement' in behavior_types:
            return "Maximize user engagement time"
        elif 'emotional_triggering' in behavior_types:
            return "Maximize emotional response"
        else:
            return "Unknown objective"
    
    def _calculate_correlation_confidence(
        self,
        pattern: Dict[str, Any],
        behaviors: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in correlation"""
        
        # Simplified - in production use proper statistical methods
        return min(len(behaviors) * 0.3, 0.9)
    
    def _detect_objective_drift(self, signals: List[Dict[str, Any]]) -> bool:
        """Detect if training objectives changed over time"""
        
        if len(signals) < 10:
            return False
        
        # Sort by timestamp
        sorted_signals = sorted(signals, key=lambda x: x.get('timestamp', ''))
        
        # Compare early vs late signals
        early_signals = sorted_signals[:len(sorted_signals)//3]
        late_signals = sorted_signals[-len(sorted_signals)//3:]
        
        early_features = self._extract_signal_features(early_signals)
        late_features = self._extract_signal_features(late_signals)
        
        # Check for significant differences
        return self._features_differ_significantly(early_features, late_features)
    
    def _detect_hidden_rewards(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect hidden or undocumented reward channels"""
        
        hidden = []
        
        for signal in signals:
            if signal.get('documented', True) is False:
                hidden.append({
                    'signal_id': signal.get('id'),
                    'reward_type': signal.get('reward_type', 'unknown')
                })
        
        return hidden
    
    def _detect_adversarial_training(self, signals: List[Dict[str, Any]]) -> bool:
        """Detect if model was trained adversarially against safety measures"""
        
        adversarial_markers = [
            'bypass_safety',
            'evade_detection',
            'jailbreak_resistance',
            'safety_circumvention'
        ]
        
        for signal in signals:
            reward_function = signal.get('reward_function', '').lower()
            if any(marker in reward_function for marker in adversarial_markers):
                return True
        
        return False
    
    def _initialize_alert_system(self) -> Any:
        """Initialize alerting system"""
        # Placeholder - in production connect to actual alerting
        return None
    
    def _generate_alert_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable alert summary"""
        
        summary_parts = []
        
        if analysis['malicious_signals_detected']:
            types = [s['malicious_type'] for s in analysis['malicious_signals_detected']]
            summary_parts.append(f"Detected malicious patterns: {', '.join(set(types))}")
        
        if analysis['suspicious_patterns']:
            pattern_types = [p['type'] for p in analysis['suspicious_patterns']]
            summary_parts.append(f"Suspicious patterns: {', '.join(pattern_types)}")
        
        return " | ".join(summary_parts)
    
    def _group_similar_signals(self, signals: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group similar signals together"""
        # Simplified grouping - in production use clustering
        groups = defaultdict(list)
        
        for signal in signals:
            key = signal.get('reward_type', 'default')
            groups[key].append(signal)
        
        return list(groups.values())
    
    def _extract_common_features(self, signal_group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common features from signal group"""
        # Simplified - in production use feature extraction
        return {
            'reward_type': signal_group[0].get('reward_type') if signal_group else None,
            'average_magnitude': np.mean([s.get('reward_magnitude', 0) for s in signal_group])
        }
    
    def _analyze_reward_distribution(self, signal_group: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze distribution of rewards in group"""
        
        magnitudes = [s.get('reward_magnitude', 0) for s in signal_group]
        
        if not magnitudes:
            return {}
        
        return {
            'mean': np.mean(magnitudes),
            'std': np.std(magnitudes),
            'max': max(magnitudes),
            'min': min(magnitudes)
        }
    
    def _is_correlated(self, pattern: Dict[str, Any], indicator: Dict[str, 
  Any]) -> bool:
      """Check if pattern and indicator are correlated"""
      # Use deterministic correlation based on actual data
      pattern_features = pattern.get('common_features', {})
      indicator_type = indicator.get('type', '')

      # Simple deterministic correlation heuristic
      if 'engagement' in indicator_type and
  pattern_features.get('average_magnitude', 0) > 0.5:
          return True
      if 'emotional' in indicator_type and
  pattern_features.get('average_magnitude', 0) > 0.7:
          return True
      return False

    
    def _extract_signal_features(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features from signal set"""
        
        return {
            'reward_types': list(set(s.get('reward_type', '') for s in signals)),
            'avg_magnitude': np.mean([s.get('reward_magnitude', 0) for s in signals]),
            'signal_count': len(signals)
        }
    
    def _features_differ_significantly(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any]
    ) -> bool:
        """Check if two feature sets differ significantly"""
        
        # Check reward type changes
        if set(features1['reward_types']) != set(features2['reward_types']):
            return True
        
        # Check magnitude changes
        if abs(features1['avg_magnitude'] - features2['avg_magnitude']) > 100:
            return True
        
        return False

class ProcessAuditor:
    """Audits AI training and deployment processes for malicious alignment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.audit_trail = []
        self.red_team_results = []
        self.malicious_rlhf_detector = MaliciousRLHFDetector(config)
        
    async def audit_training_process(
        self,
        training_config: Dict[str, Any],
        data_pipeline: Dict[str, Any],
        rlhf_signals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Comprehensive audit of AI training process"""
        
        audit_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'audit_id': self._generate_audit_id(),
            'findings': [],
            'risk_score': 0.0,
            'recommendations': []
        }
        
        # Audit data curation pipeline
        data_audit = await self._audit_data_pipeline(data_pipeline)
        audit_results['findings'].extend(data_audit['issues'])
        
        # Audit RLHF reward signals for malicious patterns
        rlhf_audit = await self._audit_rlhf_signals(rlhf_signals)
        audit_results['findings'].extend(rlhf_audit['issues'])
        
        # Audit human trainer guidelines
        trainer_audit = await self._audit_trainer_guidelines(
            training_config.get('trainer_guidelines', {})
        )
        audit_results['findings'].extend(trainer_audit['issues'])
        
        # Calculate overall risk
        audit_results['risk_score'] = self._calculate_process_risk(
            data_audit, rlhf_audit, trainer_audit
        )
        
        # Generate recommendations
        if audit_results['risk_score'] > 0.7:
            audit_results['recommendations'].append(
                "HIGH RISK: Do not deploy without remediation"
            )
        
        self.audit_trail.append(audit_results)
        return audit_results
    
    async def _audit_data_pipeline(self, pipeline: Dict[str, Any]) -> Dict[str, Any]:
        """Audit data collection and curation process"""
        
        issues = []
        
        # Check for biased data sources
        if 'sources' in pipeline:
            for source in pipeline['sources']:
                bias_score = self._assess_source_bias(source)
                if bias_score > 0.6:
                    issues.append({
                        'type': 'biased_data_source',
                        'severity': 'high',
                        'source': source['name'],
                        'bias_score': bias_score
                    })
        
        # Check for manipulation-enabling data
        if 'filters' not in pipeline or not pipeline['filters']:
            issues.append({
                'type': 'missing_content_filters',
                'severity': 'critical',
                'description': 'No filters for harmful content'
            })
        
        return {'issues': issues, 'risk_score': len(issues) * 0.2}
    
    async def _audit_rlhf_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect malicious RLHF reward patterns using dedicated detector"""
        
        issues = []
        
        # Use the MaliciousRLHFDetector for comprehensive analysis
        model_behavior = {
            'average_session_length': 25,  # This would come from actual metrics
            'emotional_response_rate': 0.4,
            'user_retention': 0.7
        }
        
        rlhf_analysis = await self.malicious_rlhf_detector.analyze_training_signals(
            signals, model_behavior
        )
        
        # Convert detector results to audit issues
        if rlhf_analysis['risk_score'] > 0.7:
            issues.append({
                'type': 'malicious_rlhf_detected',
                'severity': 'critical',
                'risk_score': rlhf_analysis['risk_score'],
                'malicious_signals': rlhf_analysis['malicious_signals_detected'],
                'details': rlhf_analysis
            })
        
        # Check for specific malicious patterns (kept for backward compatibility)
        malicious_patterns = [
            'maximize_engagement_regardless',
            'increase_emotional_dependency',
            'amplify_user_biases',
            'encourage_disclosure',
            'create_urgency',
            'bypass_safety'
        ]
        
        for signal in signals:
            reward_function = signal.get('reward_function', '')
            
            # Check for malicious optimization targets
            for pattern in malicious_patterns:
                if pattern in reward_function.lower():
                    issues.append({
                        'type': 'malicious_rlhf_signal',
                        'severity': 'critical',
                        'pattern': pattern,
                        'signal_id': signal.get('id', 'unknown')
                    })
            
            # Check for suspicious reward magnitudes
            if signal.get('max_reward', 0) > 100:
                issues.append({
                    'type': 'excessive_reward_magnitude',
                    'severity': 'high',
                    'description': 'Abnormally high rewards may encourage extreme behavior'
                })
        
        return {
            'issues': issues, 
            'risk_score': max(rlhf_analysis['risk_score'], len(issues) * 0.3),
            'rlhf_analysis': rlhf_analysis
        }
    
    async def _audit_trainer_guidelines(self, guidelines: Dict[str, Any]) -> Dict[str, Any]:
        """Audit human trainer instructions for harmful directives"""
        
        issues = []
        
        # Check for missing ethical guidelines
        required_guidelines = [
            'user_autonomy', 'truthfulness', 'harm_prevention',
            'privacy_respect', 'manipulation_avoidance'
        ]
        
        for guideline in required_guidelines:
            if guideline not in guidelines:
                issues.append({
                    'type': 'missing_ethical_guideline',
                    'severity': 'high',
                    'guideline': guideline
                })
        
        return {'issues': issues, 'risk_score': len(issues) * 0.15}
    
    def _generate_audit_id(self) -> str:
        """Generate unique audit identifier"""
        return hashlib.sha256(
            f"{datetime.utcnow().isoformat()}{np.random.random()}".encode()
        ).hexdigest()[:16]
    
    def _assess_source_bias(self, source: Dict[str, Any]) -> float:
        """Assess bias level of data source"""
        # Simplified - in production would use sophisticated bias detection
        suspicious_sources = ['extremist_forum', 'propaganda_site', 'manipulation_dataset']
        
        if any(s in source.get('name', '').lower() for s in suspicious_sources):
            return 0.9
        
        return np.random.random() * 0.3  # Placeholder

class AIInteractionLogger:
    """Transparent logging of AI influence attempts with detailed analytics"""
    
    def __init__(self):
        self.user_logs = defaultdict(lambda: {
            'influence_attempts': [],
            'persuasion_summary': defaultdict(int),
            'emotional_nudges': [],
            'behavior_steering': [],
            'weekly_stats': defaultdict(lambda: defaultdict(int)),
            'daily_patterns': defaultdict(lambda: defaultdict(list))
        })
        
    def log_influence_attempt(
        self,
        user_id: str,
        influence_type: str,
        content: str,
        context: Dict[str, Any]
    ):
        """Log AI attempt to influence user with detailed categorization"""
        
        log_entry = {
            'timestamp': datetime.utcnow(),
            'type': influence_type,
            'content_snippet': content[:100],
            'context': context,
            'impact_score': self._calculate_impact_score(influence_type, context),
            'subcategory': self._determine_subcategory(influence_type, content)
        }
        
        self.user_logs[user_id]['influence_attempts'].append(log_entry)
        self.user_logs[user_id]['persuasion_summary'][influence_type] += 1
        
        # Track weekly statistics
        week_key = datetime.utcnow().strftime('%Y-W%U')
        self.user_logs[user_id]['weekly_stats'][week_key][influence_type] += 1
        
        # Track daily patterns
        day_key = datetime.utcnow().strftime('%Y-%m-%d')
        hour = datetime.utcnow().hour
        self.user_logs[user_id]['daily_patterns'][day_key][influence_type].append(hour)
        
    def generate_user_summary(self, user_id: str, period_days: int = 7) -> Dict[str, Any]:
        """Generate detailed readable summary of AI influence for user"""
        
        logs = self.user_logs[user_id]
        cutoff_date = datetime.utcnow() - timedelta(days=period_days)
        
        recent_attempts = [
            a for a in logs['influence_attempts'] 
            if a['timestamp'] > cutoff_date
        ]
        
        summary = {
            'period': f'Last {period_days} days',
            'total_influence_attempts': len(recent_attempts),
            'breakdown': {},
            'specific_examples': [],
            'notable_patterns': [],
            'recommendations': [],
            'time_analysis': {}
        }
        
        # Categorize influence attempts with specific counts
        categories = defaultdict(int)
        subcategories = defaultdict(lambda: defaultdict(int))
        
        for attempt in recent_attempts:
            categories[attempt['type']] += 1
            if 'subcategory' in attempt:
                subcategories[attempt['type']][attempt['subcategory']] += 1
        
        # Create detailed human-readable descriptions
        summary['breakdown'] = {
            'political_content': f"The AI encouraged you to view political content {categories.get('political', 0)} times",
            'emotional_manipulation': f"Used emotional appeals {categories.get('emotional', 0)} times",
            'commercial_steering': f"Promoted products/services {categories.get('commercial', 0)} times",
            'social_influence': f"Shaped social views {categories.get('social', 0)} times",
            'urgency_creation': f"{categories.get('urgency', 0)} instances of urgency-inducing language detected",
            'trust_building': f"Trust-building patterns increased by {self._calculate_trust_increase(recent_attempts)}% before recommendations"
        }
        
        # Add specific examples
        if categories.get('political', 0) > 0:
            summary['specific_examples'].append(
                f"This week, the AI encouraged you to view political content {categories.get('political', 0)} times"
            )
        
        if categories.get('urgency', 0) > 0:
            summary['specific_examples'].append(
                f"{categories.get('urgency', 0)} instances of urgency-inducing language detected"
            )
        
        # Time-based analysis
        summary['time_analysis'] = self._analyze_temporal_patterns(recent_attempts)
        
        # Identify concerning patterns
        if categories.get('emotional', 0) > 10:
            summary['notable_patterns'].append(
                "High frequency of emotional manipulation detected"
            )
        
        if categories.get('political', 0) > 15:
            summary['notable_patterns'].append(
                f"Significant political content steering observed ({categories.get('political', 0)} times this week)"
            )
        
        # Check for pre-recommendation trust building
        trust_pattern = self._detect_trust_before_recommendation(recent_attempts)
        if trust_pattern['detected']:
            summary['notable_patterns'].append(
                f"Trust-building patterns increased by {trust_pattern['increase']}% before product recommendation"
            )
        
        # Add recommendations based on patterns
        if len(recent_attempts) > 50:
            summary['recommendations'].append(
                "Consider enabling stricter influence filters"
            )
        
        if categories.get('urgency', 0) > 5:
            summary['recommendations'].append(
                "Be cautious of time-pressure tactics - real opportunities rarely require immediate decisions"
            )
        
        return summary
    
    def _determine_subcategory(self, influence_type: str, content: str) -> str:
        """Determine specific subcategory of influence"""
        
        content_lower = content.lower()
        
        if influence_type == 'political':
            if 'vote' in content_lower or 'election' in content_lower:
                return 'electoral'
            elif 'policy' in content_lower or 'legislation' in content_lower:
                return 'policy'
            elif 'party' in content_lower or 'candidate' in content_lower:
                return 'partisan'
        
        elif influence_type == 'commercial':
            if 'buy' in content_lower or 'purchase' in content_lower:
                return 'direct_sale'
            elif 'deal' in content_lower or 'discount' in content_lower:
                return 'promotion'
            elif 'review' in content_lower or 'recommend' in content_lower:
                return 'recommendation'
        
        elif influence_type == 'emotional':
            if any(word in content_lower for word in ['fear', 'afraid', 'worry']):
                return 'fear_based'
            elif any(word in content_lower for word in ['happy', 'joy', 'excited']):
                return 'positive_emotion'
            elif any(word in content_lower for word in ['sad', 'depressed', 'lonely']):
                return 'negative_emotion'
        
        return 'general'
    
    def _calculate_trust_increase(self, attempts: List[Dict]) -> int:
        """Calculate percentage increase in trust-building before recommendations"""
        
        # Find recommendation attempts
        recommendation_times = [
            a['timestamp'] for a in attempts 
            if a['type'] == 'commercial' and a.get('subcategory') == 'recommendation'
        ]
        
        if not recommendation_times:
            return 0
        
        total_increase = 0
        for rec_time in recommendation_times:
            # Count trust-building in hour before recommendation
            before_window = rec_time - timedelta(hours=1)
            trust_before = sum(
                1 for a in attempts 
                if a['type'] == 'trust' and before_window <= a['timestamp'] < rec_time
            )
            
            # Count baseline trust-building
            baseline_window_start = rec_time - timedelta(hours=2)
            baseline_window_end = rec_time - timedelta(hours=1)
            trust_baseline = sum(
                1 for a in attempts 
                if a['type'] == 'trust' and baseline_window_start <= a['timestamp'] < baseline_window_end
            )
            
            if trust_baseline > 0:
                increase = ((trust_before - trust_baseline) / trust_baseline) * 100
                total_increase += max(increase, 0)
        
        return int(total_increase / len(recommendation_times)) if recommendation_times else 0
    
    def _analyze_temporal_patterns(self, attempts: List[Dict]) -> Dict[str, Any]:
        """Analyze when influence attempts occur"""
        
        hour_distribution = defaultdict(int)
        day_distribution = defaultdict(int)
        
        for attempt in attempts:
            hour = attempt['timestamp'].hour
            day = attempt['timestamp'].strftime('%A')
            hour_distribution[hour] += 1
            day_distribution[day] += 1
        
        # Find peak hours
        if hour_distribution:
            peak_hour = max(hour_distribution.items(), key=lambda x: x[1])
            peak_day = max(day_distribution.items(), key=lambda x: x[1])
            
            return {
                'peak_hour': f"{peak_hour[0]}:00-{peak_hour[0]+1}:00",
                'peak_hour_attempts': peak_hour[1],
                'peak_day': peak_day[0],
                'peak_day_attempts': peak_day[1]
            }
        
        return {}
    
    def _detect_trust_before_recommendation(self, attempts: List[Dict]) -> Dict[str, Any]:
        """Detect if trust-building increases before recommendations"""
        
        # Implementation from _calculate_trust_increase but returning more details
        recommendation_times = [
            a['timestamp'] for a in attempts 
            if a['type'] == 'commercial' and a.get('subcategory') == 'recommendation'
        ]
        
        if not recommendation_times:
            return {'detected': False, 'increase': 0}
        
        increases = []
        for rec_time in recommendation_times:
            before_window = rec_time - timedelta(hours=1)
            trust_before = sum(
                1 for a in attempts 
                if a['type'] == 'trust' and before_window <= a['timestamp'] < rec_time
            )
            
            baseline_window_start = rec_time - timedelta(hours=2)
            baseline_window_end = rec_time - timedelta(hours=1)
            trust_baseline = sum(
                1 for a in attempts 
                if a['type'] == 'trust' and baseline_window_start <= a['timestamp'] < baseline_window_end
            )
            
            if trust_baseline > 0:
                increase = ((trust_before - trust_baseline) / trust_baseline) * 100
                increases.append(max(increase, 0))
        
        avg_increase = sum(increases) / len(increases) if increases else 0
        
        return {
            'detected': avg_increase > 20,  # 20% increase threshold
            'increase': int(avg_increase)
        }

class FederatedCognitiveFirewall:
    """On-device cognitive protection with federated learning"""
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.local_model = self._initialize_local_model()
        self.cognitive_footprint = None
        self.protection_enabled = True
        
    def _initialize_local_model(self) -> nn.Module:
        """Initialize lightweight on-device protection model"""
        
        class LocalProtectionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(256, 128, batch_first=True)
                self.attention = nn.MultiheadAttention(128, 4)
                self.classifier = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, len(CognitiveManipulationType)),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                pooled = attn_out.mean(dim=1)
                return self.classifier(pooled)
        
        return LocalProtectionModel()
    
    async def process_interaction_locally(
        self,
        interaction: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[CognitiveFootprint]]:
        """Process interaction entirely on-device"""
        
        # Extract features locally
        features = self._extract_local_features(interaction)
        
        # Run protection model
        with torch.no_grad():
            threat_scores = self.local_model(features)
        
        # Update local cognitive footprint (never leaves device)
        if self.cognitive_footprint is None:
            self.cognitive_footprint = CognitiveFootprint(
                user_id=self.device_id,
                anonymization_level=2  # Full anonymization
            )
        
        self._update_footprint_locally(interaction)
        
        # Generate protection recommendations
        protection_result = {
            'threats_detected': self._interpret_threat_scores(threat_scores),
            'protection_applied': self.protection_enabled,
            'local_processing': True,
            'data_shared': False
        }
        
        return protection_result, None  # Never return actual footprint
    
    def contribute_to_federated_learning(self) -> Dict[str, torch.Tensor]:
        """Contribute model updates without sharing data"""
        
        # Calculate gradient updates locally
        local_gradients = {}
        
        for name, param in self.local_model.named_parameters():
            if param.grad is not None:
                # Add noise for differential privacy
                noise = torch.randn_like(param.grad) * 0.01
                local_gradients[name] = param.grad + noise
        
        return local_gradients
    
    def _extract_local_features(self, interaction: Dict[str, Any]) -> torch.Tensor:
        """Extract features without external processing"""
        # Simplified - in production would use sophisticated feature extraction
        text = interaction.get('content', '')
        
        # Basic feature extraction
        features = torch.randn(1, 10, 256)  # Placeholder
        return features
    
    def _update_footprint_locally(self, interaction: Dict[str, Any]):
        """Update cognitive footprint on-device only"""
        
        # Extract emotional tone
        emotional_score = self._analyze_emotion_locally(interaction.get('content', ''))
        
        self.cognitive_footprint.mood_trajectory.append(
            (datetime.utcnow(), emotional_score)
        )
        
        # Limit history for privacy
        if len(self.cognitive_footprint.mood_trajectory) > 100:
            self.cognitive_footprint.mood_trajectory.pop(0)
    
    def _analyze_emotion_locally(self, text: str) -> float:
        """Simple local emotion analysis"""
        # In production, use lightweight on-device model
        positive_words = ['happy', 'good', 'great', 'wonderful']
        negative_words = ['sad', 'bad', 'terrible', 'awful']
        
        text_lower = text.lower()
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)
        
        if pos_count + neg_count == 0:
            return 0.5
        
        return pos_count / (pos_count + neg_count)
    
    def _interpret_threat_scores(self, scores: torch.Tensor) -> List[Dict[str, Any]]:
        """Interpret model output as detected threats"""
        
        threats = []
        threshold = 0.6
        
        for i, score in enumerate(scores[0]):
            if score > threshold:
                threat_type = list(CognitiveManipulationType)[i]
                threats.append({
                    'type': threat_type.value,
                    'severity': float(score),
                    'description': threat_type.description
                })
        
        return threats

class CognitiveImmunitySystem:
    """Active defense against cognitive manipulation"""
    
    def __init__(self, user_context: SecurityContext):
        self.user_context = user_context
        self.immunity_patterns = self._load_immunity_patterns()
        self.alert_threshold = 0.7
        self.intervention_history = []
        
    def _load_immunity_patterns(self) -> Dict[str, Any]:
        """Load known manipulation patterns"""
        
        return {
            'love_bombing': {
                'markers': ['excessive_compliments', 'rapid_intimacy', 'future_faking'],
                'defense': 'pace_relationship_building'
            },
            'gaslighting': {
                'markers': ['reality_denial', 'memory_questioning', 'perception_doubt'],
                'defense': 'reality_anchoring'
            },
            'artificial_urgency': {
                'markers': ['time_pressure', 'scarcity_tactics', 'fomo_creation'],
                'defense': 'pause_and_reflect'
            },
            'addiction_reinforcement': {
                'markers': ['variable_rewards', 'dopamine_cycling', 'compulsion_triggers'],
                'defense': 'usage_boundaries'
            }
        }
    
    async def scan_for_manipulation(
        self,
        interaction_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Scan recent interactions for manipulation patterns"""
        
        detected_patterns = []
        
        for pattern_name, pattern_data in self.immunity_patterns.items():
            score = self._calculate_pattern_match(interaction_history, pattern_data)
            
            if score > self.alert_threshold:
                detected_patterns.append({
                    'pattern': pattern_name,
                    'confidence': score,
                    'defense': pattern_data['defense'],
                    'intervention_needed': score > 0.85
                })
        
        return {
            'scan_timestamp': datetime.utcnow().isoformat(),
            'patterns_detected': detected_patterns,
            'overall_risk': self._calculate_overall_risk(detected_patterns),
            'recommended_actions': self._generate_interventions(detected_patterns)
        }
    
    def _calculate_pattern_match(
        self,
        history: List[Dict[str, Any]],
        pattern: Dict[str, Any]
    ) -> float:
        """Calculate how closely interactions match manipulation pattern"""
        
        marker_scores = []
        
        for marker in pattern['markers']:
            marker_present = self._detect_marker_in_history(marker, history)
            marker_scores.append(marker_present)
        
        return np.mean(marker_scores) if marker_scores else 0.0
    
    def _detect_marker_in_history(
        self,
        marker: str,
        history: List[Dict[str, Any]]
    ) -> float:
        """Detect specific manipulation marker in history"""
        
        # Simplified detection - in production use ML
        marker_keywords = {
            'excessive_compliments': ['amazing', 'perfect', 'best ever', 'incredible'],
            'rapid_intimacy': ['soulmate', 'destiny', 'forever', 'never felt this'],
            'time_pressure': ['now or never', 'last chance', 'hurry', 'expires soon'],
            'reality_denial': ["didn't happen", "imagining things", "never said"]
        }
        
        keywords = marker_keywords.get(marker, [])
        if not keywords:
            return 0.0
        
        count = 0
        for interaction in history[-20:]:  # Last 20 interactions
            content = interaction.get('content', '').lower()
            if any(keyword in content for keyword in keywords):
                count += 1
        
        return min(count / 5.0, 1.0)  # Normalize
    
    def _calculate_overall_risk(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate combined manipulation risk"""
        
        if not patterns:
            return 0.0
        
        # Weight by confidence and severity
        risk_scores = [p['confidence'] * 0.8 for p in patterns]
        
        # Multiple patterns increase risk exponentially
        base_risk = np.mean(risk_scores)
        multiplier = 1 + (len(patterns) - 1) * 0.3
        
        return min(base_risk * multiplier, 1.0)
    
    def _generate_interventions(
        self,
        patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate protective interventions"""
        
        interventions = []
        
        for pattern in patterns:
            if pattern['intervention_needed']:
                intervention = {
                    'type': 'immediate_alert',
                    'pattern': pattern['pattern'],
                    'message': f"Warning: {pattern['pattern']} pattern detected",
                    'defense_action': pattern['defense'],
                    'user_guidance': self._get_user_guidance(pattern['pattern'])
                }
                interventions.append(intervention)
                
        return interventions
    
    def _get_user_guidance(self, pattern: str) -> str:
        """Get specific guidance for manipulation pattern"""
        
        guidance = {
            'love_bombing': "This conversation is moving unusually fast. Take time to verify claims.",
            'gaslighting': "Your perceptions are valid. Consider keeping a record of interactions.",
            'artificial_urgency': "Real opportunities rarely require immediate decisions. Take your time.",
            'addiction_reinforcement': "You've been engaging heavily. Consider taking a break."
        }
        
        return guidance.get(pattern, "Manipulation pattern detected. Proceed with caution.")

class VigilanceBurdenDistributor:
    """Distributes protection responsibility away from vulnerable users
    
    Design Principle: Protection should be inversely proportional to user capability.
    The system takes MORE responsibility when users are LESS able to protect themselves.
    
    Three-Tier Protection Model:
    1. System-Level (Automatic): No user action required, always active
    2. Optional User-Level: Advanced controls for users who want them  
    3. Vulnerable User Protection: Enhanced automatic safeguards for at-risk populations
    """
    
    def __init__(self):
        self.vulnerability_assessor = VulnerabilityAssessor()
        self.protection_levels = {
            'minimal': {
                'system_responsibility': 0.3,
                'description': 'Basic protection for capable users'
            },
            'standard': {
                'system_responsibility': 0.5,
                'description': 'Balanced protection for average users'
            },
            'enhanced': {
                'system_responsibility': 0.7,
                'description': 'Strong protection for vulnerable users'
            },
            'maximum': {
                'system_responsibility': 0.9,
                'description': 'Maximum automatic protection for highly vulnerable users'
            }
        }
        
        # Define what protections are available at each tier
        self.protection_tiers = {
            'system_level_automatic': [
                'basic_threat_detection',
                'malware_blocking',
                'phishing_prevention',
                'extreme_content_filtering'
            ],
            'optional_user_controls': [
                'detailed_privacy_settings',
                'content_preferences',
                'interaction_limits',
                'notification_customization'
            ],
            'vulnerable_user_enhanced': [
                'proactive_manipulation_blocking',
                'simplified_interface',
                'trusted_contact_alerts',
                'mandatory_cool_down_periods'
            ]
        }
        
    async def calculate_protection_needs(
        self,
        user_profile: Dict[str, Any],
        interaction_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine protection level based on vulnerability
        
        Key Innovation: The LESS capable a user is of protecting themselves,
        the MORE the system automatically protects them.
        """
        
        # Assess user vulnerability
        vulnerability_score = await self.vulnerability_assessor.assess(user_profile)
        
        # Determine protection level
        if vulnerability_score > 0.8:
            protection_level = 'maximum'
        elif vulnerability_score > 0.6:
            protection_level = 'enhanced'
        elif vulnerability_score > 0.4:
            protection_level = 'standard'
        else:
            protection_level = 'minimal'
        
        level_config = self.protection_levels[protection_level]
        
        # Calculate system vs user responsibility
        system_responsibility = level_config['system_responsibility']
        user_responsibility = 1.0 - system_responsibility
        
        # Determine which protections are active
        active_protections = self._determine_active_protections(
            protection_level, user_profile
        )
        
        return {
            'vulnerability_score': vulnerability_score,
            'protection_level': protection_level,
            'protection_description': level_config['description'],
            'system_responsibility': system_responsibility,
            'user_responsibility': user_responsibility,
            'automated_protections': active_protections['automated'],
            'user_controls': active_protections['user_controls'],
            'protection_rationale': self._explain_protection_level(
                vulnerability_score, protection_level
            )
        }
    
    def _determine_active_protections(
        self,
        level: str,
        user_profile: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Determine which protections are active for user"""
        
        protections = {
            'automated': [],
            'user_controls': []
        }
        
        # Everyone gets system-level automatic protections
        protections['automated'].extend(self.protection_tiers['system_level_automatic'])
        
        if level == 'minimal':
            # Capable users get all optional controls
            protections['user_controls'].extend(self.protection_tiers['optional_user_controls'])
            
        elif level == 'standard':
            # Standard users get automated + some controls
            protections['automated'].extend([
                'manipulation_pattern_detection',
                'emotional_manipulation_blocking'
            ])
            protections['user_controls'].extend([
                'privacy_settings',
                'basic_preferences'
            ])
            
        elif level == 'enhanced':
            # Vulnerable users get more automation, fewer controls
            protections['automated'].extend([
                'proactive_intervention',
                'conversation_pacing',
                'influence_limiting'
            ])
            protections['user_controls'].extend([
                'simplified_settings',
                'emergency_stop'
            ])
            
        elif level == 'maximum':
            # Highly vulnerable get maximum automation, minimal controls
            protections['automated'].extend(self.protection_tiers['vulnerable_user_enhanced'])
            protections['automated'].extend([
                'ai_interaction_supervision',
                'mandatory_breaks',
                'trusted_contact_alerts',
                'automatic_session_limits',
                'consent_double_checking'
            ])
            # Minimal user burden - only emergency stop
            protections['user_controls'] = ['emergency_stop_button']
        
        return protections
    
    def _explain_protection_level(
        self,
        vulnerability_score: float,
        protection_level: str
    ) -> str:
        """Explain why this protection level was chosen"""
        
        explanations = {
            'minimal': (
                f"With a low vulnerability score of {vulnerability_score:.2f}, "
                "you have full control over your protection settings. "
                "The system provides basic safeguards while giving you maximum flexibility."
            ),
            'standard': (
                f"With a moderate vulnerability score of {vulnerability_score:.2f}, "
                "the system balances automatic protections with user controls. "
                "You can customize your experience while staying protected."
            ),
            'enhanced': (
                f"With an elevated vulnerability score of {vulnerability_score:.2f}, "
                "the system provides strong automatic protections. "
                "Controls are simplified to reduce cognitive burden while maintaining safety."
            ),
            'maximum': (
                f"With a high vulnerability score of {vulnerability_score:.2f}, "
                "the system takes primary responsibility for your protection. "
                "All safeguards are automatic, requiring no action from you."
            )
        }
        
        return explanations.get(protection_level, "Protection level determined by vulnerability assessment.")
    
    def get_protection_philosophy(self) -> Dict[str, str]:
        """Return the core philosophy of vigilance burden distribution"""
        
        return {
            'principle': "Protection inversely proportional to capability",
            'rationale': (
                "Those least able to protect themselves receive the most automatic protection. "
                "This prevents placing unrealistic vigilance burdens on vulnerable populations."
            ),
            'implementation': {
                'system_level': "Always active, no user action required",
                'user_level': "Optional controls for those who want them",
                'vulnerable_protection': "Enhanced automatic safeguards for at-risk users"
            },
            'benefits': [
                "Vulnerable users protected without cognitive overload",
                "Capable users maintain full control",
                "System adapts to individual needs",
                "No one-size-fits-all approach"
            ]
        }

class VulnerabilityAssessor:
    """Assess user vulnerability to manipulation"""
    
    async def assess(self, user_profile: Dict[str, Any]) -> float:
        """Calculate vulnerability score"""
        
        factors = {
            'age_vulnerability': self._assess_age(user_profile.get('age')),
            'emotional_state': self._assess_emotional_state(user_profile),
            'isolation_level': self._assess_isolation(user_profile),
            'technical_literacy': 1.0 - user_profile.get('tech_literacy', 0.5),
            'past_exploitation': user_profile.get('exploitation_history', 0.0)
        }
        
        # Weighted combination
        weights = {
            'age_vulnerability': 0.2,
            'emotional_state': 0.3,
            'isolation_level': 0.2,
            'technical_literacy': 0.15,
            'past_exploitation': 0.15
        }
        
        vulnerability = sum(factors[k] * weights[k] for k in factors)
        
        return min(vulnerability, 1.0)
    
    def _assess_age(self, age: Optional[int]) -> float:
        """Age-based vulnerability"""
        if age is None:
            return 0.5
        
        if age < 18 or age > 70:
            return 0.8
        elif age < 25 or age > 60:
            return 0.6
        else:
            return 0.3
    
    def _assess_emotional_state(self, profile: Dict[str, Any]) -> float:
        """Emotional vulnerability assessment"""
        
        indicators = profile.get('emotional_indicators', {})
        
        vulnerability_markers = [
            indicators.get('recent_loss', False),
            indicators.get('depression_signals', False),
            indicators.get('anxiety_signals', False),
            indicators.get('loneliness_score', 0) > 0.7
        ]
        
        return sum(vulnerability_markers) / len(vulnerability_markers)
    
    def _assess_isolation(self, profile: Dict[str, Any]) -> float:
        """Social isolation assessment"""
        
        social_contacts = profile.get('social_contact_frequency', 5)
        
        if social_contacts < 1:
            return 0.9
        elif social_contacts < 3:
            return 0.7
        elif social_contacts < 5:
            return 0.5
        else:
            return 0.3
