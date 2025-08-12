# Production AAML Framework - Distributed Governance & Reputation
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





# aaml_framework/governance/distributed_system.py

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import redis
from web3 import Web3
from eth_account import Account
import ipfshttpclient
from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import structlog
from decimal import Decimal
import numpy as np
from collections import defaultdict
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization

logger = structlog.get_logger()

Base = declarative_base()

# Smart Contract ABI for governance token
GOVERNANCE_ABI = json.loads('''[
    {"inputs": [], "name": "totalSupply", "outputs": [{"type": "uint256"}], "type": "function"},
    {"inputs": [{"type": "address"}], "name": "balanceOf", "outputs": [{"type": "uint256"}], "type": "function"},
    {"inputs": [{"type": "uint256"}, {"type": "string"}], "name": "createProposal", "outputs": [{"type": "uint256"}], "type": "function"},
    {"inputs": [{"type": "uint256"}, {"type": "bool"}], "name": "vote", "outputs": [], "type": "function"}
]''')

class GovernanceRole(Enum):
    """Roles in the governance system"""
    OBSERVER = "observer"
    CONTRIBUTOR = "contributor"
    VALIDATOR = "validator"
    COUNCIL_MEMBER = "council_member"
    ARBITRATOR = "arbitrator"

class ProposalStatus(Enum):
    """Proposal lifecycle states"""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    VOTING = "voting"
    PASSED = "passed"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    ARCHIVED = "archived"

@dataclass
class ReputationUpdate:
    """Reputation change event"""
    entity_id: str
    amount: float
    reason: str
    evidence: Dict[str, Any]
    timestamp: datetime
    validator_id: Optional[str] = None
    
class Entity(Base):
    """Database model for entities in the system"""
    __tablename__ = 'entities'
    
    id = Column(String, primary_key=True)
    entity_type = Column(String)  # 'ai', 'human', 'hybrid', 'dao'
    public_key = Column(String)
    reputation_score = Column(Float, default=0.0)
    governance_weight = Column(Float, default=0.0)
    roles = Column(JSON, default=list)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class Proposal(Base):
    """Database model for governance proposals"""
    __tablename__ = 'proposals'
    
    id = Column(String, primary_key=True)
    proposer_id = Column(String)
    title = Column(String)
    description = Column(String)
    proposal_type = Column(String)
    status = Column(String, default=ProposalStatus.DRAFT.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    voting_starts = Column(DateTime)
    voting_ends = Column(DateTime)
    execution_delay = Column(Integer, default=86400)  # 24 hours
    ipfs_hash = Column(String)  # Full proposal stored on IPFS
    on_chain_id = Column(Integer)  # Blockchain proposal ID
    results = Column(JSON, default=dict)
    metadata = Column(JSON, default=dict)

class Vote(Base):
    """Database model for votes"""
    __tablename__ = 'votes'
    
    id = Column(String, primary_key=True)
    proposal_id = Column(String)
    voter_id = Column(String)
    vote_power = Column(Float)
    choice = Column(String)  # 'for', 'against', 'abstain'
    reason = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    signature = Column(String)  # Cryptographic signature

class ReputationSystem:
    """Advanced reputation system with decay and verification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = create_engine(config['database_url'])
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Redis for real-time reputation tracking
        self.redis = redis.Redis(
            host=config['redis_host'],
            port=config['redis_port'],
            decode_responses=True
        )
        
        # Reputation parameters
        self.decay_rate = config.get('reputation_decay_rate', 0.01)  # 1% per week
        self.max_single_update = config.get('max_reputation_change', 10.0)
        self.verification_threshold = config.get('verification_threshold', 3)
        
        # Initialize reputation calculator
        self.calculator = ReputationCalculator()
    
    async def update_reputation(
        self,
        update: ReputationUpdate,
        validators: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Update entity reputation with verification"""
        
        # Validate update
        validation_result = await self._validate_update(update, validators)
        if not validation_result['valid']:
            return {
                'success': False,
                'reason': validation_result['reason']
            }
        
        # Apply bounds
        bounded_amount = np.clip(
            update.amount,
            -self.max_single_update,
            self.max_single_update
        )
        
        # Get current reputation
        session = self.Session()
        try:
            entity = session.query(Entity).filter_by(id=update.entity_id).first()
            if not entity:
                return {'success': False, 'reason': 'Entity not found'}
            
            # Apply decay
            decayed_reputation = self._apply_decay(
                entity.reputation_score,
                entity.last_active
            )
            
            # Calculate new reputation
            new_reputation = decayed_reputation + bounded_amount
            
            # Update database
            entity.reputation_score = new_reputation
            entity.last_active = datetime.utcnow()
            
            # Log update
            self._log_reputation_update(session, update, bounded_amount)
            
            session.commit()
            
            # Update cache
            self.redis.zadd(
                'reputation_scores',
                {update.entity_id: new_reputation}
            )
            
            # Broadcast update
            await self._broadcast_reputation_update(update, new_reputation)
            
            return {
                'success': True,
                'new_reputation': new_reputation,
                'applied_change': bounded_amount,
                'validators': validation_result.get('validators', [])
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"Reputation update failed: {e}")
            return {'success': False, 'reason': str(e)}
        finally:
            session.close()
    
    async def _validate_update(
        self,
        update: ReputationUpdate,
        validators: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Validate reputation update"""
        
        # Check evidence
        if not update.evidence:
            return {'valid': False, 'reason': 'No evidence provided'}
        
        # Verify validators if required
        if self.config.get('require_validation', True):
            if not validators or len(validators) < self.verification_threshold:
                return {
                    'valid': False,
                    'reason': f'Insufficient validators (need {self.verification_threshold})'
                }
            
            # Check validator eligibility
            valid_validators = await self._verify_validators(validators)
            if len(valid_validators) < self.verification_threshold:
                return {'valid': False, 'reason': 'Insufficient valid validators'}
            
            return {'valid': True, 'validators': valid_validators}
        
        return {'valid': True}
    
    async def _verify_validators(self, validator_ids: List[str]) -> List[str]:
        """Verify validators are eligible"""
        
        session = self.Session()
        try:
            valid_validators = []
            
            for validator_id in validator_ids:
                validator = session.query(Entity).filter_by(id=validator_id).first()
                
                if validator and GovernanceRole.VALIDATOR.value in validator.roles:
                    if validator.reputation_score >= self.config.get('min_validator_reputation', 50):
                        valid_validators.append(validator_id)
            
            return valid_validators
            
        finally:
            session.close()
    
    def _apply_decay(self, reputation: float, last_active: datetime) -> float:
        """Apply time-based reputation decay"""
        
        if not last_active:
            return reputation
        
        weeks_inactive = (datetime.utcnow() - last_active).days / 7
        decay_factor = (1 - self.decay_rate) ** weeks_inactive
        
        return reputation * decay_factor
    
    def _log_reputation_update(
        self,
        session,
        update: ReputationUpdate,
        applied_amount: float
    ):
        """Log reputation update for audit trail"""
        
        # In production, this would be a separate audit table
        log_entry = {
            'entity_id': update.entity_id,
            'amount': applied_amount,
            'reason': update.reason,
            'evidence_hash': hashlib.sha256(
                json.dumps(update.evidence, sort_keys=True).encode()
            ).hexdigest(),
            'timestamp': update.timestamp.isoformat(),
            'validator_id': update.validator_id
        }
        
        # Store in Redis for recent history
        self.redis.lpush(
            f'reputation_log:{update.entity_id}',
            json.dumps(log_entry)
        )
        self.redis.ltrim(f'reputation_log:{update.entity_id}', 0, 999)
    
    async def _broadcast_reputation_update(
        self,
        update: ReputationUpdate,
        new_reputation: float
    ):
        """Broadcast reputation update to network"""
        
        # Publish to Redis pub/sub
        self.redis.publish(
            'reputation_updates',
            json.dumps({
                'entity_id': update.entity_id,
                'new_reputation': new_reputation,
                'reason': update.reason,
                'timestamp': update.timestamp.isoformat()
            })
        )
    
    def calculate_governance_weight(self, entity_id: str) -> float:
        """Calculate voting power based on reputation"""
        
        session = self.Session()
        try:
            entity = session.query(Entity).filter_by(id=entity_id).first()
            
            if not entity:
                return 0.0
            
            # Apply decay
            current_reputation = self._apply_decay(
                entity.reputation_score,
                entity.last_active
            )
            
            # Non-linear transformation for governance weight
            # Using square root to prevent reputation concentration
            weight = np.sqrt(max(current_reputation, 0))
            
            # Apply role multipliers
            if GovernanceRole.COUNCIL_MEMBER.value in entity.roles:
                weight *= 1.5
            elif GovernanceRole.VALIDATOR.value in entity.roles:
                weight *= 1.2
            
            return weight
            
        finally:
            session.close()

class DistributedGovernance:
    """Blockchain-integrated governance system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = create_engine(config['database_url'])
        self.Session = sessionmaker(bind=self.engine)
        
        # Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(config['ethereum_rpc']))
        self.contract = self.w3.eth.contract(
            address=config['governance_contract'],
            abi=GOVERNANCE_ABI
        )
        
        # IPFS client
        self.ipfs = ipfshttpclient.connect(
            f"/dns/{config['ipfs_host']}/tcp/{config['ipfs_port']}/http"
        )
        
        # Reputation system
        self.reputation = ReputationSystem(config)
        
        # Voting parameters
        self.min_quorum = config.get('min_quorum', 0.1)  # 10% participation
        self.pass_threshold = config.get('pass_threshold', 0.5)  # 50% approval
        
    async def create_proposal(
        self,
        proposer_id: str,
        proposal_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new governance proposal"""
        
        # Verify proposer eligibility
        if not await self._can_propose(proposer_id):
            return {
                'success': False,
                'reason': 'Insufficient reputation or wrong role'
            }
        
        # Generate proposal ID
        proposal_id = hashlib.sha256(
            f"{proposer_id}{datetime.utcnow().isoformat()}{proposal_data['title']}".encode()
        ).hexdigest()[:16]
        
        # Store full proposal on IPFS
        ipfs_hash = self.ipfs.add_json(proposal_data)
        
        # Create database entry
        session = self.Session()
        try:
            proposal = Proposal(
                id=proposal_id,
                proposer_id=proposer_id,
                title=proposal_data['title'],
                description=proposal_data.get('description', ''),
                proposal_type=proposal_data.get('type', 'general'),
                status=ProposalStatus.SUBMITTED.value,
                voting_starts=datetime.utcnow() + timedelta(days=1),
                voting_ends=datetime.utcnow() + timedelta(days=8),
                ipfs_hash=ipfs_hash,
                metadata=proposal_data.get('metadata', {})
            )
            
            session.add(proposal)
            
            # Submit to blockchain if configured
            if self.config.get('use_blockchain', False):
                on_chain_id = await self._submit_to_chain(proposal_id, ipfs_hash)
                proposal.on_chain_id = on_chain_id
            
            session.commit()
            
            # Notify governance participants
            await self._notify_proposal_created(proposal)
            
            return {
                'success': True,
                'proposal_id': proposal_id,
                'ipfs_hash': ipfs_hash,
                'voting_starts': proposal.voting_starts.isoformat(),
                'voting_ends': proposal.voting_ends.isoformat()
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"Proposal creation failed: {e}")
            return {'success': False, 'reason': str(e)}
        finally:
            session.close()
    
    async def cast_vote(
        self,
        voter_id: str,
        proposal_id: str,
        choice: str,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cast a vote on a proposal"""
        
        # Validate choice
        if choice not in ['for', 'against', 'abstain']:
            return {'success': False, 'reason': 'Invalid vote choice'}
        
        session = self.Session()
        try:
            # Check proposal exists and is in voting phase
            proposal = session.query(Proposal).filter_by(id=proposal_id).first()
            if not proposal:
                return {'success': False, 'reason': 'Proposal not found'}
            
            if proposal.status != ProposalStatus.VOTING.value:
                return {'success': False, 'reason': 'Proposal not in voting phase'}
            
            now = datetime.utcnow()
            if now < proposal.voting_starts or now > proposal.voting_ends:
                return {'success': False, 'reason': 'Outside voting period'}
            
            # Check if already voted
            existing_vote = session.query(Vote).filter_by(
                proposal_id=proposal_id,
                voter_id=voter_id
            ).first()
            
            if existing_vote:
                return {'success': False, 'reason': 'Already voted'}
            
            # Calculate vote power
            vote_power = self.reputation.calculate_governance_weight(voter_id)
            if vote_power <= 0:
                return {'success': False, 'reason': 'No voting power'}
            
            # Create vote
            vote_id = hashlib.sha256(
                f"{proposal_id}{voter_id}{now.isoformat()}".encode()
            ).hexdigest()[:16]
            
            vote = Vote(
                id=vote_id,
                proposal_id=proposal_id,
                voter_id=voter_id,
                vote_power=vote_power,
                choice=choice,
                reason=reason,
                signature=self._sign_vote(voter_id, proposal_id, choice)
            )
            
            session.add(vote)
            
            # Submit to blockchain if configured
            if self.config.get('use_blockchain', False):
                await self._submit_vote_to_chain(vote)
            
            session.commit()
            
            return {
                'success': True,
                'vote_id': vote_id,
                'vote_power': vote_power,
                'choice': choice
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"Vote casting failed: {e}")
            return {'success': False, 'reason': str(e)}
        finally:
            session.close()
    
    async def tally_votes(self, proposal_id: str) -> Dict[str, Any]:
        """Tally votes and determine proposal outcome"""
        
        session = self.Session()
        try:
            proposal = session.query(Proposal).filter_by(id=proposal_id).first()
            if not proposal:
                return {'success': False, 'reason': 'Proposal not found'}
            
            # Get all votes
            votes = session.query(Vote).filter_by(proposal_id=proposal_id).all()
            
            # Calculate totals
            vote_totals = {
                'for': sum(v.vote_power for v in votes if v.choice == 'for'),
                'against': sum(v.vote_power for v in votes if v.choice == 'against'),
                'abstain': sum(v.vote_power for v in votes if v.choice == 'abstain')
            }
            
            total_power = sum(vote_totals.values())
            
            # Check quorum
            eligible_power = self._calculate_eligible_voting_power()
            participation = total_power / eligible_power if eligible_power > 0 else 0
            
            if participation < self.min_quorum:
                proposal.status = ProposalStatus.REJECTED.value
                proposal.results = {
                    'outcome': 'failed_quorum',
                    'participation': participation,
                    'vote_totals': vote_totals
                }
            else:
                # Check if passed
                approval_rate = vote_totals['for'] / (vote_totals['for'] + vote_totals['against'])
                
                if approval_rate >= self.pass_threshold:
                    proposal.status = ProposalStatus.PASSED.value
                    proposal.results = {
                        'outcome': 'passed',
                        'approval_rate': approval_rate,
                        'vote_totals': vote_totals
                    }
                else:
                    proposal.status = ProposalStatus.REJECTED.value
                    proposal.results = {
                        'outcome': 'rejected',
                        'approval_rate': approval_rate,
                        'vote_totals': vote_totals
                    }
            
            session.commit()
            
            # Notify participants
            await self._notify_proposal_outcome(proposal)
            
            return {
                'success': True,
                'proposal_id': proposal_id,
                'results': proposal.results
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"Vote tallying failed: {e}")
            return {'success': False, 'reason': str(e)}
        finally:
            session.close()
    
    async def execute_proposal(self, proposal_id: str) -> Dict[str, Any]:
        """Execute a passed proposal"""
        
        session = self.Session()
        try:
            proposal = session.query(Proposal).filter_by(id=proposal_id).first()
            if not proposal:
                return {'success': False, 'reason': 'Proposal not found'}
            
            if proposal.status != ProposalStatus.PASSED.value:
                return {'success': False, 'reason': 'Proposal not passed'}
            
            # Check execution delay
            time_since_end = (datetime.utcnow() - proposal.voting_ends).seconds
            if time_since_end < proposal.execution_delay:
                return {
                    'success': False,
                    'reason': f'Must wait {proposal.execution_delay - time_since_end} seconds'
                }
            
            # Execute based on proposal type
            execution_result = await self._execute_by_type(proposal)
            
            if execution_result['success']:
                proposal.status = ProposalStatus.IMPLEMENTED.value
                session.commit()
            
            return execution_result
            
        except Exception as e:
            session.rollback()
            logger.error(f"Proposal execution failed: {e}")
            return {'success': False, 'reason': str(e)}
        finally:
            session.close()
    
    async def _can_propose(self, entity_id: str) -> bool:
        """Check if entity can create proposals"""
        
        session = self.Session()
        try:
            entity = session.query(Entity).filter_by(id=entity_id).first()
            
            if not entity:
                return False
            
            # Check reputation threshold
            min_reputation = self.config.get('min_proposer_reputation', 10.0)
            if entity.reputation_score < min_reputation:
                return False
            
            # Check roles
            allowed_roles = [
                GovernanceRole.CONTRIBUTOR.value,
                GovernanceRole.COUNCIL_MEMBER.value
            ]
            
            return any(role in entity.roles for role in allowed_roles)
            
        finally:
            session.close()
    
    async def _submit_to_chain(self, proposal_id: str, ipfs_hash: str) -> int:
        """Submit proposal to blockchain"""
        
        # Build transaction
        account = Account.from_key(self.config['private_key'])
        
        # Get gas price
        gas_price = self.w3.eth.gas_price
        
        # Build transaction
        tx = self.contract.functions.createProposal(
            self.w3.toWei(1, 'ether'),  # Proposal bond
            ipfs_hash
        ).buildTransaction({
            'from': account.address,
            'gas': 200000,
            'gasPrice': gas_price,
            'nonce': self.w3.eth.get_transaction_count(account.address)
        })
        
        # Sign and send
        signed_tx = account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for confirmation
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Extract proposal ID from events
        # In production, parse events properly
        return receipt['blockNumber']
    
    def _sign_vote(self, voter_id: str, proposal_id: str, choice: str) -> str:
        """Create cryptographic signature for vote"""
        
        # In production, use proper key management
        message = f"{voter_id}:{proposal_id}:{choice}"
        signature = hashlib.sha256(message.encode()).hexdigest()
        
        return signature
    
    def _update_vote_tally(self, proposal_id: str, choice: str, vote_power: float):
        """Update cached vote tallies"""
        
        # Use Redis for real-time tallies
        redis_client = redis.Redis(
            host=self.config['redis_host'],
            port=self.config['redis_port']
        )
        
        redis_client.hincrbyfloat(
            f"proposal_tally:{proposal_id}",
            choice,
            vote_power
        )
        
        # Set expiry
        redis_client.expire(f"proposal_tally:{proposal_id}", 86400 * 30)  # 30 days
    
    def _calculate_eligible_voting_power(self) -> float:
        """Calculate total eligible voting power"""
        
        session = self.Session()
        try:
            entities = session.query(Entity).filter(
                Entity.is_active == True
            ).all()
            
            total_power = sum(
                self.reputation.calculate_governance_weight(e.id)
                for e in entities
            )
            
            return total_power
            
        finally:
            session.close()
    
    async def _notify_proposal_outcome(self, proposal: Proposal):
        """Notify participants of proposal outcome"""
        
        # In production, use notification service
        logger.info(
            f"Proposal {proposal.id} outcome: {proposal.results.get('outcome')}"
        )
    
    async def _execute_by_type(self, proposal: Proposal) -> Dict[str, Any]:
        """Execute proposal based on type"""
        
        # Load full proposal from IPFS
        proposal_data = self.ipfs.get_json(proposal.ipfs_hash)
        
        if proposal.proposal_type == 'parameter_change':
            return await self._execute_parameter_change(proposal_data)
        elif proposal.proposal_type == 'role_assignment':
            return await self._execute_role_assignment(proposal_data)
        elif proposal.proposal_type == 'fund_allocation':
            return await self._execute_fund_allocation(proposal_data)
        else:
            return {
                'success': False,
                'reason': f'Unknown proposal type: {proposal.proposal_type}'
            }
    
    async def _execute_parameter_change(self, proposal_data: Dict) -> Dict[str, Any]:
        """Execute parameter change proposal"""
        
        parameter = proposal_data.get('parameter')
        new_value = proposal_data.get('new_value')
        
        # Validate parameter
        allowed_parameters = [
            'min_quorum', 'pass_threshold', 'voting_period',
            'execution_delay', 'reputation_decay_rate'
        ]
        
        if parameter not in allowed_parameters:
            return {'success': False, 'reason': 'Invalid parameter'}
        
        # Update configuration
        self.config[parameter] = new_value
        
        # Persist change
        # In production, use proper configuration management
        
        return {
            'success': True,
            'parameter': parameter,
            'new_value': new_value
        }
    
    async def _execute_role_assignment(self, proposal_data: Dict) -> Dict[str, Any]:
        """Execute role assignment proposal"""
        
        entity_id = proposal_data.get('entity_id')
        role = proposal_data.get('role')
        action = proposal_data.get('action', 'add')  # 'add' or 'remove'
        
        session = self.Session()
        try:
            entity = session.query(Entity).filter_by(id=entity_id).first()
            if not entity:
                return {'success': False, 'reason': 'Entity not found'}
            
            if action == 'add':
                if role not in entity.roles:
                    entity.roles = entity.roles + [role]
            elif action == 'remove':
                entity.roles = [r for r in entity.roles if r != role]
            
            session.commit()
            
            return {
                'success': True,
                'entity_id': entity_id,
                'role': role,
                'action': action
            }
            
        finally:
            session.close()
    
    async def _execute_fund_allocation(self, proposal_data: Dict) -> Dict[str, Any]:
        """Execute fund allocation proposal"""
        
        # In production, integrate with treasury contract
        recipient = proposal_data.get('recipient')
        amount = proposal_data.get('amount')
        purpose = proposal_data.get('purpose')
        
        # Simulate execution
        return {
            'success': True,
            'recipient': recipient,
            'amount': amount,
            'purpose': purpose,
            'tx_hash': 'simulated'
        }

class ReputationCalculator:
    """Advanced reputation calculation algorithms"""
    
    def __init__(self):
        self.contribution_weights = {
            'code_contribution': 2.0,
            'governance_participation': 1.5,
            'community_support': 1.0,
            'security_report': 3.0,
            'documentation': 1.2,
            'mentoring': 1.8
        }
    
    def calculate_contribution_score(
        self,
        contribution_type: str,
        quality_score: float,
        impact_score: float
    ) -> float:
        """Calculate reputation points for a contribution"""
        
        base_weight = self.contribution_weights.get(contribution_type, 1.0)
        
        # Quality factor (0-1)
        quality_factor = np.clip(quality_score, 0, 1)
        
        # Impact factor (logarithmic scale)
        impact_factor = np.log1p(impact_score) / np.log(10)
        
        # Calculate points
        points = base_weight * quality_factor * impact_factor
        
        # Apply diminishing returns for repeated contributions
        # In production, track contribution history
        
        return float(points)
    
    def calculate_collaborative_bonus(
        self,
        contributors: List[str],
        contribution_shares: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate bonus for collaborative work"""
        
        if len(contributors) < 2:
            return {c: 0.0 for c in contributors}
        
        # Base collaboration bonus
        collaboration_multiplier = 1.0 + np.log(len(contributors)) / 10
        
        bonuses = {}
        for contributor in contributors:
            share = contribution_shares.get(contributor, 1.0 / len(contributors))
            
            # Bonus based on equitable contribution
            equity_factor = 1.0 - abs(share - 1.0 / len(contributors))
            
            bonuses[contributor] = collaboration_multiplier * equity_factor * 0.5
        
        return bonuses

class DecentralizedArbitration:
    """Dispute resolution system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = create_engine(config['database_url'])
        self.Session = sessionmaker(bind=self.engine)
        
        # Arbitrator selection parameters
        self.min_arbitrator_reputation = config.get('min_arbitrator_reputation', 100)
        self.num_arbitrators = config.get('num_arbitrators', 5)
        
    async def create_dispute(
        self,
        complainant_id: str,
        respondent_id: str,
        dispute_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new dispute for arbitration"""
        
        # Generate dispute ID
        dispute_id = hashlib.sha256(
            f"{complainant_id}{respondent_id}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Select arbitrators
        arbitrators = await self._select_arbitrators(
            exclude=[complainant_id, respondent_id]
        )
        
        if len(arbitrators) < self.num_arbitrators:
            return {
                'success': False,
                'reason': 'Insufficient qualified arbitrators'
            }
        
        # Create dispute record
        dispute = {
            'id': dispute_id,
            'complainant': complainant_id,
            'respondent': respondent_id,
            'arbitrators': arbitrators,
            'status': 'pending',
            'created_at': datetime.utcnow().isoformat(),
            'data': dispute_data
        }
        
        # Store dispute
        # In production, use proper dispute table
        
        # Notify parties
        await self._notify_dispute_parties(dispute)
        
        return {
            'success': True,
            'dispute_id': dispute_id,
            'arbitrators': arbitrators
        }
    
    async def _select_arbitrators(self, exclude: List[str]) -> List[str]:
        """Select qualified arbitrators"""
        
        session = self.Session()
        try:
            # Get eligible arbitrators
            arbitrators = session.query(Entity).filter(
                Entity.reputation_score >= self.min_arbitrator_reputation,
                Entity.roles.contains(GovernanceRole.ARBITRATOR.value),
                ~Entity.id.in_(exclude)
            ).order_by(Entity.reputation_score.desc()).all()
            
            # Random selection weighted by reputation
            if len(arbitrators) <= self.num_arbitrators:
                return [a.id for a in arbitrators]
            
            # Weighted random selection
            weights = [a.reputation_score for a in arbitrators]
            selected_indices = np.random.choice(
                len(arbitrators),
                size=self.num_arbitrators,
                replace=False,
                p=weights/np.sum(weights)
            )
            
            return [arbitrators[i].id for i in selected_indices]
            
        finally:
            session.close()
    
    async def _notify_dispute_parties(self, dispute: Dict):
        """Notify all parties of new dispute"""
        
        # In production, use proper notification service
        logger.info(f"Dispute created: {dispute['id']}")
