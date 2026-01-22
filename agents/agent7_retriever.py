"""
Agent 7: Liquid Memory Retriever
--------------------------------
Implements "WaterWatch" retrieval logic:
1. Hybrid Search (Vector + Metadata)
2. Liquid Memory Formula: Score = (Sim * α) + (Reliability * β) + Decay(Time)
3. FA*IR Reranking (Algorithmic Equity)
"""

import time
import math
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
from qdrant_client import QdrantClient, models

# Reuse config from kernel if possible, but standalone here for clarity
QDRANT_URL = "http://localhost:6333"
COLLECTION = "water_memory"

client = QdrantClient(url=QDRANT_URL)


class LiquidRetriever:
    def __init__(self):
        self.client = client

    def retrieve_with_liquid_memory(
        self,
        query_vector: List[float],
        vector_name: str = "semantic_bind",
        alpha: float = 0.7,   # Vector similarity weight
        beta: float = 0.3,    # Reliability weight
        top_k: int = 10,
        decay_scale: str = "14d",
        decay_factor: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Executes the Liquid Memory Formula.
        """
        
        
        search_result = self.client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            using=vector_name, # Named vector syntax for query_points if available
            with_payload=True,
            with_vectors=True,
            limit=top_k * 2 # Fetch more for MMR filtering
        ).points # query_points returns a QueryResponse object with 'points' attribute
        
        scored_results = []
        now = time.time()
        
        for hit in search_result:
            sim = hit.score
            payload = hit.payload or {}
            
            # 1. Reliability (Default 0.5 if missing)
            reliability = payload.get("reliability_score", 0.5)
            
            # 2. Time Decay (Gaussian)
            ingested_at = payload.get("ingested_at", now)
            age_seconds = max(0, now - ingested_at)
            
            scale_seconds = 14 * 24 * 3600 
            
            decay_val = math.exp(-0.5 * (age_seconds / scale_seconds)**2)
            
            # 3. Final Liquid Score
            final_score = (sim * alpha) + (reliability * beta * sim) + (decay_val * 0.2)
            
            scored_results.append({
                "id": hit.id,
                "score": final_score,
                "components": {"sim": sim, "rel": reliability, "decay": decay_val},
                "payload": payload,
                "vector": hit.vector
            })
            
        # Resort
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results

    def mmr_reranking(
        self,
        query_vector: List[float],
        candidates: List[Dict[str, Any]],
        lambda_mult: float = 0.5,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Maximal Marginal Relevance (MMR) Reranking.
        Balancing Relevance (to query) vs Diversity (from other results).
        """
        if not candidates or k <= 0:
            return []
            
        # Ensure we have vectors
        # Note: candidates[0]["vector"] might be a dict if it was named vector response
        # We need the specific vector used relative to query
        
        # Helper to get vector array
        def get_vec(item):
            v = item.get("vector")
            if isinstance(v, dict):
                # Assume we used semantic_bind, try to extract it
                # If 'semantic_bind' key exists, use it, else use values (single vector)
                return list(v.values())[0] if v else []
            return v if v else []

        # Convert query and candidates to numpy for speed
        q_vec = np.array(query_vector)
        cand_vecs = []
        valid_candidates = []
        
        for c in candidates:
            v = get_vec(c)
            if len(v) == len(q_vec):
                cand_vecs.append(v)
                valid_candidates.append(c)
                
        if not cand_vecs:
            return candidates[:k]
            
        cand_vecs = np.array(cand_vecs)
        
        # Normalize vectors for Cosine Similarity
        norm_q = np.linalg.norm(q_vec)
        q_vec = q_vec / (norm_q + 1e-9)
        
        norm_c = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
        cand_vecs = cand_vecs / (norm_c + 1e-9)
        
        # Similarity to Query (Relevance)
        # Using the liquid score as a proxy for relevance might be better if we want to keep liquid logic?
        # Standard MMR uses raw cosine to query. Let's use raw cosine to query.
        sim_query = np.dot(cand_vecs, q_vec.T)
        
        # Selection Loop
        selected_indices = []
        candidate_indices = list(range(len(valid_candidates)))
        
        while len(selected_indices) < min(k, len(valid_candidates)):
            best_mmr = -float("inf")
            best_idx = -1
            
            for i in candidate_indices:
                # MMR = lambda * Sim(Q, D) - (1-lambda) * max(Sim(D, Selected))
                relevance = sim_query[i]
                
                if not selected_indices:
                    diversity_penalty = 0
                else:
                    # Sim to already selected
                    selected_vecs = cand_vecs[selected_indices]
                    sim_selected = np.dot(selected_vecs, cand_vecs[i].T)
                    diversity_penalty = np.max(sim_selected)
                    
                mmr_score = (lambda_mult * relevance) - ((1 - lambda_mult) * diversity_penalty)
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i
                    
            if best_idx != -1:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
                
        return [valid_candidates[i] for i in selected_indices]

    def fa_ir_reranking(
        self,
        reranked_results: List[Dict[str, Any]],
        protected_key: str = "demographic_tag",
        protected_value: str = "remote_village",
        min_proportion: float = 0.3,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Implements FA*IR (Fairness Utility) Reranking.
        Ensures 'protected_value' appears effectively in the top-k.
        """
        fair_ranking = []
        
        protected_candidates = [r for r in reranked_results if r["payload"].get(protected_key) == protected_value]
        other_candidates = [r for r in reranked_results if r["payload"].get(protected_key) != protected_value]
        
        # Simple Greedy Fairness implementation for Demo
        # We need roughly min_proportion * i items to be protected at rank i
        
        p_ptr = 0
        o_ptr = 0
        p_count = 0
        
        for i in range(1, k + 1):
            # Check if we need to insert protected item to satisfy constraint
            if p_count < i * min_proportion and p_ptr < len(protected_candidates):
                # Force insert protected
                fair_ranking.append(protected_candidates[p_ptr])
                p_ptr += 1
                p_count += 1
            elif o_ptr < len(other_candidates):
                # Insert best other
                # BUT: check if other has higher score? 
                # FA*IR usually takes the better score UNLESS constraint violated.
                
                # Compare heads
                p_score = protected_candidates[p_ptr]["score"] if p_ptr < len(protected_candidates) else -1
                o_score = other_candidates[o_ptr]["score"] if o_ptr < len(other_candidates) else -1
                
                if p_score > o_score:
                    fair_ranking.append(protected_candidates[p_ptr])
                    p_ptr += 1
                    p_count += 1
                else:
                    fair_ranking.append(other_candidates[o_ptr])
                    o_ptr += 1
            elif p_ptr < len(protected_candidates):
                 # Run out of others
                fair_ranking.append(protected_candidates[p_ptr])
                p_ptr += 1
                p_count += 1
        
        return fair_ranking


    def unified_search(
        self,
        query_vector: List[float],
        vector_name: str = "semantic_bind",
        top_k: int = 10,
        enable_fairness: bool = True,
        enable_mmr: bool = False,
        mmr_lambda: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Unified utility function:
        1. Liquid Memory Vector Search (Retrieves 2*top_k)
        2. MMR Reranking (Diversity) -> Reduces to top_k
        3. Fairness Reranking (Equity) -> Reorders top_k
        """
        # 1. Retrieve & Score (Liquid Memory)
        # Note: We fetch more candidates if MMR is enabled to allow filtering
        fetch_k = top_k * 3 if enable_mmr else top_k
        
        raw_results = self.retrieve_with_liquid_memory(
            query_vector=query_vector,
            vector_name=vector_name,
            top_k=fetch_k # Retrieve more candidates
        )
        
        current_results = raw_results

        # 2. MMR Diversity Reranking
        if enable_mmr:
            current_results = self.mmr_reranking(
                query_vector=query_vector,
                candidates=current_results,
                lambda_mult=mmr_lambda,
                k=top_k # Reduce to desired k
            )
        else:
             # Just slice to k if no MMR
             current_results = current_results[:top_k]

        # 3. Fairness Reranking
        if enable_fairness:
            return self.fa_ir_reranking(current_results, k=top_k)
            
        return current_results

# =========================================================
# DEMO / TEST RUNNER
# =========================================================
if __name__ == "__main__":
    
    print("\n[AGENT 7] Initializing Liquid Retriever...")
    retriever = LiquidRetriever()
    
    # 1. Create a Fake Query Vector (Random 512d)
    # Using simple random vector as per memory.py schema (no magic context needed for query vector itself)
    query_vec = np.random.rand(512).tolist()
    
    print("[AGENT 7] Executing Unified Search (+MMR)...")
    try:
        # SINGLE UNIFIED CALL
        results = retriever.unified_search(
            query_vector=query_vec,
            top_k=5,
            enable_fairness=True,
            enable_mmr=True,
            mmr_lambda=0.5
        )
        
        print(f"\nRETRIEVED ({len(results)}):")
        for i, res in enumerate(results):
            p = res['payload']
            c = res.get('components', {'sim': 0, 'rel': 0, 'decay': 0})
            
            # Kernel flattens context, so keys are at root
            geo = p.get('geohash', p.get('context', {}).get('geohash', 'N/A'))
            tag = p.get('demographic_tag', p.get('context', {}).get('demographic_tag', 'N/A'))
            
            print(f"{i+1}. {p.get('percept_id')} | Score: {res['score']:.4f}")
            print(f"   (Sim: {c['sim']:.2f}, Rel: {c['rel']}, Decay: {c['decay']:.2f})")
            print(f"   Geo: {geo} | Tag: {tag}")
            
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()


