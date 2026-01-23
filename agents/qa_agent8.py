
from typing import TypedDict, List, Dict, Any, Optional
import numpy as np
import json

from langgraph.graph import StateGraph, END

from agents.agent7_retriever import LiquidRetriever

import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-2.0-flash")

# =====================================================
# Geospatial Utilities
# =====================================================
import geohash2


def decode_geohash(gh: Optional[str]) -> Optional[Dict[str, float]]:
    """Convert geohash to lat/lon coordinates."""
    if not gh:
        return None
    try:
        lat, lon = geohash2.decode(gh)
        return {"lat": round(float(lat), 6), "lon": round(float(lon), 6)}
    except Exception:
        return None


# =====================================================
# LangGraph State
# =====================================================
class AgentState(TypedDict):
    query: str
    query_vector: List[float]
    policy: Dict[str, Any]
    retrieved: List[Dict[str, Any]]
    evidence_by_modality: Dict[str, List[Dict]]
    answer: Dict[str, Any]


# =====================================================
# Node 1: Retrieval Policy Router (JSON enforced)
# =====================================================
def extract_json_from_response(text: str) -> dict:
    """
    Extract JSON from Gemini response, handling markdown code blocks.
    """
    import re
    
    # Try to find JSON in markdown code block
    code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    
    if match:
        json_text = match.group(1)
    else:
        # Try to find raw JSON
        json_pattern = r"\{.*\}"
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            json_text = match.group(0)
        else:
            raise ValueError("No JSON found in response")
    
    return json.loads(json_text)


def policy_router(state: AgentState) -> AgentState:
    query = state["query"]

    prompt = f"""
You are a retrieval policy router for a water intelligence system.

Analyze the query and determine the retrieval strategy.

RULES:
- Use "liquid" mode when the query involves: trust, time-sensitive data, risk assessment, safety concerns, fairness, impact analysis, or recommendations
- Use "normal" mode for: factual lookups, descriptive queries, historical data retrieval

QUERY: "{query}"

Respond with ONLY this JSON (no markdown, no explanation):
{{
  "retrieval_mode": "normal" | "liquid",
  "enable_mmr": true,
  "enable_fairness": true | false,
  "reason": "one-sentence explanation"
}}
"""

    response = llm.generate_content(
        prompt,
        generation_config={
            "temperature": 0.0,
            "response_mime_type": "application/json"
        }
    )

    # -------------------------
    # ROBUST JSON EXTRACTION
    # -------------------------
    try:
        # Try structured extraction first
        policy = extract_json_from_response(response.text)
        
        # Validate schema
        assert policy.get("retrieval_mode") in ("normal", "liquid"), "Invalid retrieval_mode"
        assert "reason" in policy, "Missing reason field"
        assert "enable_mmr" in policy, "Missing enable_mmr field"
        assert "enable_fairness" in policy, "Missing enable_fairness field"
        
    except Exception as e:
        # LangGraph-safe fallback (never crash)
        print(f"[WARNING] Policy router failed to parse JSON: {e}")
        print(f"[WARNING] Raw response: {response.text[:200]}")
        
        policy = {
            "retrieval_mode": "normal",
            "enable_mmr": True,
            "enable_fairness": False,
            "reason": "Fallback: JSON parsing failed"
        }

    state["policy"] = policy
    return state


# =====================================================
# Node 2: Memory Retrieval
# =====================================================
def retrieve_memory(state: AgentState) -> AgentState:
    retriever = LiquidRetriever()

    policy = state["policy"]
    query_vector = state["query_vector"]

    if policy["retrieval_mode"] == "liquid":
        results = retriever.unified_search(
            query_vector=query_vector,
            top_k=8,
            enable_mmr=policy.get("enable_mmr", True),
            enable_fairness=policy.get("enable_fairness", False)
        )
    else:
        results = retriever.unified_search(
            query_vector=query_vector,
            top_k=8,
            enable_mmr=True,
            enable_fairness=False
        )

    state["retrieved"] = results
    return state


# =====================================================
# Node 3: Evidence Aggregator (Multimodal-Aware)
# =====================================================
def aggregate_evidence(state: AgentState) -> AgentState:
    """
    Group retrieved results by modality and enrich context.
    Preserves URIs and decodes geohash to coordinates.
    """
    results = state["retrieved"]
    
    grouped = {
        "text": [],
        "image": [],
        "video": [],
        "audio": []
    }
    
    for r in results:
        p = r["payload"]
        modality = p.get("modality", "text")
        
        # Decode geohash to coordinates
        geo_coords = decode_geohash(p.get("geohash"))
        
        evidence_item = {
            "percept_id": p.get("percept_id"),
            "reliability": p.get("reliability_score", 0.5),
            "timestamp": p.get("timestamp"),
            "source_id": p.get("source_id"),
            "geohash": p.get("geohash"),
            "coordinates": geo_coords,
            "raw_ref": p.get("raw_ref", {}),
            "score": r.get("score", 0.0)
        }
        
        grouped[modality].append(evidence_item)
    
    state["evidence_by_modality"] = grouped
    return state


# =====================================================
# Node 4: Multimodal Answer Synthesis
# =====================================================
def synthesize_answer(state: AgentState) -> AgentState:
    """
    Synthesize multimodal response:
    - Text evidence → LLM synthesis with citations
    - Media evidence → Direct URIs with metadata
    """
    query = state["query"]
    evidence = state["evidence_by_modality"]
    
    response = {
        "query": query,
        "text_answer": None,
        "media_evidence": [],
        "metadata": {
            "policy": state["policy"],
            "total_sources": sum(len(v) for v in evidence.values()),
            "sources_by_modality": {k: len(v) for k, v in evidence.items()}
        }
    }
    
    # =====================================================
    # 1. SYNTHESIZE TEXT EVIDENCE (if any)
    # =====================================================
    if evidence["text"]:
        text_blocks = []
        for item in evidence["text"]:
            loc_str = ""
            if item["coordinates"]:
                loc_str = f"[Location: {item['coordinates']['lat']}, {item['coordinates']['lon']}]"
            
            block = f"""
[Percept ID]: {item['percept_id']}
[Reliability]: {item['reliability']}
[Time]: {item['timestamp']}
{loc_str}
[Content]: {item['raw_ref'].get('text', 'N/A')}
"""
            text_blocks.append(block)
        
        evidence_text = "\n---\n".join(text_blocks)
        
        prompt = f"""
You are an evidence-grounded water intelligence assistant.

STRICT RULES:
- Use ONLY the evidence below
- Every factual claim MUST cite a Percept ID using [percept_id]
- Do NOT assume geography if Location is not provided
- Prefer higher reliability evidence
- Be concise but complete

Evidence:
{evidence_text}

Question:
{query}

Answer format:
- Clear paragraphs
- Each paragraph ends with citations like [percept_id]
"""
        
        llm_response = llm.generate_content(
            prompt,
            generation_config={"temperature": 0.2}
        )
        
        response["text_answer"] = llm_response.text
    
    # =====================================================
    # 2. COLLECT MEDIA EVIDENCE (images, videos, audio)
    # =====================================================
    for modality in ["image", "video", "audio"]:
        for item in evidence[modality]:
            # Determine URI key based on modality
            uri_key = f"{modality}_uri"
            
            media_item = {
                "type": modality,
                "percept_id": item["percept_id"],
                "uri": item["raw_ref"].get(uri_key),
                "reliability": item["reliability"],
                "timestamp": item["timestamp"],
                "source_id": item["source_id"],
                "geohash": item["geohash"],
                "coordinates": item["coordinates"],
                "score": item["score"]
            }
            
            # Only add if URI exists
            if media_item["uri"]:
                response["media_evidence"].append(media_item)
    
    # Sort media by score (highest first)
    response["media_evidence"].sort(key=lambda x: x["score"], reverse=True)
    
    state["answer"] = response
    return state


# =====================================================
# LangGraph Assembly
# =====================================================
graph = StateGraph(AgentState)

graph.add_node("policy_router", policy_router)
graph.add_node("retrieve_memory", retrieve_memory)
graph.add_node("aggregate_evidence", aggregate_evidence)
graph.add_node("synthesize_answer", synthesize_answer)

graph.set_entry_point("policy_router")
graph.add_edge("policy_router", "retrieve_memory")
graph.add_edge("retrieve_memory", "aggregate_evidence")
graph.add_edge("aggregate_evidence", "synthesize_answer")
graph.add_edge("synthesize_answer", END)

agent = graph.compile()


# =====================================================
# Demo Runner
# =====================================================
if __name__ == "__main__":
    query = "most reported problme related to dirty water"

    # Replace with real embedding
    query_vector = np.random.rand(512).tolist()

    result = agent.invoke({
        "query": query,
        "query_vector": query_vector
    })

    answer = result["answer"]
    
    print("\n" + "="*60)
    print("HYDRO-SEMANTIC QA AGENT (Multimodal)")
    print("="*60)
    
    print("\n--- RETRIEVAL POLICY ---")
    print(json.dumps(result["policy"], indent=2))
    
    print("\n--- METADATA ---")
    print(json.dumps(answer["metadata"], indent=2))
    
    if answer.get("text_answer"):
        print("\n--- TEXT ANSWER ---")
        print(answer["text_answer"])
    
    if answer.get("media_evidence"):
        print("\n--- MEDIA EVIDENCE ---")
        for i, media in enumerate(answer["media_evidence"], 1):
            print(f"\n[{i}] {media['type'].upper()}")
            print(f"    URI: {media['uri']}")
            print(f"    Percept ID: {media['percept_id']}")
            print(f"    Reliability: {media['reliability']:.2f}")
            if media['coordinates']:
                print(f"    Location: ({media['coordinates']['lat']}, {media['coordinates']['lon']})")
            print(f"    Source: {media['source_id']}")
            print(f"    Timestamp: {media['timestamp']}")
            print(f"    Score: {media['score']:.4f}")
    
    print("\n" + "="*60)
    print("FULL JSON RESPONSE")
    print("="*60)
    print(json.dumps(answer, indent=2))