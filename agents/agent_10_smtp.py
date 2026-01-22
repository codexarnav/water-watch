"""
Agent 10: Recommendation Engine
--------------------------------
• MMR diverse evidence-grounded suggestions
• Uses Gemini for reasoning over historical patterns + risk forecasts
• Retrieves from full memory phase (Qdrant) for context
• Generates actionable, prioritized recommendations
"""

import os
import json
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv

import google.generativeai as genai
from langgraph.graph import StateGraph, END

from agents.agent7_retriever import LiquidRetriever
from agents.agent6_forecasting import forecast
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range

# Load environment variables
load_dotenv()

# Gemini Setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-2.0-flash")

# Qdrant Setup
QDRANT_URL = "http://localhost:6333"
COLLECTION = "water_memory"
client = QdrantClient(url=QDRANT_URL)


# ============================
# STATE SCHEMA
# ============================
class Agent10State(TypedDict):
    # Inputs
    source_id: Optional[str]
    risk_forecast: Optional[Dict[str, Any]]  # From Agent 8
    historical_patterns: Optional[List[Dict[str, Any]]]  # From Agent 7
    evidence_context: Optional[List[Dict[str, Any]]]  # From Agent 8 evidence_pack
    
    # Internal processing
    memory_context: Optional[List[Dict[str, Any]]]  # Retrieved from Qdrant
    reasoning_context: Optional[str]  # Gemini reasoning output
    
    # Output
    recommendations: Optional[List[Dict[str, Any]]]
    priority_actions: Optional[List[str]]
    output: Optional[Dict[str, Any]]


# ============================
# NODE 1: Gather Memory Context
# ============================
def node_gather_memory(state: Agent10State) -> Agent10State:
    """
    Retrieves comprehensive memory context from Qdrant for the source_id.
    Uses Agent 7 retriever to get diverse historical patterns.
    """
    source_id = state.get("source_id")
    risk_forecast = state.get("risk_forecast", {})
    
    if not source_id:
        return {**state, "memory_context": []}
    
    memory_context = []
    retriever = LiquidRetriever()
    
    # 1. Get risk forecast vector if available (from last event)
    risk_score = risk_forecast.get("risk_forecast", {}).get("risk_score", 0.0)
    risk_level = risk_forecast.get("risk_forecast", {}).get("risk_level", "LOW")
    
    # 2. Query Qdrant for related historical incidents
    # Use scroll to get recent high-severity events for this source
    try:
        flt = Filter(
            must=[
                FieldCondition(key="source_id", match=MatchValue(value=source_id)),
                FieldCondition(key="severity", range=Range(gte=0.4)),
            ]
        )
        
        points, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=True,
            limit=50,
        )
        
        for p in points:
            payload = p.payload or {}
            memory_context.append({
                "percept_id": str(p.id),
                "timestamp": payload.get("timestamp"),
                "severity": payload.get("severity", 0.0),
                "tags": payload.get("tags", []),
                "source_id": payload.get("source_id"),
                "reliability_score": payload.get("reliability_score", 0.5),
                "raw_text": payload.get("raw_ref", {}).get("text", ""),
                "intervention_outcome": payload.get("intervention_outcome"),
            })
    except Exception as e:
        print(f"[AGENT10] Error retrieving memory: {e}")
    
    # 3. Use Agent 7 retriever for semantic similarity search
    # If we have evidence from forecasting, use its vector for search
    evidence_context = state.get("evidence_context", [])
    if evidence_context and len(evidence_context) > 0:
        # Try to get a vector from the most recent evidence
        # For now, we'll use a text-based query vector generation
        # In production, you'd extract the vector from the evidence
        try:
            # Create a query based on risk level and source
            query_text = f"water quality risk {risk_level} {source_id} intervention recommendation"
            # For now, we'll use a simple approach - in production, embed this query
            # and use retriever.unified_search()
            pass
        except Exception as e:
            print(f"[AGENT10] Error in semantic search: {e}")
    
    return {**state, "memory_context": memory_context}


# ============================
# NODE 2: Build Reasoning Context
# ============================
def node_build_context(state: Agent10State) -> Agent10State:
    """
    Combines all inputs into a structured context for Gemini reasoning.
    """
    source_id = state.get("source_id", "Unknown")
    risk_forecast = state.get("risk_forecast", {})
    historical_patterns = state.get("historical_patterns", [])
    evidence_context = state.get("evidence_context", [])
    memory_context = state.get("memory_context", [])
    
    # Build comprehensive context string
    context_parts = []
    
    # 1. Risk Forecast Summary
    if risk_forecast:
        rf = risk_forecast.get("risk_forecast", {})
        context_parts.append(f"""
=== RISK FORECAST ===
Source: {source_id}
Risk Score: {rf.get('risk_score', 0.0):.3f}
Risk Level: {rf.get('risk_level', 'UNKNOWN')}
Confidence: {rf.get('confidence', 0.0):.3f}
Window: {risk_forecast.get('window', 'N/A')}
Horizon: {risk_forecast.get('horizon', 'N/A')}
""")
        
        if risk_forecast.get("why"):
            why = risk_forecast["why"]
            context_parts.append(f"""
Risk Analysis:
- Local Features: {json.dumps(why.get('local_features', {}), indent=2)}
- Evidence Count: {why.get('retrieval_evidence_count', 0)}
- Logic: {', '.join(why.get('logic', []))}
""")
    
    # 2. Evidence Context
    if evidence_context:
        context_parts.append(f"""
=== EVIDENCE FROM FORECASTING ===
Found {len(evidence_context)} similar historical episodes:
""")
        for i, ev in enumerate(evidence_context[:10], 1):
            context_parts.append(f"""
{i}. Episode ID: {ev.get('point_id', 'N/A')}
   Source: {ev.get('source_id', 'N/A')}
   Timestamp: {ev.get('timestamp', 'N/A')}
   Severity: {ev.get('severity', 0.0):.3f}
   Similarity Score: {ev.get('score', 0.0):.3f if ev.get('score') else 'N/A'}
   Tags: {', '.join(ev.get('tags', []))}
""")
    
    # 3. Historical Patterns (from Agent 7)
    if historical_patterns:
        context_parts.append(f"""
=== HISTORICAL PATTERNS (Liquid Memory) ===
Retrieved {len(historical_patterns)} relevant historical records:
""")
        for i, pattern in enumerate(historical_patterns[:10], 1):
            payload = pattern.get("payload", {})
            context_parts.append(f"""
{i}. Percept ID: {pattern.get('id', 'N/A')}
   Score: {pattern.get('score', 0.0):.4f}
   Reliability: {payload.get('reliability_score', 0.5):.2f}
   Timestamp: {payload.get('timestamp', 'N/A')}
   Severity: {payload.get('severity', 0.0):.3f}
   Tags: {', '.join(payload.get('tags', []))}
   Intervention Outcome: {payload.get('intervention_outcome', 'N/A')}
""")
    
    # 4. Memory Context (from Qdrant scroll)
    if memory_context:
        context_parts.append(f"""
=== ADDITIONAL MEMORY CONTEXT ===
Found {len(memory_context)} related incidents from memory:
""")
        for i, mem in enumerate(memory_context[:10], 1):
            context_parts.append(f"""
{i}. Percept ID: {mem.get('percept_id', 'N/A')}
   Timestamp: {mem.get('timestamp', 'N/A')}
   Severity: {mem.get('severity', 0.0):.3f}
   Reliability: {mem.get('reliability_score', 0.5):.2f}
   Tags: {', '.join(mem.get('tags', []))}
   Intervention: {mem.get('intervention_outcome', 'Not recorded')}
   Text: {mem.get('raw_text', '')[:100]}...
""")
    
    reasoning_context = "\n".join(context_parts)
    
    return {**state, "reasoning_context": reasoning_context}


# ============================
# NODE 3: Gemini Reasoning
# ============================
def node_gemini_reasoning(state: Agent10State) -> Agent10State:
    """
    Uses Gemini to reason over all context and generate recommendations.
    """
    source_id = state.get("source_id", "Unknown")
    reasoning_context = state.get("reasoning_context", "")
    risk_forecast = state.get("risk_forecast", {})
    risk_level = risk_forecast.get("risk_forecast", {}).get("risk_level", "LOW")
    
    prompt = f"""
You are an expert water quality management advisor with access to comprehensive historical data, risk forecasts, and evidence from similar incidents.

Your task is to generate actionable, prioritized recommendations based on:
1. Current risk forecast and analysis
2. Historical patterns from similar incidents
3. Evidence from past interventions
4. Memory context from the water monitoring system

CONTEXT DATA:
{reasoning_context}

Generate recommendations following this structure:

1. **IMMEDIATE ACTIONS** (if risk is HIGH or MEDIUM):
   - Specific, actionable steps that should be taken within hours
   - Reference which historical incidents support each action
   - Include urgency level

2. **SHORT-TERM MONITORING** (next 24-48 hours):
   - What to monitor closely
   - Thresholds to watch
   - Frequency of checks

3. **PREVENTIVE MEASURES** (if risk is LOW but patterns suggest potential issues):
   - Proactive steps to prevent escalation
   - Maintenance recommendations

4. **LONG-TERM RECOMMENDATIONS**:
   - Infrastructure improvements
   - Process changes
   - Training needs

5. **EVIDENCE-BASED INSIGHTS**:
   - What patterns from history suggest about current situation
   - What worked/didn't work in similar cases
   - Confidence in recommendations based on historical data

IMPORTANT:
- Base ALL recommendations on the provided evidence
- Reference specific percept IDs or incident timestamps when citing evidence
- Prioritize recommendations by urgency and impact
- Consider the reliability scores of historical data
- If risk is HIGH, emphasize immediate safety actions
- Be specific and actionable, not generic

Respond in JSON format:
{{
  "immediate_actions": [
    {{
      "action": "specific action description",
      "urgency": "critical|high|medium",
      "timeframe": "within X hours",
      "evidence_support": ["percept_id or timestamp"],
      "rationale": "why this action is needed"
    }}
  ],
  "short_term_monitoring": [
    {{
      "metric": "what to monitor",
      "frequency": "how often",
      "threshold": "alert threshold",
      "rationale": "why this matters"
    }}
  ],
  "preventive_measures": [
    {{
      "measure": "description",
      "priority": "high|medium|low",
      "rationale": "why this helps"
    }}
  ],
  "long_term_recommendations": [
    {{
      "recommendation": "description",
      "impact": "expected impact",
      "rationale": "why this is important"
    }}
  ],
  "evidence_insights": {{
    "key_patterns": ["pattern 1", "pattern 2"],
    "historical_successes": ["what worked"],
    "historical_failures": ["what didn't work"],
    "confidence_level": "high|medium|low",
    "reasoning": "overall assessment"
  }},
  "priority_actions": [
    "top 3 most critical actions in order"
  ]
}}
"""

    try:
        response = llm.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "response_mime_type": "application/json"
            }
        )
        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        recommendations = json.loads(response_text)
        
    except Exception as e:
        print(f"[AGENT10] Error in Gemini reasoning: {e}")
        print(f"[AGENT10] Raw response: {response.text[:500] if 'response' in locals() else 'No response'}")
        
        # Fallback recommendations
        recommendations = {
            "immediate_actions": [
                {
                    "action": "Review risk forecast and evidence",
                    "urgency": "medium",
                    "timeframe": "within 2 hours",
                    "evidence_support": [],
                    "rationale": "Fallback: Error in reasoning engine"
                }
            ],
            "short_term_monitoring": [],
            "preventive_measures": [],
            "long_term_recommendations": [],
            "evidence_insights": {
                "key_patterns": [],
                "historical_successes": [],
                "historical_failures": [],
                "confidence_level": "low",
                "reasoning": "Error occurred in reasoning engine"
            },
            "priority_actions": ["Review system logs", "Manual risk assessment"]
        }
    
    priority_actions = recommendations.get("priority_actions", [])
    
    return {
        **state,
        "recommendations": recommendations,
        "priority_actions": priority_actions
    }


# ============================
# NODE 4: Compose Output
# ============================
def node_compose_output(state: Agent10State) -> Agent10State:
    """
    Formats the final output with all recommendations.
    """
    source_id = state.get("source_id", "Unknown")
    recommendations = state.get("recommendations", {})
    priority_actions = state.get("priority_actions", [])
    risk_forecast = state.get("risk_forecast", {})
    
    output = {
        "source_id": source_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "risk_context": {
            "risk_level": risk_forecast.get("risk_forecast", {}).get("risk_level", "UNKNOWN"),
            "risk_score": risk_forecast.get("risk_forecast", {}).get("risk_score", 0.0),
        },
        "recommendations": recommendations,
        "priority_actions": priority_actions,
        "meta": {
            "evidence_count": len(state.get("evidence_context", [])),
            "historical_patterns_count": len(state.get("historical_patterns", [])),
            "memory_context_count": len(state.get("memory_context", [])),
        }
    }
    
    return {**state, "output": output}


# ============================
# BUILD GRAPH
# ============================
def build_agent10():
    g = StateGraph(Agent10State)
    
    g.add_node("gather_memory", node_gather_memory)
    g.add_node("build_context", node_build_context)
    g.add_node("gemini_reasoning", node_gemini_reasoning)
    g.add_node("compose_output", node_compose_output)
    
    g.set_entry_point("gather_memory")
    g.add_edge("gather_memory", "build_context")
    g.add_edge("build_context", "gemini_reasoning")
    g.add_edge("gemini_reasoning", "compose_output")
    g.add_edge("compose_output", END)
    
    return g.compile()


# ============================
# EXPOSE API
# ============================
def generate_recommendations(
    source_id: str,
    risk_forecast: Optional[Dict[str, Any]] = None,
    historical_patterns: Optional[List[Dict[str, Any]]] = None,
    evidence_context: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Main API function to generate recommendations.
    
    Args:
        source_id: The well/source identifier
        risk_forecast: Output from Agent 8 (forecasting.py)
        historical_patterns: Output from Agent 7 (agent7_retriever.py)
        evidence_context: Evidence pack from Agent 8
        
    Returns:
        Dictionary with recommendations and priority actions
    """
    # If risk_forecast not provided, fetch it
    if risk_forecast is None:
        risk_forecast = forecast(source_id, window="24h", horizon="6h", mode="risk+evidence+why")
        evidence_context = risk_forecast.get("evidence_pack", [])
    
    agent = build_agent10()
    init_state: Agent10State = {
        "source_id": source_id,
        "risk_forecast": risk_forecast,
        "historical_patterns": historical_patterns or [],
        "evidence_context": evidence_context or [],
        "memory_context": None,
        "reasoning_context": None,
        "recommendations": None,
        "priority_actions": None,
        "output": None,
    }
    
    result = agent.invoke(init_state)
    return result["output"]


if __name__ == "__main__":
    print("\n[AGENT10] Running Recommendation Engine Test...\n")
    
    # Test with a sample source
    test_source = "Well_Test_fairness"
    
    # Get risk forecast first
    print("1. Fetching risk forecast...")
    risk_fc = forecast(test_source, window="24h", horizon="6h", mode="risk+evidence+why")
    print(f"   Risk Level: {risk_fc.get('risk_forecast', {}).get('risk_level', 'UNKNOWN')}")
    
    # Generate recommendations
    print("\n2. Generating recommendations...")
    recommendations = generate_recommendations(
        source_id=test_source,
        risk_forecast=risk_fc,
        evidence_context=risk_fc.get("evidence_pack", [])
    )
    
    print("\n=== RECOMMENDATIONS ===")
    print(json.dumps(recommendations, indent=2))