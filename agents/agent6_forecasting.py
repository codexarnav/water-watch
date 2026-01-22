
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict, Literal, Tuple

import numpy as np
from langgraph.graph import StateGraph, END

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

QDRANT_URL = "http://localhost:6333"
COLLECTION = "water_memory"
VECTOR_NAME = "semantic_bind"

client = QdrantClient(url=QDRANT_URL)


DEFAULT_WINDOW = "24h"
DEFAULT_HORIZON = "6h"
DEFAULT_MODE = "risk+evidence+why"

ForecastMode = Literal["risk_only", "risk+evidence", "risk+evidence+why"]

CACHE_TTL_SEC = 600
SIM_SEARCH_GATE = 0.55
SIM_TOPK = 30
SIM_SEVERITY_MIN = 0.4

SENSOR_PRODUCER = "agent2_stm_spike"  # recommended filter if present


# ============================
# UTILITIES
# ============================
def now_epoch() -> float:
    return time.time()


def parse_duration(d: str) -> int:
    d = d.strip().lower()
    if d.endswith("h"):
        return int(float(d[:-1]) * 3600)
    if d.endswith("m"):
        return int(float(d[:-1]) * 60)
    if d.endswith("s"):
        return int(float(d[:-1]))
    raise ValueError(f"Invalid duration: {d}")


def iso_to_epoch(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def risk_level(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"


# ============================
# CACHE
# ============================
_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def cache_get(k: str) -> Optional[Dict[str, Any]]:
    row = _CACHE.get(k)
    if not row:
        return None
    ts, val = row
    if now_epoch() - ts > CACHE_TTL_SEC:
        return None
    return val


def cache_set(k: str, v: Dict[str, Any]) -> None:
    _CACHE[k] = (now_epoch(), v)


# ============================
# STATE
# ============================
class Agent8State(TypedDict):
    request: Dict[str, Any]

    # query builder
    source_id: Optional[str]
    window_secs: Optional[int]
    horizon_secs: Optional[int]
    mode: Optional[ForecastMode]
    cache_key: Optional[str]
    cached_result: Optional[Dict[str, Any]]

    # Set A
    setA_payloads: Optional[List[Dict[str, Any]]]
    last_event_vector: Optional[List[float]]

    # base forecast
    base_risk: Optional[float]
    features: Optional[Dict[str, Any]]

    # Set B
    setB_evidence: Optional[List[Dict[str, Any]]]

    # final
    final_risk: Optional[float]
    confidence: Optional[float]

    output: Optional[Dict[str, Any]]


# ============================
# NODE 1: Query Builder + Cache
# ============================
def node_query_builder(state: Agent8State) -> Agent8State:
    req = state["request"]

    source_id = req.get("source_id")
    if not source_id:
        raise ValueError("source_id is required")

    window = req.get("window", DEFAULT_WINDOW)
    horizon = req.get("horizon", DEFAULT_HORIZON)
    mode = req.get("mode", DEFAULT_MODE)

    if mode not in ["risk_only", "risk+evidence", "risk+evidence+why"]:
        mode = DEFAULT_MODE

    window_secs = parse_duration(window)
    horizon_secs = parse_duration(horizon)

    cache_key = f"{source_id}|{window_secs}|{horizon_secs}|{mode}"
    cached = cache_get(cache_key)

    return {
        **state,
        "source_id": source_id,
        "window_secs": window_secs,
        "horizon_secs": horizon_secs,
        "mode": mode,  # type: ignore
        "cache_key": cache_key,
        "cached_result": cached,
    }


def router_cache(state: Agent8State) -> str:
    return "cached" if state.get("cached_result") else "fetch_setA"


def node_return_cached(state: Agent8State) -> Agent8State:
    return {**state, "output": state["cached_result"]}


# ============================
# NODE 2: Fetch Set A (scroll local time-series)
# ============================
def node_fetch_setA(state: Agent8State) -> Agent8State:
    source_id = state.get("source_id")
    window_secs = state.get("window_secs")

    if not source_id or not window_secs:
        return state

    cutoff = now_epoch() - window_secs

    # IMPORTANT: sensor events are stored as modality="text" in your kernel
    must_conditions = [
        FieldCondition(key="source_id", match=MatchValue(value=source_id)),
        FieldCondition(key="modality", match=MatchValue(value="text")),
    ]

    # producer filter (only if you stored producer in context)
    # If it's missing, scroll will still work and we filter later.
    must_conditions.append(
        FieldCondition(key="producer", match=MatchValue(value=SENSOR_PRODUCER))
    )

    flt = Filter(must=must_conditions)

    payloads: List[Dict[str, Any]] = []
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=True,     # we need vector for last event
            limit=256,
            offset=next_offset,
        )

        for p in points:
            payload = p.payload or {}
            ts = payload.get("timestamp")
            if not ts:
                continue

            try:
                ep = iso_to_epoch(ts)
            except:
                continue

            if ep < cutoff:
                continue

            # attach vector of semantic_bind if exists
            vec = None
            if hasattr(p, "vector") and p.vector:
                # p.vector can be dict of vectors or raw list
                if isinstance(p.vector, dict):
                    vec = p.vector.get(VECTOR_NAME)
                elif isinstance(p.vector, list):
                    vec = p.vector

            payload["_vector"] = vec
            payloads.append(payload)

        if next_offset is None:
            break
        if len(payloads) >= 5000:
            break

    # fallback if producer field doesn't exist in stored points
    # If empty, retry without producer filter
    if len(payloads) == 0:
        flt = Filter(must=[
            FieldCondition(key="source_id", match=MatchValue(value=source_id)),
            FieldCondition(key="modality", match=MatchValue(value="text")),
        ])

        points, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=True,
            limit=512,
        )

        for p in points:
            payload = p.payload or {}
            ts = payload.get("timestamp")
            if not ts:
                continue
            try:
                if iso_to_epoch(ts) >= cutoff:
                    vec = None
                    if isinstance(p.vector, dict):
                        vec = p.vector.get(VECTOR_NAME)
                    payload["_vector"] = vec
                    payloads.append(payload)
            except:
                continue

    # sort by time
    payloads.sort(key=lambda x: x.get("timestamp", ""))

    last_vec = None
    if payloads:
        last_vec = payloads[-1].get("_vector")

    return {**state, "setA_payloads": payloads, "last_event_vector": last_vec}


# ============================
# NODE 3: Base Risk (fast)
# ============================
def node_base_risk(state: Agent8State) -> Agent8State:
    pts = state.get("setA_payloads") or []

    if len(pts) < 5:
        return {
            **state,
            "base_risk": 0.12,
            "features": {"note": "Not enough sensor points in window"}
        }

    times = []
    sev = []

    for p in pts:
        ts = p.get("timestamp")
        if not ts:
            continue
        try:
            times.append(iso_to_epoch(ts))
            sev.append(float(p.get("severity", 0.0)))
        except:
            continue

    if len(sev) < 5:
        return {
            **state,
            "base_risk": 0.12,
            "features": {"note": "Bad timestamps / severity missing"}
        }

    times_np = np.array(times)
    sev_np = np.array(sev)

    # spike density
    spike_count = int(np.sum(sev_np > 0.5))
    duration_hr = max(1e-6, (times_np[-1] - times_np[0]) / 3600.0)
    spike_density = spike_count / duration_hr

    # recency
    if spike_count > 0:
        last_spike_time = times_np[np.where(sev_np > 0.5)[0][-1]]
    else:
        last_spike_time = times_np[-1]
    recency_sec = max(0.0, now_epoch() - last_spike_time)

    # volatility
    volatility = float(np.std(sev_np))

    # slope
    x = times_np - times_np[0]
    slope = float(np.polyfit(x, sev_np, 1)[0])  # per second
    slope_per_hr = slope * 3600.0

    # EW anomaly rate
    tau = 3600.0
    w = np.exp(-(times_np[-1] - times_np) / tau)
    ew_rate = float(np.sum(w * sev_np) / np.sum(w))

    base = (
        0.25 * clamp(ew_rate) +
        0.20 * clamp(spike_density / 5.0) +
        0.20 * clamp(volatility / 0.4) +
        0.20 * clamp((slope_per_hr + 0.2) / 0.6) +
        0.15 * clamp(max(0, 3600 - recency_sec) / 3600)
    )

    base = clamp(base)

    return {
        **state,
        "base_risk": base,
        "features": {
            "n_points": len(sev_np),
            "spike_density_per_hr": spike_density,
            "recency_sec": recency_sec,
            "volatility": volatility,
            "trend_slope_per_hr": slope_per_hr,
            "ew_severity_rate": ew_rate,
        }
    }


def router_need_similar(state: Agent8State) -> str:
    mode = state.get("mode")
    base = state.get("base_risk") or 0.0

    if mode == "risk_only":
        return "compose"
    if base < SIM_SEARCH_GATE:
        return "compose"

    # Need last vector for semantic search
    if not state.get("last_event_vector"):
        return "compose"

    return "fetch_setB"


# ============================
# NODE 4: Fetch Set B (similar episodes)
# ============================
def node_fetch_setB(state: Agent8State) -> Agent8State:
    vec = state.get("last_event_vector")
    if not vec:
        return {**state, "setB_evidence": []}

    # filter: high severity evidence
    flt = Filter(
        must=[
            FieldCondition(key="severity", range=Range(gte=SIM_SEVERITY_MIN)),
        ]
    )

    hits = client.search(
        collection_name=COLLECTION,
        query_vector=(VECTOR_NAME, vec),
        limit=SIM_TOPK,
        query_filter=flt,
        with_payload=True,
        with_vectors=False,
    )

    evidence = []
    for h in hits:
        payload = h.payload or {}
        evidence.append({
            "point_id": str(h.id),
            "source_id": payload.get("source_id"),
            "timestamp": payload.get("timestamp"),
            "severity": payload.get("severity"),
            "tags": payload.get("tags", []),
            "score": float(h.score) if h.score is not None else None
        })

    return {**state, "setB_evidence": evidence}


# ============================
# NODE 5: Adjust Forecast + Confidence
# ============================
def node_adjust(state: Agent8State) -> Agent8State:
    base = state.get("base_risk") or 0.0
    ev = state.get("setB_evidence") or []

    if not ev:
        return {
            **state,
            "final_risk": base,
            "confidence": 0.55
        }

    sev_vals = [float(e.get("severity") or 0.0) for e in ev]
    high_rate = float(np.mean([1.0 if s >= 0.75 else 0.0 for s in sev_vals]))

    alpha = 0.25
    final = clamp(base + alpha * high_rate)

    conf = clamp(
        0.55 +
        0.25 * clamp(len(ev) / 30.0) +
        0.20 * clamp(np.mean(sev_vals))
    )

    return {**state, "final_risk": final, "confidence": conf}


# ============================
# NODE 6: Compose Output
# ============================
def node_compose(state: Agent8State) -> Agent8State:
    source_id = state.get("source_id")
    window_secs = state.get("window_secs") or parse_duration(DEFAULT_WINDOW)
    horizon_secs = state.get("horizon_secs") or parse_duration(DEFAULT_HORIZON)
    mode = state.get("mode") or DEFAULT_MODE

    base = state.get("base_risk") or 0.0
    final = state.get("final_risk") if state.get("final_risk") is not None else base
    conf = state.get("confidence") if state.get("confidence") is not None else 0.55

    out: Dict[str, Any] = {
        "source_id": source_id,
        "window": f"{int(window_secs/3600)}h",
        "horizon": f"{int(horizon_secs/3600)}h",
        "risk_forecast": {
            "risk_score": round(float(final), 4),
            "risk_level": risk_level(final),
            "confidence": round(float(conf), 4),
        },
        "meta": {
            "base_risk": round(float(base), 4),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    }

    if mode in ["risk+evidence", "risk+evidence+why"]:
        out["evidence_pack"] = (state.get("setB_evidence") or [])[:10]

    if mode == "risk+evidence+why":
        out["why"] = {
            "local_features": state.get("features") or {},
            "retrieval_evidence_count": len(state.get("setB_evidence") or []),
            "logic": [
                "Base risk computed from spike density, recency, volatility, trend, EW severity",
                "Retrieval adjustment uses severity distribution of similar historical episodes"
            ]
        }

    # action hooks
    if final >= 0.75:
        out["next_actions"] = [
            "Trigger SMTP/SMS alert",
            "Request citizen verification (photo/report)",
            "Recommend immediate field test + flush"
        ]
    elif final >= 0.45:
        out["next_actions"] = [
            "Monitor closely for next 6 hours",
            "Request citizen verification (photo/report)"
        ]
    else:
        out["next_actions"] = ["Continue monitoring (low risk)."]

    # cache
    ck = state.get("cache_key")
    if ck:
        cache_set(ck, out)

    return {**state, "output": out}


# ============================
# BUILD GRAPH
# ============================
def build_agent8():
    g = StateGraph(Agent8State)

    g.add_node("query", node_query_builder)
    g.add_node("cached", node_return_cached)

    g.add_node("fetch_setA", node_fetch_setA)
    g.add_node("base_risk", node_base_risk)

    g.add_node("fetch_setB", node_fetch_setB)
    g.add_node("adjust", node_adjust)

    g.add_node("compose", node_compose)

    g.set_entry_point("query")

    g.add_conditional_edges("query", router_cache, {
        "cached": "cached",
        "fetch_setA": "fetch_setA",
    })
    g.add_edge("cached", END)

    g.add_edge("fetch_setA", "base_risk")

    g.add_conditional_edges("base_risk", router_need_similar, {
        "fetch_setB": "fetch_setB",
        "compose": "compose",
    })

    g.add_edge("fetch_setB", "adjust")
    g.add_edge("adjust", "compose")
    g.add_edge("compose", END)

    return g.compile()


# ============================
# EXPOSE API
# ============================
def forecast(source_id: str, window: str = "24h", horizon: str = "6h", mode: str = "risk+evidence+why"):
    agent = build_agent8()
    init_state: Agent8State = {
        "request": {
            "source_id": source_id,
            "window": window,
            "horizon": horizon,
            "mode": mode,
        },
        "source_id": None,
        "window_secs": None,
        "horizon_secs": None,
        "mode": None,
        "cache_key": None,
        "cached_result": None,
        "setA_payloads": None,
        "last_event_vector": None,
        "base_risk": None,
        "features": None,
        "setB_evidence": None,
        "final_risk": None,
        "confidence": None,
        "output": None,
    }

    result = agent.invoke(init_state)
    return result["output"]


if __name__ == "__main__":
    print("\n[AGENT8] Running Forecast Test...\n")
    out = forecast("Well_Test_fairness", window="24h", horizon="6h", mode="risk+evidence+why")
    print(out)
