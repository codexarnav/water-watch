"""
Test script for the Water-Watch Orchestration Agent

This script tests the orchestrator with sample data:
1. Signal data (sensor readings)
2. Citizen data (text, image reports)
"""

import json
import time
from datetime import datetime, timezone

# Import orchestrator components
from input_agent import (
    build_orchestration_graph,
    create_initial_state,
    add_citizen_input,
    publish_to_kafka,
    config
)

# =========================================================
# TEST DATA
# =========================================================

def create_test_sensor_message(well_id="Well_Test_001", ph=7.5):
    """Create a test sensor message."""
    ts = datetime.now(timezone.utc).isoformat()
    
    return {
        "source_id": well_id,
        "timestamp": ts,
        "readings": {
            "salinity_ppt": 35.2,
            "dissolved_oxygen_mgL": 6.8,
            "ph": ph,  # Can be varied to trigger spikes
            "secchi_depth_m": 2.3,
            "water_depth_m": 15.7,
            "water_temp_c": 22.4
        }
    }

def create_test_citizen_input(with_image=False):
    """Create a test citizen input."""
    citizen_input ={
        "source": "whatsapp",
        "timestamp": 1712345678,
        "sender_id": "9198xxxxxx",
        "location": {
            "lat": 28.6139,
            "lon": 77.2090
        },
        "content": {
            "text": "drain opened to freshwater lake",
            "video_uri": "9736660-hd_1920_1080_25fps.mp4"
        }
    }
    
    if with_image:
        # Add image if you have one in images/ folder
        citizen_input["content"]["image_uri"] = "test_image.jpg"
    
    return citizen_input

# =========================================================
# TEST SCENARIOS
# =========================================================

def test_signal_stream():
    """
    Test 1: Signal Stream Processing
    - Publishes normal sensor reading to Kafka
    - Should process without spike
    """
    print("\n" + "="*60)
    print("TEST 1: Signal Stream (Normal Reading)")
    print("="*60)
    
    # Create normal sensor reading
    msg = create_test_sensor_message(ph=7.2)
    
    # Publish to Kafka
    print(f"📤 Publishing sensor data: well={msg['source_id']}, ph={msg['readings']['ph']}")
    publish_to_kafka(config.KAFKA_RAW_TOPIC, msg["source_id"], msg)
    
    # Run orchestrator with limited recursion for testing
    orchestrator = build_orchestration_graph()
    state = create_initial_state()
    
    for i in range(5):
        state = orchestrator.invoke(state, {"recursion_limit": 15})
        if state["signal_status"] == "done":
            print(f"✅ Signal processed successfully")
            print(f"   Processed count: {state['processed_count']['signal']}")
            break
        time.sleep(0.5)
    else:
        print(f"⚠️ Signal processing incomplete: status={state['signal_status']}")

def test_signal_stream_with_spike():
    """
    Test 2: Signal Stream with Spike Detection
    - Publishes abnormal sensor reading
    - Should trigger spike detection and embedding
    """
    print("\n" + "="*60)
    print("TEST 2: Signal Stream (Spike Detection)")
    print("="*60)
    
    # First send normal readings to build baseline
    print("📊 Building baseline with normal readings...")
    for i in range(12):
        msg = create_test_sensor_message(ph=7.0 + (i % 3) * 0.1)
        publish_to_kafka(config.KAFKA_RAW_TOPIC, msg["source_id"], msg)
        time.sleep(0.1)
    
    # Now send spike
    spike_msg = create_test_sensor_message(ph=12.5)  # Extreme pH
    print(f"📤 Publishing spike: well={spike_msg['source_id']}, ph={spike_msg['readings']['ph']}")
    publish_to_kafka(config.KAFKA_RAW_TOPIC, spike_msg["source_id"], spike_msg)
    
    # Run orchestrator with limited recursion for testing
    orchestrator = build_orchestration_graph()
    state = create_initial_state()
    
    # Process baseline messages
    for i in range(15):
        state = orchestrator.invoke(state, {"recursion_limit": 15})
        time.sleep(0.2)
    
    # Check if spike was detected and embedded
    if state["processed_count"]["signal"] > 0:
        print(f"✅ Processed {state['processed_count']['signal']} signal messages")
        if state.get("signal_state") and state["signal_state"].get("event"):
            print(f"🔔 Spike detected!")
            print(f"   Event: {state['signal_state']['event']['derived']['semantic_text']}")
    else:
        print(f"⚠️ No signal messages processed")

def test_citizen_stream():
    """
    Test 3: Citizen Stream Processing
    - Adds citizen text report
    - Should route, embed, create voxel, and store
    """
    print("\n" + "="*60)
    print("TEST 3: Citizen Stream (Text Report)")
    print("="*60)
    
    # Create citizen input
    citizen_data = create_test_citizen_input()
    print(f"📤 Adding citizen report:")
    print(f"   Source: {citizen_data['source']}")
    print(f"   Text: {citizen_data['content']['text']}")
    print(f"   Location: {citizen_data['location']}")
    
    # Add to queue
    add_citizen_input(citizen_data)
    
    # Run orchestrator with limited recursion for testing
    orchestrator = build_orchestration_graph()
    state = create_initial_state()
    
    for i in range(10):
        state = orchestrator.invoke(state, {"recursion_limit": 15})
        
        if state["citizen_status"] == "done":
            print(f"✅ Citizen report processed successfully")
            print(f"   Processed count: {state['processed_count']['citizen']}")
            if state.get("citizen_state"):
                cs = state["citizen_state"]
                if cs.get("routed_signals"):
                    print(f"   Routed signals: {len(cs['routed_signals'])}")
                if cs.get("percepts"):
                    print(f"   Percepts created: {len(cs['percepts'])}")
                if cs.get("voxels"):
                    print(f"   Voxels created: {len(cs['voxels'])}")
            break
        
        time.sleep(0.5)
    else:
        print(f"⚠️ Citizen processing incomplete: status={state['citizen_status']}")
        if state.get("citizen_error"):
            print(f"   Error: {state['citizen_error']}")

def test_mixed_streams():
    """
    Test 4: Mixed Stream Processing
    - Both signal and citizen data
    - Tests stream prioritization and switching
    """
    print("\n" + "="*60)
    print("TEST 4: Mixed Streams (Signal + Citizen)")
    print("="*60)
    
    # Add signal data
    for i in range(3):
        msg = create_test_sensor_message(f"Well_Mix_{i}", ph=7.0 + i*0.2)
        publish_to_kafka(config.KAFKA_RAW_TOPIC, msg["source_id"], msg)
    print(f"📤 Published 3 signal messages")
    
    # Add citizen data
    for i in range(2):
        citizen_data = create_test_citizen_input()
        citizen_data["sender_id"] = f"91981234567{i}"
        add_citizen_input(citizen_data)
    print(f"📤 Added 2 citizen reports")
    
    # Run orchestrator with limited recursion for testing
    orchestrator = build_orchestration_graph()
    state = create_initial_state()
    
    print("\n🔄 Processing mixed streams...")
    for i in range(20):
        state = orchestrator.invoke(state, {"recursion_limit": 15})
        time.sleep(0.3)
    
    print(f"\n✅ Processing complete:")
    print(f"   Signal processed: {state['processed_count']['signal']}")
    print(f"   Citizen processed: {state['processed_count']['citizen']}")
    print(f"   Signal errors: {state['error_count'].get('signal', 0)}")
    print(f"   Citizen errors: {state['error_count'].get('citizen', 0)}")

def test_error_handling():
    """
    Test 5: Error Handling
    - Sends invalid citizen data
    - Should trigger error handler and retry logic
    """
    print("\n" + "="*60)
    print("TEST 5: Error Handling & Retries")
    print("="*60)
    
    # Create invalid citizen input (missing required fields)
    invalid_data = {
        "source": "whatsapp",
        # Missing timestamp, location, content
    }
    
    print(f"📤 Adding invalid citizen report (should fail)")
    add_citizen_input(invalid_data)
    
    # Run orchestrator with limited recursion for testing
    orchestrator = build_orchestration_graph()
    state = create_initial_state()
    
    for i in range(15):
        state = orchestrator.invoke(state, {"recursion_limit": 15})
        
        if state["citizen_status"] == "error":
            print(f"⚠️ Error detected:")
            print(f"   Error: {state['citizen_error']}")
            print(f"   Retry count: {state['citizen_retry_count']}")
        
        if state["citizen_status"] == "done" and state["citizen_retry_count"] > 0:
            print(f"✅ Error handled and sent to DLQ")
            print(f"   Check dead_letter_queue.jsonl for details")
            break
        
        time.sleep(0.5)

# =========================================================
# MAIN
# =========================================================

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Water-Watch Orchestration Agent - Test Suite")
    print("="*60)
    
    tests = [
        ("Signal Stream (Normal)", test_signal_stream),
        ("Signal Stream (Spike)", test_signal_stream_with_spike),
        ("Citizen Stream", test_citizen_stream),
        ("Mixed Streams", test_mixed_streams),
        ("Error Handling", test_error_handling),
    ]
    
    print("\nAvailable tests:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"  {i}. {name}")
    print(f"  0. Run all tests")
    
    choice = input("\nSelect test to run (0-5): ").strip()
    
    if choice == "0":
        # Run all tests
        for name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"❌ Test '{name}' failed: {e}")
                import traceback
                traceback.print_exc()
    elif choice.isdigit() and 1 <= int(choice) <= len(tests):
        # Run selected test
        name, test_func = tests[int(choice) - 1]
        try:
            test_func()
        except Exception as e:
            print(f"❌ Test '{name}' failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice")
        return
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)
    
    print("\n📊 Next steps:")
    print("  1. Check logs above for test results")
    print("  2. Query Qdrant to verify data storage:")
    print("     curl http://localhost:6333/collections/water_memory")
    print("  3. Check dead_letter_queue.jsonl for failed messages")

if __name__ == "__main__":
    main()
