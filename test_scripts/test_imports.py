"""
Minimal startup test - verifies backend can start without errors
"""
import sys
import os

# Set up path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

print("=" * 60)
print("Water Watch - Startup Verification")
print("=" * 60)

# Test 1: Config loads
print("\n[1/4] Testing config...")
try:
    from config import settings
    print(f"✅ Config loaded")
    print(f"    - Gemini API Key: {'*' * 20}{settings.GEMINI_API_KEY[-10:]}")
    print(f"    - Kafka: {settings.KAFKA_BOOTSTRAP_SERVERS}")
    print(f"    - Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
except Exception as e:
    print(f"❌ Config failed: {e}")
    sys.exit(1)

# Test 2: Schemas load
print("\n[2/4] Testing schemas...")
try:
    from schemas import SensorReading, HealthCheck, ChatQuery
    print("✅ Schemas loaded")
except Exception as e:
    print(f"❌ Schemas failed: {e}")
    sys.exit(1)

# Test 3: Services package loads
print("\n[3/4] Testing services package...")
try:
    import services
    print("✅ Services package loaded")
except Exception as e:
    print(f"❌ Services failed: {e}")
    sys.exit(1)

# Test 4: Main app can be imported
print("\n[4/4] Testing main app import...")
try:
    # Don't actually run it, just import
    import main
    print("✅ Main app imported successfully")
except Exception as e:
    print(f"❌ Main app failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - Backend is ready!")
print("=" * 60)
print("\nYou can now start the server:")
print("  cd backend")
print("  uvicorn main:app --reload")
print("\nOr use Docker:")
print("  docker-compose up -d")
print("=" * 60)
