"""
Quick test script to verify the Flask API is working correctly.
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_generate():
    """Test the generate endpoint."""
    print("Testing /api/generate...")
    
    payload = {
        "n_items": 30,
        "capacity_ratio": 0.5,
        "seed": 42
    }
    
    response = requests.post(f"{BASE_URL}/api/generate", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Generated problem: {data['n_items']} items, capacity={data['capacity']}")
        return data
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return None

def test_solve(problem):
    """Test the solve endpoint."""
    print("\nTesting /api/solve...")
    
    if not problem:
        print("✗ No problem to solve")
        return
    
    payload = {
        "items": problem["items"],
        "capacity": problem["capacity"],
        "use_local_search": True
    }
    
    response = requests.post(f"{BASE_URL}/api/solve", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Solved successfully!")
        print(f"  NN Value: {data['nn_solution']['total_value']}")
        print(f"  Optimal Value: {data['optimal_solution']['total_value']}")
        print(f"  Gap: {data['comparison']['gap_percent']}%")
        print(f"  Speedup: {data['comparison']['speedup']}x")
        return data
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return None

def test_model_info():
    """Test the model info endpoint."""
    print("\nTesting /api/model-info...")
    
    response = requests.get(f"{BASE_URL}/api/model-info")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Model info retrieved")
        print(f"  Model loaded: {data['model_loaded']}")
        print(f"  Device: {data['device']}")
        print(f"  Parameters: {data['n_parameters']:,}")
        return data
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("Flask API Test Suite")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print("Make sure the Flask server is running!")
    print("=" * 60)
    
    try:
        # Test 1: Model Info
        test_model_info()
        
        # Test 2: Generate Problem
        problem = test_generate()
        
        # Test 3: Solve Problem
        if problem:
            test_solve(problem)
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Cannot connect to Flask server")
        print("Make sure the server is running: python app.py")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
