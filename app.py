"""
Flask Backend for Knapsack Neural Network Demo
Provides REST API for the knapsack solver with NN and optimal comparison.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import time
import json

from knapsack_nn import (
    KnapsackNet, 
    generate_knapsack_instance, 
    solve_knapsack_dp, 
    greedy_decode
)

app = Flask(__name__)
CORS(app)

# Global model instance
model = None
device = None

def load_model():
    """Load the trained neural network model."""
    global model, device
    
    MODEL_PATH = "knapsack_model_variable.pt"
    HIDDEN_DIM = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KnapsackNet(hidden_dim=HIDDEN_DIM).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"✓ Model loaded from {MODEL_PATH}")
        return True
    except FileNotFoundError:
        print(f"✗ Model file not found: {MODEL_PATH}")
        print("Run 'python knapsack_nn.py' first to train the model!")
        return False

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_problem():
    """
    Generate a random knapsack problem instance.
    
    Request JSON:
    {
        "n_items": 20,
        "capacity_ratio": 0.5,
        "seed": 42
    }
    
    Response JSON:
    {
        "items": [
            {"id": 0, "value": 45, "weight": 12},
            ...
        ],
        "capacity": 540,
        "n_items": 20
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
        
        n_items = data.get('n_items', 20)
        capacity_ratio = data.get('capacity_ratio', 0.5)
        seed = data.get('seed', None)
        
        # Convert to proper types
        n_items = int(n_items) if n_items else 20
        capacity_ratio = float(capacity_ratio) if capacity_ratio else 0.5
        if seed and seed != "":
            seed = int(seed)
        else:
            seed = None
        
        # Validate inputs
        if n_items < 5 or n_items > 20000:
            return jsonify({"error": "Number of items must be between 5 and 20000"}), 400
        
        if capacity_ratio < 0.1 or capacity_ratio > 0.9:
            return jsonify({"error": "Capacity ratio must be between 0.1 and 0.9"}), 400
    
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    
    # Generate problem
    values, weights, capacity = generate_knapsack_instance(n_items, seed, capacity_ratio)
    
    # Format items
    items = [
        {
            "id": i,
            "value": int(values[i]),
            "weight": int(weights[i]),
            "ratio": float(values[i] / weights[i])
        }
        for i in range(n_items)
    ]
    
    return jsonify({
        "items": items,
        "capacity": int(capacity),
        "n_items": n_items,
        "total_weight": int(np.sum(weights)),
        "total_value": int(np.sum(values))
    })

@app.route('/api/solve', methods=['POST'])
def solve_problem():
    """
    Solve a knapsack problem using both NN and optimal DP.
    
    Request JSON:
    {
        "items": [
            {"id": 0, "value": 45, "weight": 12},
            ...
        ],
        "capacity": 540,
        "use_local_search": true
    }
    
    Response JSON:
    {
        "nn_solution": {
            "selected_items": [0, 3, 5, ...],
            "total_value": 890,
            "total_weight": 535,
            "n_selected": 8,
            "time_ms": 12.5,
            "feasible": true
        },
        "optimal_solution": {
            "selected_items": [0, 3, 4, 5, ...],
            "total_value": 895,
            "total_weight": 538,
            "n_selected": 9,
            "time_ms": 45.2
        },
        "comparison": {
            "gap_percent": 0.56,
            "speedup": 3.6,
            "value_difference": 5
        }
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json()
    items = data.get('items', [])
    capacity = data.get('capacity')
    use_local_search = data.get('use_local_search', True)
    
    if not items or capacity is None:
        return jsonify({"error": "Missing items or capacity"}), 400
    
    # Extract values and weights
    n_items = len(items)
    values = np.array([item['value'] for item in items])
    weights = np.array([item['weight'] for item in items])
    
    # Solve with Neural Network
    start_time = time.time()
    
    with torch.no_grad():
        values_tensor = torch.FloatTensor(values).unsqueeze(0).to(device)
        weights_tensor = torch.FloatTensor(weights).unsqueeze(0).to(device)
        capacity_tensor = torch.FloatTensor([capacity]).unsqueeze(0).to(device)
        
        # Predict
        logits = model(values_tensor, weights_tensor, capacity_tensor)
        probs = torch.sigmoid(logits)
        
        # Decode to solution
        nn_solution = greedy_decode(
            probs, weights_tensor, capacity_tensor,
            use_local_search=use_local_search,
            values_for_local_search=values_tensor
        )
        
        # Extract solution
        nn_selected = nn_solution.squeeze().cpu().numpy()
        nn_selected_items = [i for i in range(n_items) if nn_selected[i] > 0.5]
        nn_value = int(np.sum(values[nn_selected > 0.5]))
        nn_weight = int(np.sum(weights[nn_selected > 0.5]))
        nn_probabilities = probs.squeeze().cpu().numpy().tolist()
    
    nn_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Solve with Optimal DP
    start_time = time.time()
    opt_value, opt_solution = solve_knapsack_dp(values, weights, capacity)
    opt_time = (time.time() - start_time) * 1000  # Convert to ms
    
    opt_selected_items = [i for i in range(n_items) if opt_solution[i] == 1]
    opt_weight = int(np.sum(weights[opt_solution == 1]))
    
    # Compute comparison metrics
    gap_percent = ((opt_value - nn_value) / opt_value * 100) if opt_value > 0 else 0
    speedup = opt_time / nn_time if nn_time > 0 else 1
    
    return jsonify({
        "nn_solution": {
            "selected_items": nn_selected_items,
            "total_value": int(nn_value),
            "total_weight": int(nn_weight),
            "n_selected": int(len(nn_selected_items)),
            "time_ms": float(round(nn_time, 2)),
            "feasible": bool(nn_weight <= capacity),
            "probabilities": [float(p) for p in nn_probabilities]
        },
        "optimal_solution": {
            "selected_items": opt_selected_items,
            "total_value": int(opt_value),
            "total_weight": int(opt_weight),
            "n_selected": int(len(opt_selected_items)),
            "time_ms": float(round(opt_time, 2))
        },
        "comparison": {
            "gap_percent": float(round(gap_percent, 2)),
            "speedup": float(round(speedup, 2)),
            "value_difference": int(opt_value - nn_value),
            "nn_better_or_equal": bool(nn_value >= opt_value)
        }
    })

@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    """
    Run a benchmark across multiple problem sizes.
    
    Request JSON:
    {
        "sizes": [10, 20, 30, 40, 50],
        "capacity_ratio": 0.5,
        "n_tests": 10
    }
    
    Response JSON:
    {
        "results": [
            {
                "n_items": 10,
                "avg_gap": 0.45,
                "avg_nn_time": 3.2,
                "avg_dp_time": 5.1,
                "speedup": 1.6
            },
            ...
        ]
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json()
    sizes = data.get('sizes', [10, 20, 30, 40, 50])
    capacity_ratio = data.get('capacity_ratio', 0.5)
    n_tests = data.get('n_tests', 10)
    
    results = []
    
    for n_items in sizes:
        gaps = []
        nn_times = []
        dp_times = []
        
        for i in range(n_tests):
            # Generate instance
            values, weights, capacity = generate_knapsack_instance(
                n_items, seed=60000 + i, capacity_ratio=capacity_ratio
            )
            
            # Solve with NN
            start_time = time.time()
            with torch.no_grad():
                values_tensor = torch.FloatTensor(values).unsqueeze(0).to(device)
                weights_tensor = torch.FloatTensor(weights).unsqueeze(0).to(device)
                capacity_tensor = torch.FloatTensor([capacity]).unsqueeze(0).to(device)
                
                logits = model(values_tensor, weights_tensor, capacity_tensor)
                probs = torch.sigmoid(logits)
                nn_solution = greedy_decode(
                    probs, weights_tensor, capacity_tensor,
                    use_local_search=True,
                    values_for_local_search=values_tensor
                )
                nn_value = torch.sum(nn_solution * values_tensor).item()
            nn_times.append((time.time() - start_time) * 1000)
            
            # Solve with DP
            start_time = time.time()
            opt_value, _ = solve_knapsack_dp(values, weights, capacity)
            dp_times.append((time.time() - start_time) * 1000)
            
            # Compute gap
            gap = ((opt_value - nn_value) / opt_value * 100) if opt_value > 0 else 0
            gaps.append(gap)
        
        results.append({
            "n_items": n_items,
            "avg_gap": round(np.mean(gaps), 2),
            "median_gap": round(np.median(gaps), 2),
            "max_gap": round(np.max(gaps), 2),
            "avg_nn_time": round(np.mean(nn_times), 2),
            "avg_dp_time": round(np.mean(dp_times), 2),
            "speedup": round(np.mean(dp_times) / np.mean(nn_times), 2)
        })
    
    return jsonify({"results": results})

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        with open("knapsack_config_variable.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {"error": "Config file not found"}
    
    return jsonify({
        "model_loaded": model is not None,
        "device": str(device),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "config": config
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Knapsack Neural Network Demo Server")
    print("=" * 60)
    
    # Load model on startup
    if load_model():
        print("\nStarting Flask server...")
        print("Open your browser and go to: http://localhost:5000")
        print("=" * 60)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\nERROR: Could not start server - model not loaded")
        print("Please run 'python knapsack_nn.py' first to train the model")
