"""
Evaluation Script for Trained Knapsack Neural Network

This script loads a trained model and benchmarks it against:
1. Small/Medium problems (N=10-50): vs Optimal DP
2. Large problems (N=100-500): vs Time-limited MILP solver

Usage:
    python evaluate_knapsack.py
    python evaluate_knapsack.py --model knapsack_model.pt --n-small 30 --n-large 200
"""

import torch
import numpy as np
import json
import time
import argparse
from typing import Tuple, Dict, List
from knapsack_nn import (
    KnapsackNet, 
    generate_knapsack_instance, 
    solve_knapsack_dp,
    greedy_decode,
    compute_value
)


# ==============================================================================
# MILP SOLVER (for large instances)
# ==============================================================================

def solve_knapsack_milp(values: np.ndarray, weights: np.ndarray, capacity: int, 
                        time_limit: float = 5.0) -> Tuple[int, np.ndarray, float]:
    """
    Solve knapsack using MILP solver (requires pulp or similar).
    Falls back to greedy heuristic if solver not available.
    
    Args:
        values: Item values
        weights: Item weights
        capacity: Knapsack capacity
        time_limit: Time limit in seconds
    
    Returns:
        best_value: Best value found
        solution: Best solution found
        solve_time: Time taken
    """
    try:
        import pulp
        from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
        
        n = len(values)
        start_time = time.time()
        
        # Create problem
        prob = LpProblem("Knapsack", LpMaximize)
        
        # Binary variables
        x = [LpVariable(f"x{i}", cat='Binary') for i in range(n)]
        
        # Objective: maximize value
        prob += lpSum([values[i] * x[i] for i in range(n)])
        
        # Constraint: weight limit
        prob += lpSum([weights[i] * x[i] for i in range(n)]) <= capacity
        
        # Solve with time limit
        prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0))
        
        solve_time = time.time() - start_time
        
        # Extract solution
        solution = np.array([x[i].varValue for i in range(n)], dtype=np.int32)
        best_value = int(sum(values[i] * solution[i] for i in range(n)))
        
        return best_value, solution, solve_time
        
    except ImportError:
        print("Warning: PuLP not installed. Using greedy heuristic for large instances.")
        print("Install with: pip install pulp")
        return solve_knapsack_greedy(values, weights, capacity)


def solve_knapsack_greedy(values: np.ndarray, weights: np.ndarray, capacity: int) -> Tuple[int, np.ndarray, float]:
    """
    Greedy heuristic: sort by value/weight ratio and add greedily.
    
    Returns:
        value: Total value
        solution: Binary solution
        time: Time taken
    """
    start_time = time.time()
    
    n = len(values)
    # Compute value/weight ratios
    ratios = values / (weights + 1e-8)
    sorted_indices = np.argsort(ratios)[::-1]
    
    solution = np.zeros(n, dtype=np.int32)
    current_weight = 0
    
    for idx in sorted_indices:
        if current_weight + weights[idx] <= capacity:
            solution[idx] = 1
            current_weight += weights[idx]
    
    value = int(np.sum(values * solution))
    solve_time = time.time() - start_time
    
    return value, solution, solve_time


# ==============================================================================
# BENCHMARKING FUNCTIONS
# ==============================================================================

def benchmark_small_medium(model: KnapsackNet, device: torch.device, 
                          test_sizes: List[int] = [10, 20, 30, 50],
                          n_instances: int = 50) -> Dict:
    """
    Benchmark on small/medium instances against optimal DP.
    
    Args:
        model: Trained neural network
        device: Device to run on
        test_sizes: List of problem sizes to test
        n_instances: Number of test instances per size
    
    Returns:
        results: Dictionary with benchmark results
    """
    print("=" * 70)
    print("BENCHMARK: SMALL/MEDIUM PROBLEMS (vs Optimal DP)")
    print("=" * 70)
    
    results = {}
    
    for n_items in test_sizes:
        print(f"\n--- Testing N={n_items} ---")
        
        gaps = []
        nn_times = []
        dp_times = []
        feasibility_count = 0
        
        for i in range(n_instances):
            # Generate instance
            values, weights, capacity = generate_knapsack_instance(n_items, seed=20000 + i)
            
            # Solve with DP (optimal)
            dp_start = time.time()
            opt_val, opt_sol = solve_knapsack_dp(values, weights, capacity)
            dp_time = time.time() - dp_start
            dp_times.append(dp_time * 1000)  # Convert to ms
            
            # Solve with NN
            nn_start = time.time()
            with torch.no_grad():
                values_tensor = torch.FloatTensor(values).unsqueeze(0).to(device)
                weights_tensor = torch.FloatTensor(weights).unsqueeze(0).to(device)
                capacity_tensor = torch.FloatTensor([capacity]).unsqueeze(0).to(device)
                
                logits = model(values_tensor, weights_tensor, capacity_tensor)
                probs = torch.sigmoid(logits)
                
                predicted_solution = greedy_decode(
                    probs, weights_tensor, capacity_tensor,
                    use_local_search=True,
                    values_for_local_search=values_tensor
                )
                predicted_val = compute_value(predicted_solution, values_tensor).item()
            
            nn_time = time.time() - nn_start
            nn_times.append(nn_time * 1000)  # Convert to ms
            
            # Check feasibility
            predicted_weight = torch.sum(predicted_solution * weights_tensor, dim=1).item()
            is_feasible = predicted_weight <= capacity
            if is_feasible:
                feasibility_count += 1
            
            # Compute gap
            gap = (opt_val - predicted_val) / opt_val * 100 if opt_val > 0 else 0
            gaps.append(gap)
        
        # Summary
        avg_gap = np.mean(gaps)
        median_gap = np.median(gaps)
        avg_nn_time = np.mean(nn_times)
        avg_dp_time = np.mean(dp_times)
        speedup = avg_dp_time / avg_nn_time if avg_nn_time > 0 else 0
        
        print(f"  Instances:       {n_instances}")
        print(f"  Feasibility:     {feasibility_count}/{n_instances} ({feasibility_count/n_instances*100:.1f}%)")
        print(f"  Average Gap:     {avg_gap:.2f}%")
        print(f"  Median Gap:      {median_gap:.2f}%")
        print(f"  NN Time:         {avg_nn_time:.2f}ms")
        print(f"  DP Time:         {avg_dp_time:.2f}ms")
        print(f"  Speedup:         {speedup:.2f}x")
        
        results[f"n{n_items}"] = {
            "n_items": n_items,
            "n_instances": n_instances,
            "feasibility_rate": feasibility_count / n_instances,
            "avg_gap": float(avg_gap),
            "median_gap": float(median_gap),
            "avg_nn_time_ms": float(avg_nn_time),
            "avg_dp_time_ms": float(avg_dp_time),
            "speedup": float(speedup)
        }
    
    return results


def benchmark_large(model: KnapsackNet, device: torch.device,
                   test_sizes: List[int] = [100, 200, 500],
                   n_instances: int = 20,
                   milp_time_limit: float = 5.0) -> Dict:
    """
    Benchmark on large instances against time-limited MILP.
    
    Args:
        model: Trained neural network
        device: Device to run on
        test_sizes: List of problem sizes to test
        n_instances: Number of test instances per size
        milp_time_limit: Time limit for MILP solver in seconds
    
    Returns:
        results: Dictionary with benchmark results
    """
    print("\n" + "=" * 70)
    print(f"BENCHMARK: LARGE PROBLEMS (vs {milp_time_limit}s MILP)")
    print("=" * 70)
    
    results = {}
    
    for n_items in test_sizes:
        print(f"\n--- Testing N={n_items} ---")
        
        nn_values = []
        milp_values = []
        nn_times = []
        milp_times = []
        feasibility_count = 0
        
        for i in range(n_instances):
            # Generate instance
            values, weights, capacity = generate_knapsack_instance(n_items, seed=30000 + i)
            
            # Solve with MILP (time-limited)
            milp_val, milp_sol, milp_time = solve_knapsack_milp(
                values, weights, capacity, time_limit=milp_time_limit
            )
            milp_values.append(milp_val)
            milp_times.append(milp_time * 1000)  # Convert to ms
            
            # Solve with NN
            nn_start = time.time()
            with torch.no_grad():
                values_tensor = torch.FloatTensor(values).unsqueeze(0).to(device)
                weights_tensor = torch.FloatTensor(weights).unsqueeze(0).to(device)
                capacity_tensor = torch.FloatTensor([capacity]).unsqueeze(0).to(device)
                
                logits = model(values_tensor, weights_tensor, capacity_tensor)
                probs = torch.sigmoid(logits)
                
                predicted_solution = greedy_decode(
                    probs, weights_tensor, capacity_tensor,
                    use_local_search=True,
                    values_for_local_search=values_tensor
                )
                predicted_val = compute_value(predicted_solution, values_tensor).item()
            
            nn_time = time.time() - nn_start
            nn_values.append(predicted_val)
            nn_times.append(nn_time * 1000)  # Convert to ms
            
            # Check feasibility
            predicted_weight = torch.sum(predicted_solution * weights_tensor, dim=1).item()
            is_feasible = predicted_weight <= capacity
            if is_feasible:
                feasibility_count += 1
        
        # Compute relative performance
        gaps = [(milp_val - nn_val) / milp_val * 100 if milp_val > 0 else 0 
                for milp_val, nn_val in zip(milp_values, nn_values)]
        
        avg_gap = np.mean(gaps)
        median_gap = np.median(gaps)
        avg_nn_time = np.mean(nn_times)
        avg_milp_time = np.mean(milp_times)
        speedup = avg_milp_time / avg_nn_time if avg_nn_time > 0 else 0
        
        print(f"  Instances:           {n_instances}")
        print(f"  Feasibility:         {feasibility_count}/{n_instances} ({feasibility_count/n_instances*100:.1f}%)")
        print(f"  Avg Gap (vs MILP):   {avg_gap:.2f}%")
        print(f"  Median Gap:          {median_gap:.2f}%")
        print(f"  NN Time:             {avg_nn_time:.2f}ms")
        print(f"  MILP Time:           {avg_milp_time:.2f}ms (limit: {milp_time_limit*1000:.0f}ms)")
        print(f"  Speedup:             {speedup:.2f}x")
        
        results[f"n{n_items}"] = {
            "n_items": n_items,
            "n_instances": n_instances,
            "feasibility_rate": feasibility_count / n_instances,
            "avg_gap_vs_milp": float(avg_gap),
            "median_gap": float(median_gap),
            "avg_nn_time_ms": float(avg_nn_time),
            "avg_milp_time_ms": float(avg_milp_time),
            "speedup": float(speedup)
        }
    
    return results


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained knapsack model')
    parser.add_argument('--model', type=str, default='knapsack_model.pt',
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='knapsack_config.json',
                       help='Path to config file')
    parser.add_argument('--n-small', type=int, nargs='+', default=[10, 20, 30, 50],
                       help='Problem sizes for small benchmark')
    parser.add_argument('--n-large', type=int, nargs='+', default=[100, 200],
                       help='Problem sizes for large benchmark (set to 500 if you have time)')
    parser.add_argument('--instances-small', type=int, default=50,
                       help='Number of instances per size (small)')
    parser.add_argument('--instances-large', type=int, default=20,
                       help='Number of instances per size (large)')
    parser.add_argument('--milp-timeout', type=float, default=5.0,
                       help='MILP time limit in seconds')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        hidden_dim = config['hidden_dim']
        print(f"Loaded config from {args.config}")
    except FileNotFoundError:
        print(f"Warning: Config file {args.config} not found. Using default hidden_dim=128")
        hidden_dim = 128
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = KnapsackNet(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Model loaded successfully!\n")
    
    # Benchmark small/medium problems
    small_results = benchmark_small_medium(
        model=model,
        device=device,
        test_sizes=args.n_small,
        n_instances=args.instances_small
    )
    
    # Benchmark large problems
    large_results = benchmark_large(
        model=model,
        device=device,
        test_sizes=args.n_large,
        n_instances=args.instances_large,
        milp_time_limit=args.milp_timeout
    )
    
    # Combine and save results
    all_results = {
        "small_medium_problems": small_results,
        "large_problems": large_results,
        "config": {
            "model_path": args.model,
            "hidden_dim": hidden_dim,
            "device": str(device)
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {args.output}")
    print("\nQuick Summary:")
    print(f"  Small problems (N={args.n_small[0]}-{args.n_small[-1]}): avg {np.mean([r['avg_gap'] for r in small_results.values()]):.2f}% gap")
    print(f"  Large problems (N={args.n_large[0]}-{args.n_large[-1]}): avg {np.mean([r['avg_gap_vs_milp'] for r in large_results.values()]):.2f}% gap vs MILP")
    print("=" * 70)


if __name__ == "__main__":
    main()
