"""
Quick test script for the variable-N trained knapsack model.
Run this to test the model on custom problem sizes and capacities.
"""

import torch
import numpy as np
from knapsack_nn import KnapsackNet, generate_knapsack_instance, solve_knapsack_dp, greedy_decode


def test_model_on_size(model, n_items, capacity_ratio=0.5, n_tests=10, use_local_search=True):
    """
    Test model on a specific problem size.
    
    Args:
        model: Trained model
        n_items: Number of items to test
        capacity_ratio: Capacity tightness (0.3=tight, 0.5=balanced, 0.7=loose)
        n_tests: Number of test instances
        use_local_search: Whether to use 2-swap improvement
    """
    model.eval()
    device = next(model.parameters()).device
    
    gaps = []
    feasible_count = 0
    
    print(f"\n{'='*60}")
    print(f"Testing N={n_items}, Capacity={capacity_ratio:.1%}")
    print(f"{'='*60}")
    
    with torch.no_grad():
        for i in range(n_tests):
            # Generate test instance
            values, weights, capacity = generate_knapsack_instance(
                n_items, seed=50000 + i, capacity_ratio=capacity_ratio
            )
            opt_val, opt_sol = solve_knapsack_dp(values, weights, capacity)
            
            # Convert to tensors
            values_tensor = torch.FloatTensor(values).unsqueeze(0).to(device)
            weights_tensor = torch.FloatTensor(weights).unsqueeze(0).to(device)
            capacity_tensor = torch.FloatTensor([capacity]).unsqueeze(0).to(device)
            
            # Predict
            logits = model(values_tensor, weights_tensor, capacity_tensor)
            probs = torch.sigmoid(logits)
            
            # Decode with optional local search
            predicted_solution = greedy_decode(
                probs, weights_tensor, capacity_tensor,
                use_local_search=use_local_search,
                values_for_local_search=values_tensor
            )
            
            # Compute metrics
            predicted_val = torch.sum(predicted_solution * values_tensor, dim=1).item()
            predicted_weight = torch.sum(predicted_solution * weights_tensor, dim=1).item()
            is_feasible = predicted_weight <= capacity
            
            if is_feasible:
                feasible_count += 1
            
            gap = (opt_val - predicted_val) / opt_val * 100 if opt_val > 0 else 0
            gaps.append(gap)
            
            # Print first 3 instances
            if i < 3:
                print(f"\nInstance {i+1}:")
                print(f"  Items selected: {predicted_solution.sum().item():.0f}/{n_items}")
                print(f"  Weight used: {predicted_weight:.0f}/{capacity}")
                print(f"  Value: {predicted_val:.0f} vs Optimal: {opt_val}")
                print(f"  Gap: {gap:.2f}%")
                print(f"  Feasible: {'✓' if is_feasible else '✗'}")
    
    # Summary
    print(f"\n{'-'*60}")
    print(f"Summary for N={n_items}, Cap={capacity_ratio:.1%}:")
    print(f"  Tests: {n_tests}")
    print(f"  Feasibility: {feasible_count}/{n_tests} ({feasible_count/n_tests*100:.1f}%)")
    print(f"  Average Gap: {np.mean(gaps):.2f}%")
    print(f"  Median Gap: {np.median(gaps):.2f}%")
    print(f"  Max Gap: {np.max(gaps):.2f}%")
    print(f"  Std Dev: {np.std(gaps):.2f}%")
    print(f"{'-'*60}")
    
    return {
        'n_items': n_items,
        'capacity_ratio': capacity_ratio,
        'avg_gap': np.mean(gaps),
        'feasibility': feasible_count / n_tests
    }


def main():
    """Test the variable-N model on various configurations."""
    
    # Load the variable-N model
    MODEL_PATH = "knapsack_model_variable.pt"
    HIDDEN_DIM = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = KnapsackNet(hidden_dim=HIDDEN_DIM).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"✓ Loaded model from: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"✗ Model file not found: {MODEL_PATH}")
        print("Run 'python knapsack_nn.py' first to train the model!")
        return
    
    model.eval()
    
    print("\n" + "="*60)
    print("TESTING VARIABLE-N TRAINED MODEL")
    print("="*60)
    
    # Test 1: Different problem sizes with balanced capacity
    print("\n### TEST 1: Generalization across problem sizes (balanced capacity)")
    for n in [15, 25, 35, 45, 55]:
        test_model_on_size(model, n, capacity_ratio=0.5, n_tests=20)
    
    # Test 2: Different capacity tightness on medium problems
    print("\n\n### TEST 2: Robustness to capacity variation (N=40)")
    for cap in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
        test_model_on_size(model, n_items=40, capacity_ratio=cap, n_tests=20)
    
    # Test 3: Extrapolation to larger sizes (beyond training range)
    print("\n\n### TEST 3: Extrapolation beyond training range")
    for n in [70, 80, 100, 150]:
        test_model_on_size(model, n, capacity_ratio=0.5, n_tests=10)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE!")
    print("="*60)
    print("\nKey Findings:")
    print("1. Check if gaps remain low across all N ∈ [10, 60]")
    print("2. Check if model handles varied capacities (0.3-0.7)")
    print("3. Check extrapolation to N > 60 (not in training range)")
    print("="*60)


if __name__ == "__main__":
    main()
