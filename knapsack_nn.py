"""
Neural Network for 0/1 Knapsack Problem Approximation

This script trains a neural network to predict solutions for the 0/1 Knapsack problem.
It uses a DeepSets-style architecture for permutation invariance and combines binary
cross-entropy loss with a regret-based loss to improve solution quality.

Key Features:
- Uses logits + BCEWithLogitsLoss for numerical stability
- Engineered features: v/w ratio, w/C ratio, value/weight proportions
- Greedy feasibility decoding with clamping to avoid NaN/inf
- Reports feasibility (should be 100%) and evaluation speed
- Regret loss encourages better solution quality
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List
import random


# ==============================================================================
# DATA GENERATION AND OPTIMAL SOLVER
# ==============================================================================

def generate_knapsack_instance(n_items: int, seed: int = None, capacity_ratio: float = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate a random knapsack instance.
    
    Args:
        n_items: Number of items
        seed: Random seed for reproducibility
        capacity_ratio: Ratio of capacity to total weight (if None, defaults to 0.5)
                       Can vary (e.g., 0.3 = tight, 0.5 = balanced, 0.7 = loose)
    
    Returns:
        values: Array of item values
        weights: Array of item weights
        capacity: Knapsack capacity
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random values and weights
    values = np.random.randint(1, 100, size=n_items)
    weights = np.random.randint(1, 100, size=n_items)
    
    # Set capacity based on ratio (default 50% if not specified)
    if capacity_ratio is None:
        capacity_ratio = 0.5
    capacity = int(np.sum(weights) * capacity_ratio)
    
    return values, weights, capacity


def solve_knapsack_dp(values: np.ndarray, weights: np.ndarray, capacity: int) -> Tuple[int, np.ndarray]:
    """
    Solve 0/1 Knapsack problem using dynamic programming.
    
    Args:
        values: Array of item values
        weights: Array of item weights
        capacity: Knapsack capacity
    
    Returns:
        optimal_value: Maximum value achievable
        solution: Binary array indicating which items to include
    """
    n = len(values)
    
    # DP table: dp[i][w] = max value using first i items with capacity w
    dp = np.zeros((n + 1, capacity + 1), dtype=np.int32)
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i-1
            dp[i][w] = dp[i-1][w]
            
            # Take item i-1 if possible
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
    
    # Backtrack to find which items were selected
    solution = np.zeros(n, dtype=np.int32)
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            solution[i-1] = 1
            w -= weights[i-1]
    
    optimal_value = dp[n][capacity]
    return optimal_value, solution


# ==============================================================================
# NEURAL NETWORK MODEL (DeepSets-style)
# ==============================================================================

class KnapsackNet(nn.Module):
    """
    DeepSets-style neural network for knapsack problem.
    
    Architecture:
    1. Each item is embedded independently (permutation equivariant)
    2. Global pooling aggregates information
    3. Per-item decoder predicts inclusion probability
    """
    
    def __init__(self, hidden_dim: int = 128):
        super(KnapsackNet, self).__init__()
        
        # Encoder: maps (value, weight, capacity + engineered features) -> hidden representation
        # Input: 3 raw features + 4 engineered features = 7 total
        self.item_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Decoder: maps (item_encoding + global_encoding) -> logits (no sigmoid for stability)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, values: torch.Tensor, weights: torch.Tensor, capacity: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional masking for variable-length sequences.
        
        Args:
            values: (batch_size, n_items) - item values (padded if variable length)
            weights: (batch_size, n_items) - item weights (padded if variable length)
            capacity: (batch_size, 1) - knapsack capacity
            mask: (batch_size, n_items) - binary mask (1 = real item, 0 = padding)
        
        Returns:
            logits: (batch_size, n_items) - raw logits (apply sigmoid for probabilities)
        """
        batch_size, n_items = values.shape
        device = values.device
        eps = 1e-8  # Small epsilon to avoid division by zero
        
        # Create default mask if not provided (all items are real)
        if mask is None:
            mask = torch.ones_like(values)
        
        # Expand capacity to match each item
        capacity_expanded = capacity.unsqueeze(1).expand(-1, n_items, -1)  # (batch, n_items, 1)
        
        # Create raw features: [value, weight, capacity]
        raw_features = torch.stack([values, weights], dim=-1)  # (batch, n_items, 2)
        raw_features = torch.cat([raw_features, capacity_expanded], dim=-1)  # (batch, n_items, 3)
        
        # Compute engineered features for better learning
        # 1. Value-to-weight ratio (efficiency)
        value_weight_ratio = values / (weights + eps)
        
        # 2. Weight-to-capacity ratio (relative size)
        weight_capacity_ratio = weights / (capacity.squeeze(1).unsqueeze(1) + eps)
        
        # 3. Value relative to total value (importance)
        total_value = values.sum(dim=1, keepdim=True) + eps
        value_relative = values / total_value
        
        # 4. Weight relative to total weight (size proportion)
        total_weight = weights.sum(dim=1, keepdim=True) + eps
        weight_relative = weights / total_weight
        
        # Stack all features
        engineered_features = torch.stack([
            value_weight_ratio,
            weight_capacity_ratio,
            value_relative,
            weight_relative
        ], dim=-1)  # (batch, n_items, 4)
        
        # Combine raw and engineered features
        item_features = torch.cat([raw_features, engineered_features], dim=-1)  # (batch, n_items, 7)
        
        # Encode each item independently
        item_encodings = self.item_encoder(item_features)  # (batch, n_items, hidden)
        
        # Global pooling (mean) - aggregates information across all items
        # Apply mask to ignore padded items in pooling
        mask_expanded = mask.unsqueeze(-1)  # (batch, n_items, 1)
        masked_encodings = item_encodings * mask_expanded
        sum_encodings = torch.sum(masked_encodings, dim=1, keepdim=True)  # (batch, 1, hidden)
        count_items = torch.sum(mask_expanded, dim=1, keepdim=True).clamp(min=1)  # Avoid div by 0
        global_encoding = sum_encodings / count_items  # (batch, 1, hidden)
        global_encoding = global_encoding.expand(-1, n_items, -1)  # (batch, n_items, hidden)
        
        # Concatenate item and global encodings
        combined = torch.cat([item_encodings, global_encoding], dim=-1)  # (batch, n_items, 2*hidden)
        
        # Decode to logits (no sigmoid for numerical stability)
        logits = self.decoder(combined).squeeze(-1)  # (batch, n_items)
        
        # Mask out padded items by setting their logits to very negative (will sigmoid to ~0)
        logits = logits.masked_fill(mask == 0, -1e9)
        
        return logits


# ==============================================================================
# LOSS FUNCTION WITH REGRET
# ==============================================================================

def greedy_decode(probs: torch.Tensor, weights: torch.Tensor, capacity: torch.Tensor, 
                  use_local_search: bool = False, values_for_local_search: torch.Tensor = None) -> torch.Tensor:
    """
    Greedily decode probabilities to feasible solution.
    Sort items by predicted probability (descending) and add while under capacity.
    
    Args:
        probs: (batch_size, n_items) - predicted probabilities
        weights: (batch_size, n_items) - item weights
        capacity: (batch_size, 1) - knapsack capacity
        use_local_search: If True, apply 2-swap local search after greedy
        values_for_local_search: Real item values to use in local search (if None, uses probs)
    
    Returns:
        solution: (batch_size, n_items) - binary solution
    """
    batch_size, n_items = probs.shape
    device = probs.device
    
    # Clamp probabilities to avoid NaN/inf issues
    probs = probs.clamp(1e-6, 1 - 1e-6)
    
    # Sort items by probability (descending)
    sorted_probs, sorted_indices = torch.sort(probs, dim=1, descending=True)
    
    # Initialize solutions
    solutions = torch.zeros_like(probs)
    
    # For each instance in batch
    for b in range(batch_size):
        current_weight = 0
        cap = capacity[b, 0].item()
        
        for idx in sorted_indices[b]:
            item_weight = weights[b, idx].item()
            if current_weight + item_weight <= cap:
                solutions[b, idx] = 1
                current_weight += item_weight
    
    # Apply 2-swap local search if requested
    if use_local_search:
        # CRITICAL: Use real item values for local search, not probabilities!
        base_values = values_for_local_search if values_for_local_search is not None else probs
        solutions = two_swap_local_search(solutions, weights, capacity, base_values)
    
    return solutions


def two_swap_local_search(solutions: torch.Tensor, weights: torch.Tensor, 
                          capacity: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Apply 2-swap local search to improve solutions.
    Try swapping pairs of items (one in, one out) to increase value while maintaining feasibility.
    
    Args:
        solutions: (batch_size, n_items) - current binary solutions
        weights: (batch_size, n_items) - item weights
        capacity: (batch_size, 1) - knapsack capacity
        values: (batch_size, n_items) - item values (or probabilities)
    
    Returns:
        improved_solutions: (batch_size, n_items) - improved solutions
    """
    batch_size, n_items = solutions.shape
    improved = solutions.clone()
    
    for b in range(batch_size):
        current_solution = improved[b]
        current_weight = torch.sum(current_solution * weights[b]).item()
        current_value = torch.sum(current_solution * values[b]).item()
        cap = capacity[b, 0].item()
        
        improved_local = False
        # Try all pairs: swap item i (in) with item j (out)
        for i in range(n_items):
            if current_solution[i] == 0:  # Item i is not selected
                for j in range(n_items):
                    if current_solution[j] == 1:  # Item j is selected
                        # Try swap: remove j, add i
                        new_weight = current_weight - weights[b, j].item() + weights[b, i].item()
                        new_value = current_value - values[b, j].item() + values[b, i].item()
                        
                        # Check if swap is beneficial and feasible
                        if new_weight <= cap and new_value > current_value:
                            current_solution[j] = 0
                            current_solution[i] = 1
                            current_weight = new_weight
                            current_value = new_value
                            improved_local = True
                            break
                if improved_local:
                    break
    
    return improved


def compute_value(solution: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Compute total value of a solution.
    
    Args:
        solution: (batch_size, n_items) - binary solution
        values: (batch_size, n_items) - item values
    
    Returns:
        total_value: (batch_size,) - total value
    """
    return torch.sum(solution * values, dim=1)


def knapsack_loss(logits: torch.Tensor, optimal_solution: torch.Tensor, 
                  values: torch.Tensor, weights: torch.Tensor, 
                  capacity: torch.Tensor, optimal_values: torch.Tensor,
                  regret_weight: float = 1.0, pos_weight: torch.Tensor = None) -> torch.Tensor:
    """
    Combined loss: Binary Cross-Entropy with Logits + Regret Loss.
    
    Args:
        logits: Predicted logits (raw, no sigmoid applied)
        optimal_solution: Ground truth optimal solution
        values: Item values
        weights: Item weights
        capacity: Knapsack capacity
        optimal_values: Optimal values for each instance
        regret_weight: Weight for regret loss component
        pos_weight: Optional weight for positive class (for imbalanced data)
    
    Returns:
        total_loss: Combined loss value
    """
    # Binary Cross-Entropy Loss with Logits (more numerically stable)
    bce_loss = nn.functional.binary_cross_entropy_with_logits(
        logits, optimal_solution.float(), pos_weight=pos_weight
    )
    
    # Apply sigmoid to get probabilities for greedy decode only
    probs = torch.sigmoid(logits)
    
    # Decode to feasible solution (no local search during training for speed)
    predicted_solution = greedy_decode(probs, weights, capacity, use_local_search=False)
    
    # Compute predicted value
    predicted_values = compute_value(predicted_solution, values)
    
    # Regret = max(0, optimal_value - predicted_value)
    regret = torch.clamp(optimal_values - predicted_values, min=0)
    regret_loss = torch.mean(regret)
    
    # Combined loss
    total_loss = bce_loss + regret_weight * regret_loss
    
    return total_loss


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_model(model: KnapsackNet, n_epochs: int = 100, batch_size: int = 64, 
                n_items_range: Tuple[int, int] = (20, 20), lr: float = 0.001, regret_weight: float = 1.0,
                save_path: str = "knapsack_model.pt", use_pos_weight: bool = False,
                use_curriculum: bool = True, capacity_range: Tuple[float, float] = (0.3, 0.7)):
    """
    Train the knapsack neural network with variable problem sizes and curriculum learning.
    
    Args:
        model: Neural network model
        n_epochs: Number of training epochs
        batch_size: Batch size
        n_items_range: (min_items, max_items) for variable-N training
        lr: Learning rate
        regret_weight: Weight for regret loss
        save_path: Path to save the trained model
        use_pos_weight: If True, use class balancing in BCE loss
        use_curriculum: If True, gradually increase problem size during training
        capacity_range: (min_ratio, max_ratio) for capacity tightness variation
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    
    n_min, n_max = n_items_range
    cap_min, cap_max = capacity_range
    
    print("Starting training with variable problem sizes...")
    print(f"Epochs: {n_epochs}, Batch Size: {batch_size}")
    print(f"Items Range: N ∈ [{n_min}, {n_max}]")
    print(f"Capacity Range: C ∈ [{cap_min:.1%}, {cap_max:.1%}] of total weight")
    print(f"Curriculum Learning: {'Enabled' if use_curriculum else 'Disabled'}")
    print(f"Class balancing (pos_weight): {'Enabled' if use_pos_weight else 'Disabled'}")
    print("-" * 60)
    
    for epoch in range(n_epochs):
        model.train()
        
        # Curriculum: gradually expand N range
        if use_curriculum:
            # Start with smaller range, expand over training
            progress = epoch / n_epochs
            current_n_min = n_min
            current_n_max = n_min + int((n_max - n_min) * progress)
        else:
            current_n_min, current_n_max = n_min, n_max
        
        # Generate batch of random instances with variable sizes
        values_batch = []
        weights_batch = []
        capacity_batch = []
        optimal_solutions = []
        optimal_vals = []
        n_items_batch = []
        
        for _ in range(batch_size):
            # Random problem size
            n_items = np.random.randint(current_n_min, current_n_max + 1)
            n_items_batch.append(n_items)
            
            # Random capacity ratio
            cap_ratio = np.random.uniform(cap_min, cap_max)
            
            # Generate instance
            values, weights, capacity = generate_knapsack_instance(n_items, capacity_ratio=cap_ratio)
            opt_val, opt_sol = solve_knapsack_dp(values, weights, capacity)
            
            values_batch.append(values)
            weights_batch.append(weights)
            capacity_batch.append(capacity)
            optimal_solutions.append(opt_sol)
            optimal_vals.append(opt_val)
        
        # Pad to max length in batch
        max_items = max(n_items_batch)
        values_padded = []
        weights_padded = []
        solutions_padded = []
        masks = []
        
        for i in range(batch_size):
            n = n_items_batch[i]
            # Pad with zeros
            values_pad = np.pad(values_batch[i], (0, max_items - n), constant_values=0)
            weights_pad = np.pad(weights_batch[i], (0, max_items - n), constant_values=0)
            solution_pad = np.pad(optimal_solutions[i], (0, max_items - n), constant_values=0)
            # Create mask (1 for real items, 0 for padding)
            mask = np.array([1] * n + [0] * (max_items - n))
            
            values_padded.append(values_pad)
            weights_padded.append(weights_pad)
            solutions_padded.append(solution_pad)
            masks.append(mask)
        
        # Convert to tensors
        values_tensor = torch.FloatTensor(np.array(values_padded)).to(device)
        weights_tensor = torch.FloatTensor(np.array(weights_padded)).to(device)
        capacity_tensor = torch.FloatTensor(np.array(capacity_batch)).unsqueeze(1).to(device)
        optimal_solution_tensor = torch.FloatTensor(np.array(solutions_padded)).to(device)
        optimal_vals_tensor = torch.FloatTensor(np.array(optimal_vals)).to(device)
        mask_tensor = torch.FloatTensor(np.array(masks)).to(device)
        
        # Compute pos_weight for class imbalance if requested
        pos_weight_tensor = None
        if use_pos_weight:
            # Only count real (non-padded) items
            n_pos = (optimal_solution_tensor * mask_tensor).sum()
            n_neg = mask_tensor.sum() - n_pos
            if n_pos > 0:
                pos_weight_tensor = (n_neg / n_pos).unsqueeze(0).to(device)
        
        # Forward pass (returns logits) with mask
        logits = model(values_tensor, weights_tensor, capacity_tensor, mask=mask_tensor)
        
        # Compute loss (mask is applied in model forward, so logits for padding are already very negative)
        loss = knapsack_loss(logits, optimal_solution_tensor, values_tensor, 
                            weights_tensor, capacity_tensor, optimal_vals_tensor,
                            regret_weight=regret_weight, pos_weight=pos_weight_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress with current N range
        if (epoch + 1) % 10 == 0:
            avg_n = np.mean(n_items_batch)
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}, Avg N: {avg_n:.1f}, "
                  f"N Range: [{current_n_min}, {current_n_max}]")
    
    print("-" * 60)
    print("Training completed!")
    
    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
    print("-" * 60)


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_model(model: KnapsackNet, n_test: int = 100, n_items: int = 20, 
                   use_local_search: bool = True, capacity_ratio: float = 0.5):
    """
    Evaluate the trained model on new random instances.
    
    Args:
        model: Trained neural network model
        n_test: Number of test instances
        n_items: Number of items per instance
        use_local_search: Whether to apply 2-swap local search
        capacity_ratio: Capacity tightness (0.3=tight, 0.5=balanced, 0.7=loose)
    """
    import time
    
    model.eval()
    device = next(model.parameters()).device
    
    print(f"\nEvaluating model on test instances (N={n_items}, capacity={capacity_ratio:.1%})...")
    print(f"Local Search: {'Enabled' if use_local_search else 'Disabled'}")
    print("-" * 60)
    
    gaps = []
    feasibility_count = 0
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(n_test):
            # Generate test instance with specified capacity ratio
            values, weights, capacity = generate_knapsack_instance(
                n_items, seed=10000 + i, capacity_ratio=capacity_ratio
            )
            opt_val, opt_sol = solve_knapsack_dp(values, weights, capacity)
            
            # Convert to tensors
            values_tensor = torch.FloatTensor(values).unsqueeze(0).to(device)
            weights_tensor = torch.FloatTensor(weights).unsqueeze(0).to(device)
            capacity_tensor = torch.FloatTensor([capacity]).unsqueeze(0).to(device)
            
            # Predict (returns logits) - no mask needed for fixed-size eval
            logits = model(values_tensor, weights_tensor, capacity_tensor)
            probs = torch.sigmoid(logits)  # Convert to probabilities
            
            # Decode to solution (CRITICAL: pass real values for local search!)
            predicted_solution = greedy_decode(
                probs, weights_tensor, capacity_tensor, 
                use_local_search=use_local_search,
                values_for_local_search=values_tensor  # Use real item values, not probs!
            )
            predicted_val = compute_value(predicted_solution, values_tensor).item()
            
            # Check feasibility
            predicted_weight = torch.sum(predicted_solution * weights_tensor, dim=1).item()
            is_feasible = predicted_weight <= capacity
            if is_feasible:
                feasibility_count += 1
            
            # Compute gap
            gap = (opt_val - predicted_val) / opt_val * 100 if opt_val > 0 else 0
            gaps.append(gap)
            
            # Print first 3 instances
            if i < 3:
                print(f"Instance {i + 1}: N={n_items}, Cap={capacity_ratio:.1%}")
                print(f"  Predicted Value: {predicted_val:.0f}")
                print(f"  Optimal Value:   {opt_val}")
                print(f"  Relative Gap:    {gap:.2f}%")
                print(f"  Feasible:        {'Yes' if is_feasible else 'No'}")
                print()
    
    elapsed_time = time.time() - start_time
    
    # Summary statistics
    print("-" * 60)
    print(f"EVALUATION SUMMARY (N={n_items}, Cap={capacity_ratio:.1%})")
    print("-" * 60)
    print(f"Test Instances:  {n_test}")
    print(f"Feasibility:     {feasibility_count}/{n_test} ({feasibility_count/n_test*100:.1f}%)")
    print(f"Evaluation Time: {elapsed_time:.2f}s ({elapsed_time/n_test*1000:.1f}ms per instance)")
    print()
    print(f"Average Gap:  {np.mean(gaps):.2f}%")
    print(f"Median Gap:   {np.median(gaps):.2f}%")
    print(f"Min Gap:      {np.min(gaps):.2f}%")
    print(f"Max Gap:      {np.max(gaps):.2f}%")
    print(f"Std Dev:      {np.std(gaps):.2f}%")
    print("-" * 60)
    
    return {
        'avg_gap': float(np.mean(gaps)),
        'median_gap': float(np.median(gaps)),
        'feasibility': feasibility_count / n_test,
        'time_ms': elapsed_time / n_test * 1000
    }


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main function to orchestrate training and evaluation.
    """
    import json
    import time
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Enable deterministic behavior for reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Hyperparameters
    N_ITEMS_MIN = 10      # Minimum number of items (variable-N training)
    N_ITEMS_MAX = 60      # Maximum number of items (variable-N training)
    HIDDEN_DIM = 128      # Hidden dimension for neural network
    N_EPOCHS = 100        # Number of training epochs
    BATCH_SIZE = 64       # Training batch size
    LEARNING_RATE = 0.001 # Learning rate
    REGRET_WEIGHT = 1.0   # Weight for regret loss component
    N_TEST = 50           # Number of test instances per configuration
    USE_POS_WEIGHT = False  # Use class balancing in loss
    USE_CURRICULUM = True   # Gradually increase problem size during training
    CAPACITY_MIN = 0.3    # Minimum capacity ratio (tight knapsacks)
    CAPACITY_MAX = 0.7    # Maximum capacity ratio (loose knapsacks)
    MODEL_PATH = "knapsack_model_variable.pt"  # Path to save/load model
    CONFIG_PATH = "knapsack_config_variable.json"  # Path to save config
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Deterministic mode: Enabled")
    print()
    
    # Initialize model
    model = KnapsackNet(hidden_dim=HIDDEN_DIM).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print()
    
    # Save configuration
    config = {
        "n_items_min": N_ITEMS_MIN,
        "n_items_max": N_ITEMS_MAX,
        "hidden_dim": HIDDEN_DIM,
        "n_epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "regret_weight": REGRET_WEIGHT,
        "use_pos_weight": USE_POS_WEIGHT,
        "use_curriculum": USE_CURRICULUM,
        "capacity_min": CAPACITY_MIN,
        "capacity_max": CAPACITY_MAX,
        "device": str(device),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {CONFIG_PATH}\n")
    
    # Train model with variable N and capacity
    train_model(
        model=model,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        n_items_range=(N_ITEMS_MIN, N_ITEMS_MAX),
        lr=LEARNING_RATE,
        regret_weight=REGRET_WEIGHT,
        save_path=MODEL_PATH,
        use_pos_weight=USE_POS_WEIGHT,
        use_curriculum=USE_CURRICULUM,
        capacity_range=(CAPACITY_MIN, CAPACITY_MAX)
    )
    
    # Comprehensive evaluation across multiple N and capacity settings
    print("\n" + "=" * 70)
    print("COMPREHENSIVE EVALUATION: VARIABLE N AND CAPACITY")
    print("=" * 70)
    
    # Test different problem sizes
    test_sizes = [10, 20, 30, 40, 50, 60]
    # Test different capacity tightness
    capacity_configs = [
        (0.3, "Tight"),
        (0.5, "Balanced"),
        (0.7, "Loose")
    ]
    
    all_results = {}
    
    # Evaluate across N and capacity combinations
    for n_size in test_sizes:
        size_results = {}
        print(f"\n{'='*70}")
        print(f"Testing N={n_size}")
        print(f"{'='*70}")
        
        for cap_ratio, cap_name in capacity_configs:
            print(f"\n--- {cap_name} Capacity ({cap_ratio:.1%}) ---")
            result = evaluate_model(
                model=model,
                n_test=N_TEST,
                n_items=n_size,
                use_local_search=True,
                capacity_ratio=cap_ratio
            )
            size_results[cap_name.lower()] = result
        
        all_results[f"n{n_size}"] = size_results
        
        # Print summary for this N
        avg_gap = np.mean([r['avg_gap'] for r in size_results.values()])
        avg_feas = np.mean([r['feasibility'] for r in size_results.values()])
        print(f"\nSummary for N={n_size}:")
        print(f"  Overall Avg Gap: {avg_gap:.2f}%")
        print(f"  Overall Feasibility: {avg_feas*100:.1f}%")
    
    # Save comprehensive metrics
    results = {
        "training_config": {
            "n_range": [N_ITEMS_MIN, N_ITEMS_MAX],
            "capacity_range": [CAPACITY_MIN, CAPACITY_MAX],
            "curriculum": USE_CURRICULUM
        },
        "evaluation_results": all_results,
        "config_file": CONFIG_PATH,
        "model_file": MODEL_PATH
    }
    
    results_path = "knapsack_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Compute overall statistics
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: GAP vs N ANALYSIS")
    print("=" * 70)
    
    print(f"\n{'N':<6} {'Tight':<10} {'Balanced':<12} {'Loose':<10} {'Average':<10}")
    print("-" * 60)
    for n_size in test_sizes:
        key = f"n{n_size}"
        tight_gap = all_results[key]['tight']['avg_gap']
        balanced_gap = all_results[key]['balanced']['avg_gap']
        loose_gap = all_results[key]['loose']['avg_gap']
        avg_gap = (tight_gap + balanced_gap + loose_gap) / 3
        print(f"{n_size:<6} {tight_gap:>6.2f}%    {balanced_gap:>6.2f}%      {loose_gap:>6.2f}%    {avg_gap:>6.2f}%")
    
    print("\n" + "=" * 70)
    print("ALL EVALUATIONS COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Config saved to: {CONFIG_PATH}")
    print(f"Results saved to: {results_path}")
    print("\nKey Improvements:")
    print("  ✓ Variable-N training (N ∈ [10, 60])")
    print("  ✓ Curriculum learning (gradual difficulty increase)")
    print("  ✓ Variable capacity tightness (30-70%)")
    print("  ✓ Padding + masking for variable-length batching")
    print("  ✓ Comprehensive evaluation across N and capacity")
    print("\nTo load the model later, use:")
    print(f"  model = KnapsackNet(hidden_dim={HIDDEN_DIM})")
    print(f"  model.load_state_dict(torch.load('{MODEL_PATH}'))")
    print(f"  model.eval()")
    print("=" * 70)


if __name__ == "__main__":
    main()
