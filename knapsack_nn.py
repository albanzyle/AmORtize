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

def generate_knapsack_instance(n_items: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate a random knapsack instance.
    
    Args:
        n_items: Number of items
        seed: Random seed for reproducibility
    
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
    
    # Set capacity to roughly 50% of total weight
    capacity = int(np.sum(weights) * 0.5)
    
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
    
    def forward(self, values: torch.Tensor, weights: torch.Tensor, capacity: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            values: (batch_size, n_items) - item values
            weights: (batch_size, n_items) - item weights
            capacity: (batch_size, 1) - knapsack capacity
        
        Returns:
            logits: (batch_size, n_items) - raw logits (apply sigmoid for probabilities)
        """
        batch_size, n_items = values.shape
        device = values.device
        eps = 1e-8  # Small epsilon to avoid division by zero
        
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
        global_encoding = torch.mean(item_encodings, dim=1, keepdim=True)  # (batch, 1, hidden)
        global_encoding = global_encoding.expand(-1, n_items, -1)  # (batch, n_items, hidden)
        
        # Concatenate item and global encodings
        combined = torch.cat([item_encodings, global_encoding], dim=-1)  # (batch, n_items, 2*hidden)
        
        # Decode to logits (no sigmoid for numerical stability)
        logits = self.decoder(combined).squeeze(-1)  # (batch, n_items)
        
        return logits


# ==============================================================================
# LOSS FUNCTION WITH REGRET
# ==============================================================================

def greedy_decode(probs: torch.Tensor, weights: torch.Tensor, capacity: torch.Tensor) -> torch.Tensor:
    """
    Greedily decode probabilities to feasible solution.
    Sort items by predicted probability (descending) and add while under capacity.
    
    Args:
        probs: (batch_size, n_items) - predicted probabilities
        weights: (batch_size, n_items) - item weights
        capacity: (batch_size, 1) - knapsack capacity
    
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
    
    return solutions


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
                  regret_weight: float = 1.0) -> torch.Tensor:
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
    
    Returns:
        total_loss: Combined loss value
    """
    # Binary Cross-Entropy Loss with Logits (more numerically stable)
    bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, optimal_solution.float())
    
    # Apply sigmoid to get probabilities for greedy decode only
    probs = torch.sigmoid(logits)
    
    # Decode to feasible solution
    predicted_solution = greedy_decode(probs, weights, capacity)
    
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
                n_items: int = 20, lr: float = 0.001, regret_weight: float = 1.0):
    """
    Train the knapsack neural network.
    
    Args:
        model: Neural network model
        n_epochs: Number of training epochs
        batch_size: Batch size
        n_items: Number of items per instance
        lr: Learning rate
        regret_weight: Weight for regret loss
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    
    print("Starting training...")
    print(f"Epochs: {n_epochs}, Batch Size: {batch_size}, Items: {n_items}")
    print("-" * 60)
    
    for epoch in range(n_epochs):
        model.train()
        
        # Generate batch of random instances
        values_batch = []
        weights_batch = []
        capacity_batch = []
        optimal_solutions = []
        optimal_vals = []
        
        for _ in range(batch_size):
            values, weights, capacity = generate_knapsack_instance(n_items)
            opt_val, opt_sol = solve_knapsack_dp(values, weights, capacity)
            
            values_batch.append(values)
            weights_batch.append(weights)
            capacity_batch.append(capacity)
            optimal_solutions.append(opt_sol)
            optimal_vals.append(opt_val)
        
        # Convert to tensors
        values_tensor = torch.FloatTensor(np.array(values_batch)).to(device)
        weights_tensor = torch.FloatTensor(np.array(weights_batch)).to(device)
        capacity_tensor = torch.FloatTensor(np.array(capacity_batch)).unsqueeze(1).to(device)
        optimal_solution_tensor = torch.FloatTensor(np.array(optimal_solutions)).to(device)
        optimal_vals_tensor = torch.FloatTensor(np.array(optimal_vals)).to(device)
        
        # Forward pass (returns logits)
        logits = model(values_tensor, weights_tensor, capacity_tensor)
        
        # Compute loss
        loss = knapsack_loss(logits, optimal_solution_tensor, values_tensor, 
                            weights_tensor, capacity_tensor, optimal_vals_tensor,
                            regret_weight=regret_weight)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    print("-" * 60)
    print("Training completed!")


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_model(model: KnapsackNet, n_test: int = 100, n_items: int = 20):
    """
    Evaluate the trained model on new random instances.
    
    Args:
        model: Trained neural network model
        n_test: Number of test instances
        n_items: Number of items per instance
    """
    import time
    
    model.eval()
    device = next(model.parameters()).device
    
    print("\nEvaluating model on test instances...")
    print("-" * 60)
    
    gaps = []
    feasibility_count = 0
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(n_test):
            # Generate test instance
            values, weights, capacity = generate_knapsack_instance(n_items, seed=10000 + i)
            opt_val, opt_sol = solve_knapsack_dp(values, weights, capacity)
            
            # Convert to tensors
            values_tensor = torch.FloatTensor(values).unsqueeze(0).to(device)
            weights_tensor = torch.FloatTensor(weights).unsqueeze(0).to(device)
            capacity_tensor = torch.FloatTensor([capacity]).unsqueeze(0).to(device)
            
            # Predict (returns logits)
            logits = model(values_tensor, weights_tensor, capacity_tensor)
            probs = torch.sigmoid(logits)  # Convert to probabilities
            
            # Decode to solution
            predicted_solution = greedy_decode(probs, weights_tensor, capacity_tensor)
            predicted_val = compute_value(predicted_solution, values_tensor).item()
            
            # Check feasibility
            predicted_weight = torch.sum(predicted_solution * weights_tensor, dim=1).item()
            is_feasible = predicted_weight <= capacity
            if is_feasible:
                feasibility_count += 1
            
            # Compute gap
            gap = (opt_val - predicted_val) / opt_val * 100 if opt_val > 0 else 0
            gaps.append(gap)
            
            # Print first 10 instances
            if i < 10:
                print(f"Instance {i + 1}:")
                print(f"  Predicted Value: {predicted_val:.0f}")
                print(f"  Optimal Value:   {opt_val}")
                print(f"  Relative Gap:    {gap:.2f}%")
                print(f"  Feasible:        {'Yes' if is_feasible else 'No'}")
                print()
    
    elapsed_time = time.time() - start_time
    
    # Summary statistics
    print("-" * 60)
    print("EVALUATION SUMMARY")
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


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main function to orchestrate training and evaluation.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Hyperparameters
    N_ITEMS = 20          # Number of items per knapsack instance
    HIDDEN_DIM = 128      # Hidden dimension for neural network
    N_EPOCHS = 100        # Number of training epochs
    BATCH_SIZE = 64       # Training batch size
    LEARNING_RATE = 0.001 # Learning rate
    REGRET_WEIGHT = 1.0   # Weight for regret loss component
    N_TEST = 100          # Number of test instances
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Initialize model
    model = KnapsackNet(hidden_dim=HIDDEN_DIM).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print()
    
    # Train model
    train_model(
        model=model,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        n_items=N_ITEMS,
        lr=LEARNING_RATE,
        regret_weight=REGRET_WEIGHT
    )
    
    # Evaluate model
    evaluate_model(
        model=model,
        n_test=N_TEST,
        n_items=N_ITEMS
    )


if __name__ == "__main__":
    main()
