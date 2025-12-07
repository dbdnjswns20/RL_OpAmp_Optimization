"""
Deep Reinforcement Learning Result Analyzer
Description: Analyzes the training log (CSV) to evaluate convergence and extract final design parameters.
"""

import pandas as pd
import os
import argparse
import sys

def analyze_final_design(csv_path, n_samples=5000):
    """
    Reads the training log CSV and calculates statistics (Mean, Std) 
    for the last N samples to verify convergence and extract design values.
    """
    
    # 1. File Existence Check
    if not os.path.exists(csv_path):
        print(f"\n[Error] File not found: {csv_path}")
        print("Please check the file path or run the training script first.\n")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"\n[Error] Failed to read CSV: {e}\n")
        return

    # 2. Adjust Sample Size if needed
    total_steps = len(df)
    if total_steps < n_samples:
        print(f"[Info] Not enough data points. Using all {total_steps} steps.")
        n_samples = total_steps
        
    # 3. Extract Last N Samples (Steady State Analysis)
    final_batch = df.tail(n_samples)
    
    # 4. Find the Best Performing Design (Max Reward) in the final batch
    best_design = final_batch.loc[final_batch['Reward'].idxmax()]

    print(f"\n" + "="*60)
    print(f"📊 Training Result Analysis")
    print(f"   - File: {csv_path}")
    print(f"   - Total Steps: {total_steps}")
    print(f"   - Analysis Window: Last {n_samples} steps")
    print("="*60)
    
    # --- Helper Function for Formatting ---
    def print_metric(name, key, unit=""):
        mean_val = final_batch[key].mean()
        std_val = final_batch[key].std()
        best_val = best_design[key]
        
        # Determine scale for display
        scale = 1.0
        display_unit = unit
        
        # Formatting
        print(f" {name:<15} | Mean: {mean_val*scale:8.3f} {display_unit:<4} (Std: {std_val*scale:6.3f}) | Best: {best_val*scale:8.3f} {display_unit}")

    # 5. Performance Metrics
    print(f"\n[1] Performance Metrics (Convergence Check)")
    print("-" * 60)
    print_metric("Reward", "Reward")
    print_metric("DC Gain", "Gain(dB)", "dB")
    print_metric("Phase Margin", "PM(deg)", "deg")
    print_metric("UGBW", "UGBW(MHz)", "MHz")
    print_metric("Power", "Power(mW)", "mW")
    print_metric("Area", "Area(um2)", "um2")

    # 6. Design Parameters
    print(f"\n[2] Optimized Design Parameters (W / L / Cc)")
    print("-" * 60)
    
    # Mosfet Widths
    print_metric("W1,2 (Input)", "W_12", "um")
    print_metric("W3,4 (Load)", "W_34", "um")
    print_metric("W5 (Tail)", "W_5", "um")
    print_metric("W6 (Stage2)", "W_6", "um")
    print_metric("W7 (Bias)", "W_7", "um")
    print("-" * 60)
    
    # Mosfet Lengths
    print_metric("L1,2", "L_12", "um")
    print_metric("L3,4", "L_34", "um")
    print_metric("L5", "L_5", "um")
    print_metric("L6", "L_6", "um")
    print_metric("L7", "L_7", "um")
    print("-" * 60)

    # Compensation Capacitor
    print_metric("Cc", "Cc(pF)", "pF")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    # Argument Parser for flexibility
    parser = argparse.ArgumentParser(description="Analyze RL Training Results for Op-Amp Design")
    
    # Default path matches the main.py save directory
    default_path = os.path.join("saved_results", "training_log.csv")
    
    parser.add_argument("--file", type=str, default=default_path, help="Path to the training log CSV file")
    parser.add_argument("--n", type=int, default=5000, help="Number of last samples to analyze")

    args = parser.parse_args()

    analyze_final_design(args.file, args.n)