"""
Inference Script for 2-Stage Op-Amp Optimization
Loads the trained PPO model and verifies the circuit performance.
"""

import os
from stable_baselines3 import PPO
from main import CircuitEnv  # Must import CircuitEnv from main.py

def run_inference():
    # Model Filename (Without .zip extension)
    MODEL_NAME = "model"

    if not os.path.exists(MODEL_NAME + ".zip"):
        print(f"[ERROR] Model file '{MODEL_NAME}.zip' not found.")
        return

    # 1. Load Environment & Model
    print(f">> Loading Model: {MODEL_NAME}.zip ...")
    env = CircuitEnv()
    model = PPO.load(MODEL_NAME, env=env)

    # 2. Predict Best Action (Deterministic)
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    # 3. Run Simulation with Predicted Parameters
    obs, reward, done, truncated, info = env.step(action)
    real_params = info['real_actions']

    # 4. Print Results
    print("\n" + "="*60)
    print(f"🏆 Final Optimization Results ({MODEL_NAME})")
    print("="*60)
    
    # Design Parameters
    print(f"1. Design Parameters (W / L / Cc)")
    print(f"   - M1,2 (Input):  W = {real_params[0]*1e6:6.2f} um  |  L = {real_params[5]*1e6:5.3f} um")
    print(f"   - M3,4 (Load):   W = {real_params[1]*1e6:6.2f} um  |  L = {real_params[6]*1e6:5.3f} um")
    print(f"   - M5   (Tail):   W = {real_params[2]*1e6:6.2f} um  |  L = {real_params[7]*1e6:5.3f} um")
    print(f"   - M6   (CS):     W = {real_params[3]*1e6:6.2f} um  |  L = {real_params[8]*1e6:5.3f} um")
    print(f"   - M7   (Bias):   W = {real_params[4]*1e6:6.2f} um  |  L = {real_params[9]*1e6:5.3f} um")
    print(f"   - Cc:            {real_params[10]*1e12:.4f} pF")

    # Simulation Performance
    print(f"\n2. Simulation Performance")
    print(f"   - DC Gain:       {info['gain']:7.2f} dB    (Target: > 60 dB)")
    print(f"   - Phase Margin:  {info['pm']:7.2f} deg   (Target: > 60 deg)")
    print(f"   - UGBW:          {info['ugbw']/1e6:7.2f} MHz   (Target: > 50 MHz)")
    print(f"   - Power:         {info['power']*1000:7.4f} mW    (Target: < 0.8 mW)")
    print(f"   - Area:          {info['area']:7.2f} um²   (Target: < 1000 um²)")
    print("="*60)

if __name__ == "__main__":
    run_inference()