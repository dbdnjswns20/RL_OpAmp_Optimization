"""
Automatic Training Loop Script
Executes the main training script iteratively to manage memory and check for completion signals.
"""

import subprocess
import time
import sys
import os

# Configuration
SCRIPT_NAME = "main.py"            # Target script to run
STOP_SIGNAL_FILE = "TRAINING_COMPLETE" # Signal file created when target specs are met
TOTAL_ITERATIONS = 20              # Maximum number of re-runs

def main():
    # Remove existing signal file if it exists
    if os.path.exists(STOP_SIGNAL_FILE):
        os.remove(STOP_SIGNAL_FILE)

    print(f"🚀 Starting Auto-Training Loop (Max {TOTAL_ITERATIONS} Iterations)")
    print(f"📂 Target Script: {SCRIPT_NAME}\n")

    for i in range(1, TOTAL_ITERATIONS + 1):
        print(f"========================================")
        print(f"▶️ [Loop {i}/{TOTAL_ITERATIONS}] Running Training Script...")
        print(f"========================================")
        
        try:
            # Run the main script and wait for it to finish
            # check=True raises CalledProcessError if the script fails
            subprocess.run([sys.executable, SCRIPT_NAME], check=True)
            
            # Check for completion signal
            if os.path.exists(STOP_SIGNAL_FILE):
                print("\n========================================")
                print("🏆 Target Specifications Met!")
                print("🎉 Terminating Training Loop.")
                print("========================================")
                break

            print(f"\n✅ [Loop {i}] Finished. Memory cleared.")
            print("⏳ Restarting in 3 seconds...\n")
            time.sleep(3) 
            
        except subprocess.CalledProcessError:
            print(f"\n❌ [Error] Script crashed at Loop {i}.")
            print("⚠️ Stopping the loop.")
            break
        except KeyboardInterrupt:
            print("\n🛑 Stopped by User.")
            break

    print("🎉 All Training Loops Finished.")

if __name__ == "__main__":
    main()