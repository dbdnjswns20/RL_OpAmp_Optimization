"""
Deep Reinforcement Learning for 180nm 2-Stage Op-Amp Optimization
Course: CSE5516-01 (Reinforcement Learning)
Author: 120250652 Wonjun Yu
Date: 2025.12.07
"""

import os
import csv
import gc
import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# PySpice & NgSpice
import PySpice.Logging.Logging as Logging
import PySpice.Spice.NgSpice.Shared as NgSpiceShared
from PySpice.Spice.Netlist import Circuit

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ==============================================================================
# 0. System Configuration & Logging
# ==============================================================================
# Prevent Windows OS conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set NgSpice DLL Path (Relative to main.py)
current_path = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(current_path, "ngspice.dll")
NgSpiceShared.NgSpiceShared.LIBRARY_PATH = dll_path

# Suppress PySpice/NgSpice logs
logger_pyspice = Logging.setup_logging()
logging.getLogger("PySpice.Spice.NgSpice.Shared.NgSpiceShared").setLevel(logging.CRITICAL)


# ==============================================================================
# 1. Simulation Engine (NgSpice)
# ==============================================================================
def calculate_phase_margin(freqs, gains_db, phases_deg):
    """Calculates Phase Margin (PM) from AC analysis results."""
    try:
        # Find UGBW (Unity Gain Bandwidth) index where Gain crosses 0dB
        idx = np.where(np.diff(np.sign(gains_db)))[0]
        if len(idx) == 0: return -180.0
        
        ugbw_idx = idx[0]
        phase_at_ugbw = phases_deg[ugbw_idx]
        
        # Normalize phase to -180 ~ 180
        pm = 180 + phase_at_ugbw
        while pm > 180: pm -= 360
        while pm < -180: pm += 360
        return pm
    except:
        return 0.0

def run_simulation(w_dict, l_dict, cc_val):
    """Generates Netlist and runs AC/DC simulation."""
    try:
        circuit = Circuit('Two-Stage OpAmp')

        # Load Process Library (180nm)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(script_dir, "180nm_bulk.lib")
        abs_path = lib_path.replace("\\", "/")
        
        if not os.path.exists(lib_path):
            return {"gain": -100.0, "power": 0.02, "pm": 0.0, "ugbw": 0.0}
        
        circuit.raw_spice += f".include \"{abs_path}\"\n"
        
        # Simulation Options for Convergence
        circuit.raw_spice += ".options savecurrents\n"
        circuit.raw_spice += ".options reltol=0.01 vntol=100uV abstol=1pA\n"
        circuit.raw_spice += ".options method=gear\n"
        circuit.raw_spice += ".options rshunt=1.0e12\n"

        # Netlist Definition
        circuit.V('dd', 'vdd', '0', 1.8)
        circuit.V('in_p', 'vip', '0', 'dc 0.9 ac 0.5')
        circuit.V('in_n', 'vin', '0', 'dc 0.9 ac -0.5')
        circuit.V('b', 'vbias', '0', 0.7)

        def get_mos(name, d, g, s, b, model, w, l):
            L_diff = 0.5e-6; ad = w*L_diff; pd = 2*(w+L_diff)
            return f"M{name} {d} {g} {s} {b} {model} w={w} l={l} ad={ad} as={ad} pd={pd} ps={pd}\n"

        # 2-Stage OpAmp Topology
        circuit.raw_spice += get_mos('1', 'node_x', 'vin', 'tail', '0', 'nmos', w_dict['m1'], l_dict['m1'])
        circuit.raw_spice += get_mos('2', 'node_y', 'vip', 'tail', '0', 'nmos', w_dict['m2'], l_dict['m2'])
        circuit.raw_spice += get_mos('3', 'node_x', 'node_x', 'vdd', 'vdd', 'pmos', w_dict['m3'], l_dict['m3'])
        circuit.raw_spice += get_mos('4', 'node_y', 'node_x', 'vdd', 'vdd', 'pmos', w_dict['m4'], l_dict['m4'])
        circuit.raw_spice += get_mos('5', 'tail', 'vbias', '0', '0', 'nmos', w_dict['m5'], l_dict['m5'])
        circuit.raw_spice += get_mos('6', 'out', 'node_y', 'vdd', 'vdd', 'pmos', w_dict['m6'], l_dict['m6'])
        circuit.raw_spice += get_mos('7', 'out', 'vbias', '0', '0', 'nmos', w_dict['m7'], l_dict['m7'])

        circuit.C('c', 'node_y', 'out', cc_val)
        circuit.C('L', 'out', '0', 1e-12)

        # Run Simulation
        simulator = circuit.simulator()
        simulator.operating_point() # DC Analysis
        ac = simulator.ac(start_frequency=1, stop_frequency=100e6, number_of_points=50, variation='dec') # AC Analysis

        # Extract Metrics
        freqs = np.array(ac.frequency)
        out_node = ac.nodes['out'] if 'out' in ac.nodes else list(ac.nodes.values())[0]
        out_val = np.array(out_node)
        
        gain_db = 20 * np.log10(np.abs(out_val) + 1e-15)
        dc_gain = gain_db[0]
        
        phase_deg = np.degrees(np.angle(out_val))
        pm = calculate_phase_margin(np.array(ac.frequency), gain_db, phase_deg)

        idx = np.where(np.diff(np.sign(gain_db)))[0]
        ugbw = freqs[idx[0]] if len(idx) > 0 else 0.0

        # Calculate Power (Id_M5 + Id_M7) * VDD
        kp_n = 200e-6; vth = 0.45; vgs = 0.7
        i_m5 = 0.5 * kp_n * (w_dict['m5']/l_dict['m5']) * ((vgs-vth)**2)
        i_m7 = 0.5 * kp_n * (w_dict['m7']/l_dict['m7']) * ((vgs-vth)**2)
        power = 1.8 * (i_m5 + i_m7)

        if np.isnan(dc_gain) or np.isnan(pm) or np.isinf(dc_gain):
            return {"gain": -100.0, "power": 1.0, "pm": 0.0, "ugbw": 0.0}
            
        return {"gain": dc_gain, "power": power, "pm": pm, "ugbw": ugbw}

    except Exception:
        return {"gain": -100.0, "power": 1.0, "pm": 0.0, "ugbw": 0.0}


# ==============================================================================
# 2. Gymnasium Environment (RL Interface)
# ==============================================================================
class CircuitEnv(gym.Env):
    def __init__(self):
        super(CircuitEnv, self).__init__()
        
        # Design Space Definition (Lower/Upper bounds)
        self.w_low = [2e-6, 2e-6, 2e-6, 20e-6, 10e-6]
        self.w_high = [100e-6, 150e-6, 100e-6, 600e-6, 300e-6]
        self.l_low = [0.5e-6, 0.5e-6, 0.5e-6, 0.3e-6, 0.5e-6]
        self.l_high = [2.0e-6, 2.0e-6, 2.0e-6, 1.0e-6, 2.0e-6]
        self.cc_low = [0.5e-12]
        self.cc_high = [4.0e-12]

        self.real_low = np.array(self.w_low + self.l_low + self.cc_low, dtype=np.float32)
        self.real_high = np.array(self.w_high + self.l_high + self.cc_high, dtype=np.float32)

        # Action Space: Normalized [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(11,), dtype=np.float32)
        # Observation Space: [Gain, Power, PM, UGBW, Area]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def denormalize_action(self, action):
        """Converts normalized action [-1, 1] to physical values."""
        action = np.clip(action, -1.0, 1.0)
        real_val = self.real_low + (self.real_high - self.real_low) * (action + 1.0) / 2.0
        return np.round(real_val, 15)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start with nominal values
        nominal_action = np.zeros(11, dtype=np.float32)
        real_action = self.denormalize_action(nominal_action)
        
        # Map array to dictionary
        w_values, l_values, cc_val = real_action[0:5], real_action[5:10], real_action[10]
        w_dict = {'m1': w_values[0], 'm2': w_values[0], 'm3': w_values[1], 'm4': w_values[1],
                  'm5': w_values[2], 'm6': w_values[3], 'm7': w_values[4]}
        l_dict = {'m1': l_values[0], 'm2': l_values[0], 'm3': l_values[1], 'm4': l_values[1],
                  'm5': l_values[2], 'm6': l_values[3], 'm7': l_values[4]}
        
        result = run_simulation(w_dict, l_dict, cc_val)
        
        # Calculate Area
        area_sum = sum([w_dict[k] * l_dict[k] for k in w_dict]) * 1e12 
        # Note: Simplified area calculation for initial observation. 
        # Correct area is calculated in step() properly with multipliers.
        
        # Normalize Observation
        obs = np.array([result['gain']/100.0, result['power']*1000.0, result['pm']/100.0, 
                        result['ugbw']/1e7, area_sum/100.0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        real_action = self.denormalize_action(action)
        
        w_values, l_values, cc_val = real_action[0:5], real_action[5:10], real_action[10]
        w_dict = {'m1': w_values[0], 'm2': w_values[0], 'm3': w_values[1], 'm4': w_values[1],
                  'm5': w_values[2], 'm6': w_values[3], 'm7': w_values[4]}
        l_dict = {'m1': l_values[0], 'm2': l_values[0], 'm3': l_values[1], 'm4': l_values[1],
                  'm5': l_values[2], 'm6': l_values[3], 'm7': l_values[4]}
        
        # 1. Design Rule Check (DRC)
        is_safe = True
        if np.any(real_action[0:5] < 0.5e-6) or np.any(real_action[5:10] < 0.18e-6): is_safe = False
        # Ratio check (W/L)
        for k in w_dict:
            ratio = w_dict[k] / l_dict[k]
            if ratio < 0.1 or ratio > 2000.0: is_safe = False

        if not is_safe:
             return np.zeros(5, dtype=np.float32), -100.0, True, False, \
                    {"gain": -100, "power": 0.02, "pm": 0.0, "ugbw": 0.0, "area": 0.0, "real_actions": real_action}

        # 2. Run Simulation
        result = run_simulation(w_dict, l_dict, cc_val)
        gain, power, pm, ugbw = result['gain'], result['power'], result['pm'], result['ugbw']

        # 3. Calculate Area
        area_m1_m2 = (w_dict['m1'] * l_dict['m1']) * 2 * 1e12
        area_m3_m4 = (w_dict['m3'] * l_dict['m3']) * 2 * 1e12
        area_m5    = (w_dict['m5'] * l_dict['m5']) * 1 * 1e12
        area_m6    = (w_dict['m6'] * l_dict['m6']) * 1 * 1e12
        area_m7    = (w_dict['m7'] * l_dict['m7']) * 1 * 1e12
        total_active_area = area_m1_m2 + area_m3_m4 + area_m5 + area_m6 + area_m7

        # 4. Reward Logic (Two-Phase Strategy)
        # Target Specifications
        TARGET_GAIN, TARGET_PM, TARGET_UGBW = 60.0, 60.0, 50e6
        TARGET_POWER, TARGET_MOS_AREA = 0.8e-3, 1000.0

        # Score Calculations
        # (1) Gain Score
        if gain >= TARGET_GAIN: score_gain = 1.0 + 0.1 * np.tanh((gain - TARGET_GAIN) / 10.0)
        else: score_gain = 0.5 * ((np.maximum(gain, 0) / TARGET_GAIN) ** 4.0)

        # (2) Phase Margin Score
        pm_diff = np.abs(pm - TARGET_PM)
        if pm_diff <= 5.0: score_pm = 1.0 - (pm_diff / 25.0) 
        else: score_pm = np.maximum(0.0, 1.0 - (pm_diff / TARGET_PM)) ** 4.0
        if pm < 10.0: score_pm = -1.0

        # (3) UGBW Score
        score_ugbw = 1.0 if ugbw >= TARGET_UGBW else (ugbw / TARGET_UGBW) ** 4.0

        # (4) Check Specs
        specs_met = (gain >= TARGET_GAIN) and (pm_diff <= 5.0) and (ugbw >= TARGET_UGBW * 0.95)

        # (5) Power & Area Score (Bonus)
        if power <= TARGET_POWER: score_power = 1.0 + 1.0 * ((TARGET_POWER - power) / TARGET_POWER)
        else: score_power = 0.5 * np.exp(-(power - TARGET_POWER) / TARGET_POWER)

        score_area = 1.0 if total_active_area <= TARGET_MOS_AREA else \
                     max(0.0, 1.0 - (total_active_area - TARGET_MOS_AREA)/TARGET_MOS_AREA)

        # Final Reward Calculation
        if specs_met:
            # Phase 2: Optimization (Bonus for Power/Area)
            reward = (2.0 * score_gain) + (2.0 * score_pm) + (2.0 * score_ugbw) + \
                     (10.0 * score_power) + (4.0 * score_area) + 5.0
        else:
            # Phase 1: Survival (Focus on Specs)
            reward = (5.0 * score_gain) + (5.0 * score_pm) + (5.0 * score_ugbw) - 5.0
            
        # Penalties for failure
        if gain < 1.0 or pm < 0.0 or np.isnan(ugbw): reward = -10.0

        # 5. Observation Generation
        obs = np.array([gain/100.0, power*1000.0, pm/100.0, ugbw/1e7, total_active_area/100.0], dtype=np.float32)
        
        info = {
            "gain": gain, "power": power, "pm": pm, "ugbw": ugbw, "area": total_active_area,
            "real_actions": real_action
        }
        return obs, reward, True, False, info


# ==============================================================================
# 3. Main Execution & Logging
# ==============================================================================
class ExcelLoggerCallback(BaseCallback):
    """Custom callback to log training data to CSV."""
    def __init__(self, filename, verbose=0):
        super(ExcelLoggerCallback, self).__init__(verbose)
        self.filename = filename
        self.file = None
        self.writer = None

    def _on_training_start(self):
        file_exists = os.path.isfile(self.filename)
        self.file = open(self.filename, 'a', newline='', encoding='utf-8-sig')
        self.writer = csv.writer(self.file)
        
        if not file_exists:
            header = ["Step", "Reward", "Gain(dB)", "Power(mW)", "PM(deg)", "UGBW(MHz)", "Area(um2)",
                      "W_12", "W_34", "W_5", "W_6", "W_7", "L_12", "L_34", "L_5", "L_6", "L_7", "Cc(pF)"]
            self.writer.writerow(header)

    def _on_step(self) -> bool:
        infos = self.locals['infos'][0]
        ra = infos['real_actions']
        
        # Tensorboard Logging
        self.logger.record("circuit/Gain_dB", infos['gain'])
        self.logger.record("circuit/Power_mW", infos['power'] * 1000)
        self.logger.record("circuit/PhaseMargin_deg", infos['pm'])
        self.logger.record("circuit/UGBW_MHz", infos.get('ugbw', 0.0)/1e6)
        self.logger.record("circuit/Area_um2", infos.get('area', 0.0))

        # CSV Logging
        row = [
            self.num_timesteps, round(self.locals['rewards'][0], 4), 
            round(infos['gain'], 2), round(infos['power']*1000, 4), round(infos['pm'], 2), round(infos['ugbw']/1e6, 2), round(infos['area'], 2),
            round(ra[0]*1e6, 2), round(ra[1]*1e6, 2), round(ra[2]*1e6, 2), round(ra[3]*1e6, 2), round(ra[4]*1e6, 2),
            round(ra[5]*1e6, 3), round(ra[6]*1e6, 3), round(ra[7]*1e6, 3), round(ra[8]*1e6, 3), round(ra[9]*1e6, 3),
            round(ra[10]*1e12, 3)
        ]
        self.writer.writerow(row)
        self.file.flush()
        
        if self.num_timesteps % 10000 == 0: gc.collect()
        return True

    def _on_training_end(self):
        if self.file: self.file.close()

def main():
    # Directories
    save_dir = "saved_results"
    os.makedirs(save_dir, exist_ok=True)
    
    CHECKPOINT_NAME = "latest_checkpoint_new_PPO"
    CSV_NAME = "training_log.csv"
    TB_LOG_NAME = "2stage_opamp_ppo"
    
    checkpoint_path = os.path.join(save_dir, CHECKPOINT_NAME)
    csv_file_path = os.path.join(save_dir, CSV_NAME)
    LOG_DIR = "./ppo_2stage_logs/"

    # Initialize Env & Callback
    env = CircuitEnv()
    excel_callback = ExcelLoggerCallback(filename=csv_file_path)

    # Load or Create Model
    if os.path.exists(checkpoint_path + ".zip"):
        print(f"\n[INFO] Resuming Training from: {checkpoint_path}.zip")
        model = PPO.load(checkpoint_path, env=env, tensorboard_log=LOG_DIR)
    else:
        print("\n[INFO] Starting New Training...")
        policy_kwargs = dict(net_arch=[128, 128])
        model = PPO("MlpPolicy", env, verbose=1, 
                    learning_rate=0.01, n_steps=2048, batch_size=128, 
                    ent_coef=0.01, clip_range=0.1, 
                    policy_kwargs=policy_kwargs, tensorboard_log=LOG_DIR)

    # Start Training
    STEPS_PER_RUN = 51200
    print(f">> Training Goals: {STEPS_PER_RUN} Steps")
    
    try:
        model.learn(total_timesteps=STEPS_PER_RUN, progress_bar=True,
                    callback=excel_callback, reset_num_timesteps=False, tb_log_name=TB_LOG_NAME)
        
        model.save(checkpoint_path)
        print(f"\n[SUCCESS] Model Saved: {checkpoint_path}.zip")
        
    except Exception as e:
        print(f"[ERROR] Training Failed: {e}")
        model.save(checkpoint_path)

if __name__ == "__main__":
    main()