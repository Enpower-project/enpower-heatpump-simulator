import math
from typing import Tuple, Dict, Any, Optional

import streamlit as st
import gymnasium as gym
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

PHYSICS_CONFIG = {
    "C_WALL_J_K": 2000.0 * 3600,    # Thermal capacity of the building envelope [J/K]
    "UA_WALL_W_K": 300.0,           # Overall thermal transmittance [W/K]
    "C_TANK_J_K": 200.0 * 4186,     # Thermal capacity of DHW tank (200L water) [J/K]
    "UA_TANK_W_K": 2.0,             # Tank thermal loss coefficient [W/K]
    "Q_HP_MAX_W": 10000.0,          # Rated Heat Pump thermal output [W]
    "TIME_STEP_SEC": 900,           # Simulation step (15 minutes)
    "SIMULATION_DAYS": 3            # Default simulation horizon
}

class HeatPumpDigitalTwin(gym.Env):
    def __init__(self, user_config: Dict[str, float]):
        super().__init__()

        self.cfg = user_config
        self.phys = PHYSICS_CONFIG
        
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -30.0]), 
            high=np.array([50.0, 95.0, 50.0]), 
            dtype=np.float32
        )

        self.current_step = 0
        self.steps_per_day = 24 * 4 
        self.max_steps = self.steps_per_day * self.phys["SIMULATION_DAYS"]
        
        self.state = np.zeros(3, dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        
        t_in_init = self.np_random.uniform(19.0, 21.0)
        t_tank_init = self.np_random.uniform(45.0, 55.0)
        t_ext_init = self._get_ambient_temp(0)
        
        self.state = np.array([t_in_init, t_tank_init, t_ext_init], dtype=np.float32)
        return self.state, {}

    def _get_ambient_temp(self, step: int) -> float:
        day_fraction = (step % self.steps_per_day) / self.steps_per_day
        return 7.0 + 5.0 * math.sin(2 * math.pi * (day_fraction - 0.25))

    def _get_dhw_demand(self, step: int) -> float:
        hour = (step % self.steps_per_day) / 4.0
        if (7.0 <= hour <= 8.0) or (19.0 <= hour <= 20.0):
            return 2500.0
        return 0.0

    def _calculate_cop(self, t_ambient: float, t_supply: float) -> float:
        # rough linear COP fit
        cop = 3.5 + 0.08 * (t_ambient - 7.0) - 0.05 * (t_supply - 35.0)
        return max(1.0, min(cop, 5.5))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        t_in, t_tank, _ = self.state
        u_signal = np.clip(action[0], 0.0, 1.0)
        q_thermal_generated = u_signal * self.phys["Q_HP_MAX_W"]

        t_ext = self._get_ambient_temp(self.current_step)

        q_to_tank = 0.0
        q_to_heating = 0.0
        target_supply_temp = 35.0 
        if t_tank < (self.cfg['t_set_dhw'] - 4.0):
            q_to_tank = q_thermal_generated
            target_supply_temp = 55.0
        else:
            q_to_heating = q_thermal_generated
            target_supply_temp = 35.0
        
        current_cop = self._calculate_cop(t_ext, target_supply_temp)
        p_electrical_w = q_thermal_generated / current_cop

        # Tank
        q_loss_tank = self.phys["UA_TANK_W_K"] * (t_tank - 20.0)
        q_user_demand = self._get_dhw_demand(self.current_step)
        delta_t_tank = (q_to_tank - q_loss_tank - q_user_demand) / self.phys["C_TANK_J_K"] * self.phys["TIME_STEP_SEC"]
        t_tank_new = t_tank + delta_t_tank

        # Building
        q_loss_wall = self.phys["UA_WALL_W_K"] * (t_in - t_ext)
        q_gains = 250.0 
        delta_t_in = (q_to_heating + q_gains - q_loss_wall) / self.phys["C_WALL_J_K"] * self.phys["TIME_STEP_SEC"]
        t_in_new = t_in + delta_t_in
        
        # REWARD
        temp_error = abs(t_in_new - self.cfg['t_set_heating'])
        reward_comfort = self.cfg['w_comfort'] * math.exp(-temp_error)
        reward_energy = -self.cfg['w_energy'] * u_signal 
        
        penalty_qos = -20.0 if t_tank_new < 35.0 else 0.0
        
        total_reward = reward_comfort + reward_energy + penalty_qos

        self.current_step += 1
        self.state = np.array([t_in_new, t_tank_new, t_ext], dtype=np.float32)
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            "t_in": t_in_new,
            "t_tank": t_tank_new,
            "t_ext": t_ext,
            "power_kw": p_electrical_w / 1000.0, # ELECTRICAL kW
            "cop": current_cop,
            "mode": "DHW" if q_to_tank > 0 else "Heating"
        }

        return self.state, total_reward, terminated, truncated, info

class StreamlitProgressCallback(BaseCallback):
    """Custom callback to update Streamlit progress bar during training."""
    def __init__(self, total_timesteps: int, progress_bar: Any):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.progress_bar = progress_bar

    def _on_step(self) -> bool:
        progress = np.clip(self.num_timesteps / self.total_timesteps, 0.0, 1.0)
        self.progress_bar.progress(progress)
        return True

def run_simulation_episode(model: PPO, env: HeatPumpDigitalTwin) -> pd.DataFrame:
    """Runs a full inference episode and returns time-series data."""
    obs, _ = env.reset()
    done = False
    
    logs = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        logs.append({
            "Step": env.current_step,
            "Indoor Temp (°C)": info["t_in"],
            "Tank Temp (°C)": info["t_tank"],
            "Ambient Temp (°C)": info["t_ext"],
            "Power (kW)": info["power_kw"],
            "Mode": 1 if info["mode"] == "DHW" else 0, # 1 for DHW, 0 for Heating
            "Reward": reward
        })
    
    return pd.DataFrame(logs)


def main():
    st.set_page_config(
        page_title="ENPOWER | Smart Heat Pump Control",
        page_icon="⚡",
        layout="wide"
    )

    # --- Main Dashboard ---
    st.title("Heat Pump Control")
    st.markdown("### Reinforcement Learning control (PPO)")
    st.markdown("PPO controller on simplified thermal model")

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        st.subheader("Thermostat Settings")
        t_set_heating = st.slider("Target Indoor Temp (°C)", 18.0, 24.0, 20.0, 0.5)
        t_set_dhw = st.slider("Target DHW Temp (°C)", 40.0, 60.0, 50.0, 1.0)
        
        st.divider()
        
        st.subheader("RL Optimization Weights")
        st.info("Reward function weights")
        w_comfort = st.slider("Comfort Weight (c1)", 1.0, 50.0, 20.0)
        w_energy = st.slider("Energy Cost Weight (c2)", 0.0, 5.0, 0.5)
        
        st.divider()
        
        st.subheader("Training Parameters")
        train_steps = st.select_slider("Training Horizon (Steps)", options=[2000, 5000, 10000, 25000], value=5000)
        
        run_btn = st.button("Initialize & Train Agent", type="primary")


    if run_btn:
        # Configuration Dictionary
        user_config = {
            't_set_heating': t_set_heating,
            't_set_dhw': t_set_dhw,
            'w_comfort': w_comfort,
            'w_energy': w_energy
        }

        env = HeatPumpDigitalTwin(user_config)
        
        with st.spinner(f"Training PPO Agent ({train_steps} steps)..."):
            progress_bar = st.progress(0.0)
            
            model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4)
            
            # Execute Training
            callback = StreamlitProgressCallback(train_steps, progress_bar)
            model.learn(total_timesteps=train_steps, callback=callback)
            
            progress_bar.empty()
        
        st.success("Training Complete")

        df_results = run_simulation_episode(model, env)
        
        df_results["Time (h)"] = df_results["Step"] * 15 / 60.0

        # --- Metrics Row ---
        avg_temp = df_results["Indoor Temp (°C)"].mean()
        avg_dev = abs(avg_temp - t_set_heating)
        total_kwh = df_results["Power (kW)"].sum() * 0.25 # 15 min steps
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Indoor Temp", f"{avg_temp:.2f} °C", delta=f"{avg_temp - t_set_heating:.2f}")
        col2.metric("Mean Deviation", f"{avg_dev:.2f} °C", help="Lower is better (Comfort)")
        col3.metric("Total Energy", f"{total_kwh:.2f} kWh", help="Total consumption over 3 days")

        # --- Plots ---
        
        # Plot 1: Thermal Comfort
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=df_results["Time (h)"], y=df_results["Indoor Temp (°C)"], 
                                      mode='lines', name='Indoor Temp', line=dict(color='#FF4B4B')))
        fig_temp.add_trace(go.Scatter(x=df_results["Time (h)"], y=[t_set_heating]*len(df_results), 
                                      mode='lines', name='Setpoint', line=dict(color='green', dash='dash')))
        fig_temp.add_trace(go.Scatter(x=df_results["Time (h)"], y=df_results["Ambient Temp (°C)"], 
                                      mode='lines', name='Ambient', line=dict(color='blue', width=1, dash='dot')))
        fig_temp.update_layout(title="Building Thermal Dynamics", xaxis_title="Time (hours)", yaxis_title="Temperature (°C)", height=350)
        st.plotly_chart(fig_temp, use_container_width=True)

        # Plot 2: DHW Tank Dynamics
        fig_dhw = go.Figure()
        fig_dhw.add_trace(go.Scatter(x=df_results["Time (h)"], y=df_results["Tank Temp (°C)"], 
                                     mode='lines', name='Tank Temp', line=dict(color='#1f77b4')))
        fig_dhw.add_trace(go.Scatter(x=df_results["Time (h)"], y=[t_set_dhw]*len(df_results), 
                                     mode='lines', name='DHW Setpoint', line=dict(color='green', dash='dash')))
        fig_dhw.add_hline(y=40, line_dash="dot", line_color="red", annotation_text="Critical Limit")
        fig_dhw.update_layout(title="DHW Storage Dynamics", xaxis_title="Time (hours)", yaxis_title="Temperature (°C)", height=300)
        st.plotly_chart(fig_dhw, use_container_width=True)

        # Plot 3: Agent Action (Power Modulation)
        fig_power = go.Figure()
        fig_power.add_trace(go.Scatter(x=df_results["Time (h)"], y=df_results["Power (kW)"], 
                                       mode='lines', name='Power Output', fill='tozeroy', line=dict(color='#FFA15A')))
        fig_power.update_layout(title="Agent Control Policy (Inverter Modulation)", xaxis_title="Time (hours)", yaxis_title="Power (kW)", height=300)
        st.plotly_chart(fig_power, use_container_width=True)

    else:
        # Landing State
        st.info("👈 Please configure the simulation parameters in the sidebar and click **'Initialize & Train Agent'**.")
        
        with st.expander("Model notes"):
            st.markdown("""
                - Building and tank are simulated with simple energy-balance updates at 15-min steps.
                - The agent observes indoor, tank and ambient temperature and outputs a continuous command in [0, 1].
                - Reward combines comfort tracking, a control-effort penalty, and a low-tank safety penalty.
            """)

if __name__ == "__main__":
    main()
