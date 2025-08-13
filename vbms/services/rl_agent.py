#!/usr/bin/env python3
"""
Enhanced Reinforcement Learning Agent for VBMS Platform
Complete ThermalControlEnv with realistic thermal dynamics and PPO training
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import json
from pathlib import Path

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logging.warning("Stable-baselines3 not available")
    
    # Create dummy BaseCallback for when SB3 is not available
    class BaseCallback:
        def __init__(self, verbose=0):
            self.verbose = verbose
            self.n_calls = 0
            self.model = None
        
        def _on_step(self) -> bool:
            return True

from .fmu_simulator import FMUSimulator

logger = logging.getLogger(__name__)

class ThermalControlEnv(gym.Env):
    """Enhanced Gym environment for thermal control with realistic dynamics"""
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, fmu_simulator: Optional[FMUSimulator] = None, config: Optional[Dict] = None):
        super().__init__()
        
        self.config = config or self._default_config()
        self.fmu_simulator = fmu_simulator
        
        # Environment parameters
        self.num_zones = 10
        self.comfort_temp_min = 20.0
        self.comfort_temp_max = 24.0
        self.optimal_temp = 22.0
        self.max_episode_steps = 288  # 24 hours with 5-minute steps
        self.current_step = 0
        
        # Define action space (setpoint temperatures for 10 zones)
        self.action_space = spaces.Box(
            low=18.0, high=26.0, shape=(self.num_zones,), dtype=np.float32
        )
        
        # Define observation space
        # [zone_temps(10), outdoor_temp(1), time_of_day(1), day_of_week(1), comfort_metrics(2)]
        self.observation_space = spaces.Box(
            low=-20.0, high=50.0, shape=(15,), dtype=np.float32
        )
        
        # State variables
        self.zone_temperatures = np.full(self.num_zones, 22.0, dtype=np.float32)
        self.outdoor_temperature = 20.0
        self.time_of_day = 0.0  # Hours (0-24)
        self.day_of_week = 0.0  # (0-6)
        self.comfort_metrics = {"PMV": 0.0, "PPD": 10.0}
        
        # Simulation parameters
        self.thermal_mass = np.random.uniform(0.8, 1.2, self.num_zones)  # Thermal inertia
        self.heat_transfer_coeff = np.random.uniform(0.02, 0.05, self.num_zones)  # Heat transfer
        self.internal_gains_schedule = self._create_internal_gains_schedule()
        
        # Performance tracking
        self.episode_rewards = []
        self.comfort_violations = 0
        self.energy_consumption = 0.0
        
        logger.info(f"ThermalControlEnv initialized with {self.num_zones} zones")
    
    def _default_config(self) -> Dict:
        """Default environment configuration"""
        return {
            "comfort_weight": 1.0,
            "energy_weight": 0.3,
            "comfort_penalty_factor": 2.0,
            "energy_efficiency_target": 0.8,
            "max_temp_change_per_step": 2.0,
            "occupancy_schedule_enabled": True,
            "weather_variability": True
        }
    
    def _create_internal_gains_schedule(self) -> np.ndarray:
        """Create realistic internal heat gains schedule"""
        # 24-hour schedule with higher gains during occupied hours
        schedule = np.zeros(24)
        
        # Business hours (9 AM - 6 PM): higher internal gains
        schedule[9:18] = np.random.uniform(2.0, 3.0, 9)
        
        # Early morning and evening: moderate gains
        schedule[6:9] = np.random.uniform(0.5, 1.0, 3)
        schedule[18:22] = np.random.uniform(1.0, 1.5, 4)
        
        # Night time: minimal gains
        schedule[22:24] = np.random.uniform(0.1, 0.3, 2)
        schedule[0:6] = np.random.uniform(0.1, 0.3, 6)
        
        return schedule
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset simulation time
        self.current_step = 0
        self.time_of_day = np.random.uniform(0.0, 24.0)
        self.day_of_week = np.random.randint(0, 7)
        
        # Initialize zone temperatures with realistic variation
        base_temp = np.random.uniform(20.0, 24.0)
        self.zone_temperatures = base_temp + np.random.normal(0, 0.5, self.num_zones)
        self.zone_temperatures = np.clip(self.zone_temperatures, 18.0, 26.0)
        
        # Initialize outdoor temperature
        if self.config.get("weather_variability", True):
            season_factor = np.sin(2 * np.pi * self.day_of_week / 7.0)  # Weekly variation
            daily_factor = np.sin(2 * np.pi * self.time_of_day / 24.0)  # Daily variation
            self.outdoor_temperature = 20.0 + 5.0 * season_factor + 8.0 * daily_factor
            self.outdoor_temperature += np.random.normal(0, 2.0)  # Weather noise
        else:
            self.outdoor_temperature = 20.0
        
        # Reset performance tracking
        self.comfort_violations = 0
        self.energy_consumption = 0.0
        self.comfort_metrics = {"PMV": 0.0, "PPD": 10.0}
        
        # Initialize FMU simulator if available
        if self.fmu_simulator and self.fmu_simulator.is_initialized:
            try:
                self.fmu_simulator.reset(start_time=0.0)
                logger.debug("FMU simulator reset successfully")
            except Exception as e:
                logger.warning(f"Failed to reset FMU simulator: {e}")
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation.astype(np.float32), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        self.current_step += 1
        
        # Validate and clip actions
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Use FMU simulation if available, otherwise use physics-based model
        if self.fmu_simulator and self.fmu_simulator.is_initialized:
            new_state = self._fmu_simulation_step(action)
        else:
            new_state = self._physics_simulation_step(action)
        
        # Update state from simulation results
        if new_state:
            self._update_state_from_simulation(new_state)
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        terminated = self.current_step >= self.max_episode_steps
        truncated = False
        
        # Update time
        self.time_of_day = (self.time_of_day + 0.083333) % 24.0  # 5-minute step = 0.083333 hours
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation.astype(np.float32), reward, terminated, truncated, info
    
    def _fmu_simulation_step(self, action: np.ndarray) -> Optional[Dict]:
        """Execute simulation step using FMU"""
        try:
            # For now, FMU provides environmental data
            # In future versions, we could pass setpoints to FMU
            simulation_results = self.fmu_simulator.step()
            
            if simulation_results:
                logger.debug("FMU simulation step completed")
                return simulation_results
            else:
                logger.warning("FMU simulation step failed, using physics model")
                return None
                
        except Exception as e:
            logger.error(f"FMU simulation error: {e}")
            return None
    
    def _physics_simulation_step(self, action: np.ndarray) -> Dict:
        """Execute simulation step using physics-based model"""
        
        # Get internal heat gains based on time of day
        hour = int(self.time_of_day) % 24
        internal_gains = self.internal_gains_schedule[hour]
        
        # Calculate occupancy factor
        if self.config.get("occupancy_schedule_enabled", True):
            if 9 <= hour <= 17 and self.day_of_week < 5:  # Business hours, weekdays
                occupancy_factor = 1.0
            elif 6 <= hour <= 9 or 17 <= hour <= 20:  # Transition hours
                occupancy_factor = 0.5
            else:
                occupancy_factor = 0.1
        else:
            occupancy_factor = 1.0
        
        new_temperatures = []
        
        for i, (current_temp, setpoint) in enumerate(zip(self.zone_temperatures, action)):
            # HVAC control: try to reach setpoint
            hvac_power = (setpoint - current_temp) * 0.2 * occupancy_factor
            
            # Heat transfer with outdoor environment
            outdoor_influence = (self.outdoor_temperature - current_temp) * self.heat_transfer_coeff[i]
            
            # Internal heat gains
            internal_heat = internal_gains * occupancy_factor * 0.1
            
            # Thermal mass effect (resistance to temperature change)
            thermal_inertia = self.thermal_mass[i]
            
            # Calculate temperature change
            temp_change = (hvac_power + outdoor_influence + internal_heat) / thermal_inertia
            
            # Limit maximum temperature change per step
            max_change = self.config.get("max_temp_change_per_step", 2.0)
            temp_change = np.clip(temp_change, -max_change, max_change)
            
            # Add some realistic noise
            noise = np.random.normal(0, 0.05)
            
            new_temp = current_temp + temp_change + noise
            new_temp = np.clip(new_temp, 16.0, 30.0)  # Physical limits
            
            new_temperatures.append(new_temp)
        
        # Update outdoor temperature (daily cycle with weather variability)
        if self.config.get("weather_variability", True):
            daily_cycle = 8.0 * np.sin(2 * np.pi * self.time_of_day / 24.0)
            weather_noise = np.random.normal(0, 0.5)
            self.outdoor_temperature = 20.0 + daily_cycle + weather_noise
        
        # Calculate comfort metrics
        avg_temp = np.mean(new_temperatures)
        pmv = self._calculate_pmv(avg_temp, occupancy_factor)
        ppd = self._calculate_ppd(pmv)
        
        return {
            "zone_temperatures": {f"zone_{i}": temp for i, temp in enumerate(new_temperatures)},
            "outdoor_temperature": self.outdoor_temperature,
            "comfort_metrics": {"PMV": pmv, "PPD": ppd},
            "simulation_mode": "physics"
        }
    
    def _update_state_from_simulation(self, simulation_results: Dict):
        """Update environment state from simulation results"""
        
        if "zone_temperatures" in simulation_results:
            zone_temps = simulation_results["zone_temperatures"]
            
            # Convert zone temperatures to array
            if isinstance(zone_temps, dict):
                # Handle both FMU format and physics format
                temp_values = []
                if len(zone_temps) == self.num_zones:
                    temp_values = list(zone_temps.values())
                else:
                    # If zones don't match exactly, use available ones and fill missing
                    for i in range(self.num_zones):
                        zone_key = f"zone_{i}"
                        if zone_key in zone_temps:
                            temp_values.append(zone_temps[zone_key])
                        else:
                            # Use existing temperature if zone not found
                            temp_values.append(self.zone_temperatures[i])
                
                self.zone_temperatures = np.array(temp_values, dtype=np.float32)
            
        if "outdoor_temperature" in simulation_results:
            self.outdoor_temperature = float(simulation_results["outdoor_temperature"])
            
        if "comfort_metrics" in simulation_results:
            self.comfort_metrics.update(simulation_results["comfort_metrics"])
    
    def _calculate_pmv(self, temperature: float, occupancy_factor: float) -> float:
        """Calculate Predicted Mean Vote (simplified)"""
        # Simplified PMV calculation based on temperature deviation from optimal
        temp_deviation = temperature - self.optimal_temp
        
        # Account for occupancy (more people = more heat)
        occupancy_effect = occupancy_factor * 0.2
        
        pmv = (temp_deviation + occupancy_effect) * 0.5
        return np.clip(pmv, -3.0, 3.0)
    
    def _calculate_ppd(self, pmv: float) -> float:
        """Calculate Predicted Percentage Dissatisfied from PMV"""
        # Standard PPD calculation
        ppd = 100.0 - 95.0 * np.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
        return max(5.0, min(100.0, ppd))
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward based on comfort and energy efficiency"""
        
        # Comfort reward component
        comfort_reward = 0.0
        violations = 0
        
        for temp in self.zone_temperatures:
            if self.comfort_temp_min <= temp <= self.comfort_temp_max:
                # Temperature within comfort range
                deviation = abs(temp - self.optimal_temp)
                comfort_reward += max(0, 1.0 - deviation / 2.0)  # Decreasing reward with deviation
            else:
                # Temperature outside comfort range
                if temp < self.comfort_temp_min:
                    deviation = self.comfort_temp_min - temp
                else:
                    deviation = temp - self.comfort_temp_max
                
                penalty = -deviation * self.config.get("comfort_penalty_factor", 2.0)
                comfort_reward += penalty
                violations += 1
        
        self.comfort_violations += violations
        
        # Energy efficiency reward component
        energy_penalty = 0.0
        setpoint_deviations = []
        
        for setpoint in action:
            # Penalize extreme setpoints
            if setpoint < 19.0 or setpoint > 25.0:
                deviation = min(abs(setpoint - 19.0), abs(setpoint - 25.0))
                energy_penalty += deviation * 0.5
            
            setpoint_deviations.append(abs(setpoint - self.optimal_temp))
        
        # Average setpoint deviation
        avg_setpoint_deviation = np.mean(setpoint_deviations)
        energy_penalty += avg_setpoint_deviation * 0.2
        
        self.energy_consumption += energy_penalty
        
        # PMV-based comfort penalty
        pmv_penalty = abs(self.comfort_metrics.get("PMV", 0.0)) * 0.5
        
        # PPD-based comfort penalty
        ppd = self.comfort_metrics.get("PPD", 10.0)
        ppd_penalty = max(0, (ppd - 10.0) / 90.0)  # Normalize PPD penalty
        
        # Combined reward
        total_reward = (
            self.config.get("comfort_weight", 1.0) * comfort_reward +
            -self.config.get("energy_weight", 0.3) * energy_penalty +
            -pmv_penalty +
            -ppd_penalty
        )
        
        return total_reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        observation = np.concatenate([
            self.zone_temperatures,  # 10 values
            [self.outdoor_temperature],  # 1 value
            [self.time_of_day],  # 1 value
            [self.day_of_week],  # 1 value
            [self.comfort_metrics.get("PMV", 0.0)],  # 1 value
            [self.comfort_metrics.get("PPD", 10.0)]  # 1 value
        ])
        
        return observation
    
    def _get_info(self) -> Dict:
        """Get additional information"""
        return {
            "comfort_violations": self.comfort_violations,
            "energy_consumption": self.energy_consumption,
            "average_zone_temp": np.mean(self.zone_temperatures),
            "outdoor_temp": self.outdoor_temperature,
            "time_of_day": self.time_of_day,
            "pmv": self.comfort_metrics.get("PMV", 0.0),
            "ppd": self.comfort_metrics.get("PPD", 10.0),
            "step": self.current_step
        }
    
    def render(self, mode: str = "human"):
        """Render environment (optional)"""
        if mode == "human":
            print(f"Step: {self.current_step}, Time: {self.time_of_day:.1f}h")
            print(f"Zone temps: {self.zone_temperatures}")
            print(f"Outdoor: {self.outdoor_temperature:.1f}°C")
            print(f"PMV: {self.comfort_metrics.get('PMV', 0.0):.2f}, PPD: {self.comfort_metrics.get('PPD', 10.0):.1f}%")
            print("-" * 50)


class TrainingCallback(BaseCallback):
    """Custom callback for monitoring training progress"""
    
    def __init__(self, save_freq: int = 10000, save_path: str = "./models/", verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)
        
    def _on_step(self) -> bool:
        # Save model periodically
        if self.n_calls % self.save_freq == 0:
            model_path = self.save_path / f"thermal_control_model_{self.n_calls}"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls}")
        
        return True


class ThermalControlAgent:
    """RL Agent for thermal control using PPO"""
    
    def __init__(self, fmu_simulator: Optional[FMUSimulator] = None, config: Optional[Dict] = None):
        self.fmu_simulator = fmu_simulator
        self.config = config or {}
        self.model = None
        self.env = None
        self.is_trained = False
        
        logger.info("ThermalControlAgent initialized")
    
    def create_environment(self) -> ThermalControlEnv:
        """Create training environment"""
        self.env = ThermalControlEnv(
            fmu_simulator=self.fmu_simulator,
            config=self.config.get("env_config", {})
        )
        return self.env
    
    def train(self, total_timesteps: int = 100000, save_path: str = "./models/thermal_control_final"):
        """Train the PPO agent"""
        
        if not SB3_AVAILABLE:
            logger.error("stable-baselines3 not available, cannot train agent")
            return False
        
        try:
            logger.info("Starting PPO training...")
            
            # Create environment
            if self.env is None:
                self.create_environment()
            
            # Create vectorized environment for training
            vec_env = make_vec_env(lambda: self.env, n_envs=1)
            
            # Create PPO model
            self.model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                learning_rate=self.config.get("learning_rate", 3e-4),
                n_steps=self.config.get("n_steps", 2048),
                batch_size=self.config.get("batch_size", 64),
                n_epochs=self.config.get("n_epochs", 10),
                gamma=self.config.get("gamma", 0.99),
                gae_lambda=self.config.get("gae_lambda", 0.95),
                clip_range=self.config.get("clip_range", 0.2),
                ent_coef=self.config.get("ent_coef", 0.01),
                tensorboard_log="./tensorboard_logs/"
            )
            
            # Setup callback
            callback = TrainingCallback(
                save_freq=self.config.get("save_freq", 10000),
                save_path="./models/"
            )
            
            # Train the model
            self.model.learn(total_timesteps=total_timesteps, callback=callback)
            
            # Save final model
            Path(save_path).parent.mkdir(exist_ok=True)
            self.model.save(save_path)
            
            self.is_trained = True
            logger.info(f"Training completed. Model saved to {save_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def load_model(self, model_path: str):
        """Load pre-trained model"""
        if not SB3_AVAILABLE:
            logger.error("stable-baselines3 not available, cannot load model")
            return False
        
        try:
            self.model = PPO.load(model_path)
            self.is_trained = True
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Predict action given observation"""
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained or loaded, using random action")
            # Return reasonable setpoints around 22°C
            return np.full(10, 22.0, dtype=np.float32) + np.random.normal(0, 0.5, 10)
        
        try:
            action, _ = self.model.predict(observation, deterministic=True)
            return action
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return np.full(10, 22.0, dtype=np.float32)
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate trained model"""
        if not self.is_trained or self.model is None:
            logger.error("Model not trained, cannot evaluate")
            return {}
        
        if self.env is None:
            self.create_environment()
        
        episode_rewards = []
        comfort_violations_total = 0
        energy_consumption_total = 0.0
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            comfort_violations_total += info.get("comfort_violations", 0)
            energy_consumption_total += info.get("energy_consumption", 0.0)
        
        results = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_comfort_violations": comfort_violations_total / n_episodes,
            "mean_energy_consumption": energy_consumption_total / n_episodes
        }
        
        logger.info(f"Evaluation results: {results}")
        return results
