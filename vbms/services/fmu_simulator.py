"""
FMU Simulator Service for VBMS
Handles dual FMU simulation for optimized and non-optimized building models
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import gymnasium as gym
from gymnasium import spaces

try:
    from fmpy import read_model_description, simulate_fmu
    from fmpy.simulation import Simulator
    FMPY_AVAILABLE = True
except ImportError:
    FMPY_AVAILABLE = False
    print("FMPy not available - using mock simulation")

logger = logging.getLogger(__name__)


class DualFMUSimulator:
    """Simulator for handling both optimized and non-optimized FMU models"""
    
    def __init__(self, optimized_fmu_path: str, non_optimized_fmu_path: str):
        self.optimized_fmu_path = optimized_fmu_path
        self.non_optimized_fmu_path = non_optimized_fmu_path
        
        # FMU simulators
        self.optimized_simulator = None
        self.non_optimized_simulator = None
        
        # Simulation parameters
        self.current_time = 0.0
        self.step_size = 2.0  # Match real-time: 2 seconds per step (same as main.py sleep)
        self.is_initialized = False
        
        # All FMU output variables based on the updated FMU specification
        self.fmu_variables = {
            # Temperature outputs
            "OutdoorTemp": "OutdoorTemp",
            "B1SW_Temp": "B1SW_Temp",
            "B1SE_Temp": "B1SE_Temp", 
            "B1NW_Temp": "B1NW_Temp",
            "B1NE_Temp": "B1NE_Temp",
            "B1Corr_Temp": "B1Corr_Temp",
            "B2SW_Temp": "B2SW_Temp",
            "B2SE_Temp": "B2SE_Temp",
            "B2NW_Temp": "B2NW_Temp", 
            "B2NE_Temp": "B2NE_Temp",
            "B2Corr_Temp": "B2Corr_Temp",
            
            # Heating energy outputs  
            "B1SW_Heating_Energy": "B1SW_Heating_Energy",
            "B1SE_Heating_Energy": "B1SE_Heating_Energy",
            "B1NW_Heating_Energy": "B1NW_Heating_Energy",
            "B1NE_Heating_Energy": "B1NE_Heating_Energy",
            "B1Corr_Heating_Energy": "B1Corr_Heating_Energy",
            "B2SW_Heating_Energy": "B2SW_Heating_Energy",
            "B2SE_Heating_Energy": "B2SE_Heating_Energy", 
            "B2NW_Heating_Energy": "B2NW_Heating_Energy",
            "B2NE_Heating_Energy": "B2NE_Heating_Energy",
            "B2Corr_Heating_Energy": "B2Corr_Heating_Energy",
            
            # Cooling energy outputs
            "B1SW_Cooling_Energy": "B1SW_Cooling_Energy",
            "B1SE_Cooling_Energy": "B1SE_Cooling_Energy",
            "B1NW_Cooling_Energy": "B1NW_Cooling_Energy", 
            "B1NE_Cooling_Energy": "B1NE_Cooling_Energy",
            "B1Corr_Cooling_Energy": "B1Corr_Cooling_Energy",
            "B2SW_Cooling_Energy": "B2SW_Cooling_Energy",
            "B2SE_Cooling_Energy": "B2SE_Cooling_Energy",
            "B2NW_Cooling_Energy": "B2NW_Cooling_Energy",
            "B2NE_Cooling_Energy": "B2NE_Cooling_Energy", 
            "B2Corr_Cooling_Energy": "B2Corr_Cooling_Energy",
            
            # Power and energy outputs
            "Total_Electricity_Power": "Total_Electricity_Power",
            "Total_Lights_Energy": "Total_Lights_Energy",
            "Total_Equipment_Energy": "Total_Equipment_Energy",
            "Total_Fan_Energy": "Total_Fan_Energy",
            
            # Environmental outputs
            "Global_Horizontal_Irradiance": "Global_Horizontal_Irradiance",
            "Direct_Solar_Radiation": "Direct_Solar_Radiation", 
            "Diffuse_Solar_Radiation": "Diffuse_Solar_Radiation",
            
            # Comfort outputs
            "PMV": "PMV",
            "PPD": "PPD"
        }
        
        # Zone mapping for easier access
        self.zone_names = [
            "B1SW", "B1SE", "B1NW", "B1NE", "B1Corr",
            "B2SW", "B2SE", "B2NW", "B2NE", "B2Corr"
        ]
        
        # Initialize thermal state for both models (proper thermal inertia)
        self.thermal_state = {
            "optimized": {zone: 22.0 for zone in self.zone_names},
            "non_optimized": {zone: 22.0 for zone in self.zone_names}
        }
        
        # Initialize energy state for both models
        self.energy_state = {
            "optimized": {
                "heating_energy": {zone: 0.0 for zone in self.zone_names},
                "cooling_energy": {zone: 0.0 for zone in self.zone_names},
                "total_electricity_power": 0.0,
                "total_lights_energy": 0.0,
                "total_equipment_energy": 0.0, 
                "total_fan_energy": 0.0
            },
            "non_optimized": {
                "heating_energy": {zone: 0.0 for zone in self.zone_names},
                "cooling_energy": {zone: 0.0 for zone in self.zone_names},
                "total_electricity_power": 0.0,
                "total_lights_energy": 0.0,
                "total_equipment_energy": 0.0,
                "total_fan_energy": 0.0
            }
        }
        
        # Initialize simulators
        self._initialize_simulators()
        
    def initialize(self, start_time: float = 0.0, end_time: float = 86400.0):
        """Initialize the dual FMU simulator"""
        self.current_time = start_time
        logger.info(f"Dual FMU simulator initialized from {start_time} to {end_time}")
        
    def _initialize_simulators(self):
        """Initialize both FMU simulators"""
        try:
            if FMPY_AVAILABLE:
                logger.info("Initializing FMU simulators")
                # For now, we'll use mock simulation until we can test with real FMUs
                logger.info("Using mock simulation for testing")
            else:
                logger.warning("FMPy not available - using mock simulation")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize simulators: {e}")
            self.is_initialized = False
    
    def step(self, actions: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Execute one simulation step for both models
        
        Args:
            actions: Dictionary of control actions (zone setpoints)
            
        Returns:
            Dictionary containing results from both optimized and non-optimized models
        """
        if not self.is_initialized:
            logger.warning("Simulator not initialized - using mock data")
            return self._generate_mock_data()
        
        try:
            # For now, return mock data with realistic building thermal behavior
            results = {
                "optimized": self._simulate_step("optimized", actions),
                "non_optimized": self._simulate_step("non_optimized", actions)
            }
            
            self.current_time += self.step_size
            return results
            
        except Exception as e:
            logger.error(f"Simulation step failed: {e}")
            return self._generate_mock_data()
    
    def _simulate_step(self, model_type: str, actions: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Simulate one step for a specific model with proper thermal dynamics and all outputs"""
        
        # Use real-time based calculation for realistic temperature progression
        # Create a slower, more realistic daily temperature cycle
        time_hours = (self.current_time / 3600.0) % 24  # 24-hour cycle
        
        # Base temperature with realistic daily variation (±4°C around 18°C base)
        # Peak temperature at 14:00 (2 PM), minimum at 6:00 (6 AM)
        daily_cycle = np.sin((time_hours - 6) / 24.0 * 2 * np.pi)
        base_outdoor_temp = 18.0  # Comfortable spring temperature
        outdoor_temp = base_outdoor_temp + 4.0 * daily_cycle  # 14°C to 22°C range
        
        # Add very small seasonal variation (simulate different days)
        day_of_year = (self.current_time / 86400.0) % 365  # Day within year
        seasonal_variation = 2.0 * np.sin((day_of_year - 80) / 365.0 * 2 * np.pi)  # ±2°C seasonal
        outdoor_temp += seasonal_variation
        
        # Building efficiency characteristics
        if model_type == "optimized":
            thermal_resistance = 1.2  # Better insulation
            hvac_efficiency = 0.95    # More efficient HVAC
            infiltration_rate = 0.3   # Lower air leakage
            energy_efficiency = 0.8   # Better energy efficiency
        else:
            thermal_resistance = 0.8  # Standard insulation  
            hvac_efficiency = 0.85    # Standard HVAC
            infiltration_rate = 0.5   # Higher air leakage
            energy_efficiency = 1.0   # Standard energy efficiency
        
        results = {}
        
        # Set consistent random seed for reproducible noise patterns
        np.random.seed(int(self.current_time / 10) + (1 if model_type == "optimized" else 2))
        
        # Outdoor temperature
        results["OutdoorTemp"] = round(outdoor_temp, 2)
        
        # Calculate solar radiation based on time of day
        solar_elevation = max(0, np.sin((time_hours - 6) / 12.0 * np.pi))  # Sun elevation
        global_irradiance = 800 * solar_elevation  # Max 800 W/m2
        direct_radiation = global_irradiance * 0.7 if solar_elevation > 0.1 else 0
        diffuse_radiation = global_irradiance * 0.3
        
        results["Global_Horizontal_Irradiance"] = round(global_irradiance, 2)
        results["Direct_Solar_Radiation"] = round(direct_radiation, 2)
        results["Diffuse_Solar_Radiation"] = round(diffuse_radiation, 2)
        
        # Calculate total energy consumption for this timestep
        total_heating_energy = 0.0
        total_cooling_energy = 0.0
        
        # Simulate zone temperatures and energy consumption
        for zone_id in self.zone_names:
            # Get current zone temperature from thermal state
            current_temp = self.thermal_state[model_type][zone_id]
            
            # Determine target temperature (setpoint or free-floating)
            if actions and zone_id in actions:
                target_temp = actions[zone_id]  # HVAC setpoint
                hvac_active = True
            else:
                # Free-floating: influenced by outdoor temperature
                target_temp = 20.0 + (outdoor_temp - base_outdoor_temp) * 0.3  # Passive thermal gains
                hvac_active = False
            
            # Calculate thermal dynamics with proper time constants
            # Buildings have thermal inertia - temperature changes very slowly
            
            # Heat transfer from outdoor (conduction + infiltration) - very slow
            outdoor_influence = (outdoor_temp - current_temp) * (1.0 / thermal_resistance + infiltration_rate) * 0.01
            
            # HVAC response (if active) - moderate speed
            if hvac_active:
                hvac_influence = (target_temp - current_temp) * hvac_efficiency * 0.05
            else:
                hvac_influence = 0.0
            
            # Internal heat gains (occupancy, equipment, lighting) - small daily variation
            internal_gains = 0.5 + 0.3 * np.sin((time_hours - 10) / 8.0 * np.pi)  # Peak during work hours
            if time_hours < 7 or time_hours > 19:  # Outside work hours
                internal_gains *= 0.3
            
            # Calculate temperature change with thermal inertia
            # Real buildings change temperature very slowly (thermal mass effect)
            temp_change = (outdoor_influence + hvac_influence + internal_gains * 0.1) * 0.01  # Very slow changes
            
            # Limit maximum temperature change per step (realistic building thermal response)
            max_temp_change = 0.1  # Maximum 0.1°C change per 2-second step
            temp_change = np.clip(temp_change, -max_temp_change, max_temp_change)
            
            # Update thermal state with realistic constraints
            new_temp = current_temp + temp_change
            
            # Minimal measurement noise (realistic sensor accuracy)
            measurement_noise = np.random.normal(0, 0.01)  # ±0.03°C sensor noise (reduced further)
            new_temp += measurement_noise
            
            # Reasonable temperature bounds
            new_temp = max(10.0, min(35.0, new_temp))
            
            # Update thermal state for next iteration
            self.thermal_state[model_type][zone_id] = new_temp
            results[f"{zone_id}_Temp"] = round(new_temp, 2)
            
            # Calculate energy consumption based on HVAC operation
            zone_heating_energy = 0.0
            zone_cooling_energy = 0.0
            
            if hvac_active:
                temp_diff = abs(current_temp - target_temp)
                base_energy = temp_diff * 1000.0 * self.step_size  # Base energy in Joules
                
                if current_temp < target_temp:  # Heating needed
                    zone_heating_energy = base_energy * energy_efficiency
                elif current_temp > target_temp:  # Cooling needed  
                    zone_cooling_energy = base_energy * energy_efficiency
            
            # Add solar gains impact on cooling
            if solar_elevation > 0.1:
                solar_cooling_load = global_irradiance * 0.1 * self.step_size
                zone_cooling_energy += solar_cooling_load * energy_efficiency
            
            # Update energy state (cumulative)
            self.energy_state[model_type]["heating_energy"][zone_id] += zone_heating_energy
            self.energy_state[model_type]["cooling_energy"][zone_id] += zone_cooling_energy
            
            results[f"{zone_id}_Heating_Energy"] = round(self.energy_state[model_type]["heating_energy"][zone_id], 2)
            results[f"{zone_id}_Cooling_Energy"] = round(self.energy_state[model_type]["cooling_energy"][zone_id], 2)
            
            total_heating_energy += zone_heating_energy
            total_cooling_energy += zone_cooling_energy
        
        # Calculate total building energy consumption
        lights_energy = 1000.0 * self.step_size  # Base lighting energy
        if time_hours < 7 or time_hours > 19:  # Reduce lighting outside work hours
            lights_energy *= 0.3
        
        equipment_energy = 800.0 * self.step_size  # Base equipment energy
        if time_hours < 7 or time_hours > 19:  # Reduce equipment outside work hours
            equipment_energy *= 0.5
        
        fan_energy = (total_heating_energy + total_cooling_energy) * 0.1  # Fan energy proportional to HVAC
        
        # Update cumulative energy states
        self.energy_state[model_type]["total_lights_energy"] += lights_energy * energy_efficiency
        self.energy_state[model_type]["total_equipment_energy"] += equipment_energy * energy_efficiency
        self.energy_state[model_type]["total_fan_energy"] += fan_energy
        
        # Total electricity power (instantaneous)
        total_power = (total_heating_energy + total_cooling_energy + lights_energy + equipment_energy + fan_energy) / self.step_size
        self.energy_state[model_type]["total_electricity_power"] = total_power * energy_efficiency
        
        results["Total_Electricity_Power"] = round(self.energy_state[model_type]["total_electricity_power"], 2)
        results["Total_Lights_Energy"] = round(self.energy_state[model_type]["total_lights_energy"], 2)
        results["Total_Equipment_Energy"] = round(self.energy_state[model_type]["total_equipment_energy"], 2)
        results["Total_Fan_Energy"] = round(self.energy_state[model_type]["total_fan_energy"], 2)
        
        # Calculate comfort metrics (PMV and PPD)
        # Simplified PMV calculation based on temperature deviation from comfort range
        avg_temp = np.mean([results[f"{zone}_Temp"] for zone in self.zone_names])
        comfort_temp = 22.0  # Ideal comfort temperature
        temp_deviation = avg_temp - comfort_temp
        
        # PMV scale: -3 (cold) to +3 (hot), 0 is neutral
        pmv = np.clip(temp_deviation * 0.5, -3.0, 3.0)
        
        # PPD calculation based on PMV (Fanger's equation simplified)
        ppd = 100 - 95 * np.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
        ppd = max(5, min(100, ppd))  # PPD minimum is 5%
        
        results["PMV"] = round(pmv, 3)
        results["PPD"] = round(ppd, 1)
        
        return results
    
    def _generate_mock_data(self) -> Dict[str, Dict[str, float]]:
        """Generate mock simulation data for testing"""
        return {
            "optimized": self._simulate_step("optimized", None),
            "non_optimized": self._simulate_step("non_optimized", None)
        }
    
    def reset(self):
        """Reset simulation to initial state"""
        self.current_time = 0.0
        # Reset thermal state to comfortable room temperature
        self.thermal_state = {
            "optimized": {zone: 22.0 for zone in self.zone_names},
            "non_optimized": {zone: 22.0 for zone in self.zone_names}
        }
        # Reset energy state to zero
        self.energy_state = {
            "optimized": {
                "heating_energy": {zone: 0.0 for zone in self.zone_names},
                "cooling_energy": {zone: 0.0 for zone in self.zone_names},
                "total_electricity_power": 0.0,
                "total_lights_energy": 0.0,
                "total_equipment_energy": 0.0, 
                "total_fan_energy": 0.0
            },
            "non_optimized": {
                "heating_energy": {zone: 0.0 for zone in self.zone_names},
                "cooling_energy": {zone: 0.0 for zone in self.zone_names},
                "total_electricity_power": 0.0,
                "total_lights_energy": 0.0,
                "total_equipment_energy": 0.0,
                "total_fan_energy": 0.0
            }
        }
        logger.info("Simulation reset to initial state")
    
    def get_zone_temperatures(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Extract zone temperatures from simulation results"""
        zone_temps = {"optimized": {}, "non_optimized": {}}
        
        for model_type in ["optimized", "non_optimized"]:
            if model_type in results:
                for zone_id in self.zone_names:
                    temp_var = f"{zone_id}_Temp"
                    if temp_var in results[model_type]:
                        zone_temps[model_type][zone_id] = results[model_type][temp_var]
                
                # Add outdoor temperature
                if "OutdoorTemp" in results[model_type]:
                    zone_temps[model_type]["outdoor"] = results[model_type]["OutdoorTemp"]
        
        return zone_temps
    
    def get_zone_names(self) -> List[str]:
        """Get list of available zone names"""
        return self.zone_names


class EnergyPlusGymEnvironment(gym.Env):
    """Gymnasium environment for EnergyPlus FMU simulation"""
    
    def __init__(self, dual_simulator: DualFMUSimulator):
        super().__init__()
        
        self.dual_simulator = dual_simulator
        
        # Action space: temperature setpoints for 10 zones (18-26°C)
        self.action_space = spaces.Box(
            low=18.0, high=26.0, shape=(10,), dtype=np.float32
        )
        
        # Observation space: temperatures for 11 zones (10 indoor + 1 outdoor)
        self.observation_space = spaces.Box(
            low=0.0, high=50.0, shape=(11,), dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 24 * 7  # One week simulation
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.dual_simulator.reset()
        self.current_step = 0
        
        # Get initial observation
        results = self.dual_simulator.step()
        observation = self._extract_observation(results["optimized"])
        
        return observation, {}
    
    def step(self, action):
        """Execute one environment step"""
        # Convert action array to zone setpoint dictionary
        actions = {}
        for i, zone_id in enumerate(self.zone_names):
            actions[zone_id] = float(action[i])
        
        # Execute simulation step
        results = self.dual_simulator.step(actions)
        
        # Extract observation from optimized model
        observation = self._extract_observation(results["optimized"])
        
        # Calculate reward based on comfort and energy efficiency
        reward = self._calculate_reward(observation, action)
        
        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return observation, reward, done, False, {}
    
    def _extract_observation(self, results: Dict[str, float]) -> np.ndarray:
        """Extract observation array from simulation results"""
        temps = []
        
        # Add zone temperatures
        for zone_id in self.zone_names:
            temp_var = f"{zone_id}_Temp"
            temp = results.get(temp_var, 22.0)  # Default to 22°C if missing
            temps.append(temp)
        
        # Add outdoor temperature
        outdoor_temp = results.get("OutdoorTemp", 20.0)
        temps.append(outdoor_temp)
        
        return np.array(temps, dtype=np.float32)
    
    def _calculate_reward(self, observation: np.ndarray, action: np.ndarray) -> float:
        """Calculate reward based on comfort and energy efficiency"""
        zone_temps = observation[:10]  # First 10 are zone temperatures
        
        # Comfort reward: penalty for deviation from comfort range (20-24°C)
        comfort_target = 22.0
        comfort_penalty = np.mean(np.maximum(0, np.abs(zone_temps - comfort_target) - 2.0))
        comfort_reward = -comfort_penalty * 10.0
        
        # Stability reward: penalty for large temperature variations
        temp_std = np.std(zone_temps)
        stability_reward = -temp_std * 5.0
        
        # Energy reward: penalty for extreme setpoints
        energy_penalty = np.mean(np.maximum(0, action - 24.0) + np.maximum(0, 18.0 - action))
        energy_reward = -energy_penalty * 2.0
        
        total_reward = comfort_reward + stability_reward + energy_reward
        
        return float(total_reward)


# Maintain backwards compatibility
FMUSimulator = DualFMUSimulator
