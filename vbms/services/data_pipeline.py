#!/usr/bin/env python3
"""
Data Pipeline Service for VBMS Platform
Integrates dual FMU simulation data with MQTT and InfluxDB
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .mqtt_service import MQTTService
from .influxdb_service import VBMSInfluxDB
from .fmu_simulator import DualFMUSimulator, EnergyPlusGymEnvironment

logger = logging.getLogger(__name__)

class VBMSDataPipeline:
    """Enhanced data pipeline for VBMS thermal management system with dual FMU simulation"""
    
    def __init__(self):
        self.mqtt_service = MQTTService()
        self.influxdb_service = VBMSInfluxDB()
        self.dual_fmu_simulator = None
        self.gym_environment = None
        
        self.running = False
        self.simulation_step = 0
        self.episode = 0
        
        # Data collection configuration
        self.collection_interval = 5.0  # seconds
        self.storage_batch_size = 10
        self.data_buffer = []
        
    async def initialize(self):
        """Initialize all services"""
        try:
            logger.info("Initializing VBMS Data Pipeline...")
            
            # Initialize MQTT service
            await self.mqtt_service.connect()
            
            # Initialize InfluxDB service
            await self.influxdb_service.initialize()
            
            # Initialize dual FMU simulation components
            optimized_fmu_path = "/app/fmu/optimized_office_building.fmu"
            non_optimized_fmu_path = "/app/fmu/not_optimized_office_building.fmu"
            
            self.dual_fmu_simulator = DualFMUSimulator(
                optimized_fmu_path=optimized_fmu_path,
                non_optimized_fmu_path=non_optimized_fmu_path
            )
            self.dual_fmu_simulator.initialize()
            
            # Initialize EnergyPlus Gym environment
            self.gym_environment = EnergyPlusGymEnvironment(self.dual_fmu_simulator)
            
            # Setup MQTT subscriptions
            self._setup_mqtt_subscriptions()
            
            # Publish initial system status
            await self.mqtt_service.publish_system_status(
                "data_pipeline", 
                "online",
                {"initialized": True, "timestamp": datetime.now().isoformat()}
            )
            
            logger.info("Data Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize data pipeline: {e}")
            raise
    
    def _setup_mqtt_subscriptions(self):
        """Setup MQTT topic subscriptions"""
        # Subscribe to control commands
        self.mqtt_service.subscribe(
            "vbms/control/commands",
            self._handle_control_command
        )
        
        # Subscribe to setpoint changes
        self.mqtt_service.subscribe(
            "vbms/zones/+/setpoint",
            self._handle_setpoint_change
        )
        
        # Subscribe to system commands
        self.mqtt_service.subscribe(
            "vbms/system/commands",
            self._handle_system_command
        )
    
    async def _handle_control_command(self, topic: str, payload: Dict[str, Any]):
        """Handle control commands from MQTT"""
        try:
            command = payload.get("command")
            
            if command == "start_simulation":
                await self.start_simulation()
            elif command == "stop_simulation":
                await self.stop_simulation()
            elif command == "reset_simulation":
                await self.reset_simulation()
                
        except Exception as e:
            logger.error(f"Error handling control command: {e}")
    
    async def _handle_setpoint_change(self, topic: str, payload: Dict[str, Any]):
        """Handle zone setpoint changes from MQTT"""
        try:
            # Extract zone_id from topic
            zone_id = topic.split("/")[-2]
            setpoint = payload.get("setpoint")
            
            logger.info(f"Manual setpoint change for {zone_id}: {setpoint}°C")
                
        except Exception as e:
            logger.error(f"Error handling setpoint change: {e}")
    
    async def _handle_system_command(self, topic: str, payload: Dict[str, Any]):
        """Handle system commands from MQTT"""
        try:
            command = payload.get("command")
            
            if command == "status":
                await self.publish_system_status()
            elif command == "diagnostics":
                await self.publish_diagnostics()
                
        except Exception as e:
            logger.error(f"Error handling system command: {e}")
    
    async def start_simulation(self):
        """Start thermal simulation"""
        try:
            if not self.dual_fmu_simulator:
                raise ValueError("Dual FMU simulator not initialized")
                
            self.running = True
            self.simulation_step = 0
            
            # Reset simulation
            self.dual_fmu_simulator.reset()
            
            # Publish status
            await self.mqtt_service.publish_simulation_status(
                "running",
                {"step": self.simulation_step, "timestamp": datetime.now().isoformat()}
            )
            
            logger.info("Simulation started")
            
        except Exception as e:
            logger.error(f"Error starting simulation: {e}")
            raise
    
    async def stop_simulation(self):
        """Stop thermal simulation"""
        self.running = False
        
        await self.mqtt_service.publish_simulation_status(
            "stopped",
            {"step": self.simulation_step, "timestamp": datetime.now().isoformat()}
        )
        
        logger.info("Simulation stopped")
    
    async def reset_simulation(self):
        """Reset thermal simulation"""
        try:
            self.simulation_step = 0
            
            if self.dual_fmu_simulator:
                self.dual_fmu_simulator.reset()
            
            if self.gym_environment:
                self.gym_environment.reset()
            
            await self.mqtt_service.publish_simulation_status(
                "reset",
                {"step": self.simulation_step, "timestamp": datetime.now().isoformat()}
            )
            
            logger.info("Simulation reset")
            
        except Exception as e:
            logger.error(f"Error resetting simulation: {e}")
    
    async def step_simulation(self) -> Dict[str, Any]:
        """Execute a single simulation step for both models"""
        try:
            if not self.dual_fmu_simulator:
                raise ValueError("Dual FMU simulator not initialized")
            
            # Execute simulation step for both models
            results = self.dual_fmu_simulator.step()
            
            self.simulation_step += 1
            
            # Publish data to MQTT
            await self._publish_simulation_data(results)
            
            # Store data in InfluxDB
            await self._store_simulation_data(results)
            
            # Log progress
            if self.simulation_step % 10 == 0:
                logger.info(f"Simulation step {self.simulation_step} completed")
            
            return {
                "step": self.simulation_step,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in simulation step: {e}")
            raise
    
    async def _publish_simulation_data(self, results: Dict[str, Dict[str, float]]):
        """Publish comprehensive simulation data to MQTT"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Publish data for both building types
            for building_type in ["optimized", "non_optimized"]:
                if building_type not in results:
                    continue
                    
                building_data = results[building_type]
                building_id = "optimized_building" if building_type == "optimized" else "non_optimized_building"
                
                # Get zone names from simulator
                zone_names = self.dual_fmu_simulator.get_zone_names()
                
                # Publish zone temperature data
                for zone_id in zone_names:
                    temp_key = f"{zone_id}_Temp"
                    if temp_key in building_data:
                        await self.mqtt_service.publish_zone_data(
                            zone_id=f"{zone_id}_{building_type}",
                            temperature=building_data[temp_key],
                            setpoint=22.0,  # Default setpoint
                            comfort_metrics={"pmv": building_data.get("PMV", 0.0), "ppd": building_data.get("PPD", 5.0)}
                        )
                
                # Publish environmental data (only once, from optimized building)
                if building_type == "optimized":
                    env_data = {
                        "outdoor_temperature": building_data.get("OutdoorTemp", 20.0),
                        "global_horizontal_irradiance": building_data.get("Global_Horizontal_Irradiance", 0.0),
                        "direct_solar_radiation": building_data.get("Direct_Solar_Radiation", 0.0),
                        "diffuse_solar_radiation": building_data.get("Diffuse_Solar_Radiation", 0.0),
                        "timestamp": timestamp
                    }
                    await self.mqtt_service.publish("vbms/environment/weather", env_data)
                
                # Publish building energy data
                energy_data = {
                    "building_id": building_id,
                    "total_electricity_power": building_data.get("Total_Electricity_Power", 0.0),
                    "total_lights_energy": building_data.get("Total_Lights_Energy", 0.0),
                    "total_equipment_energy": building_data.get("Total_Equipment_Energy", 0.0),
                    "total_fan_energy": building_data.get("Total_Fan_Energy", 0.0),
                    "timestamp": timestamp
                }
                await self.mqtt_service.publish(f"vbms/buildings/{building_id}/energy", energy_data)
                
                # Publish zone energy data
                for zone_id in zone_names:
                    heating_key = f"{zone_id}_Heating_Energy"
                    cooling_key = f"{zone_id}_Cooling_Energy"
                    if heating_key in building_data and cooling_key in building_data:
                        zone_energy_data = {
                            "zone_id": f"{zone_id}_{building_type}",
                            "heating_energy": building_data[heating_key],
                            "cooling_energy": building_data[cooling_key],
                            "timestamp": timestamp
                        }
                        await self.mqtt_service.publish(f"vbms/zones/{zone_id}_{building_type}/energy", zone_energy_data)
                
                # Publish comfort metrics
                comfort_data = {
                    "building_id": building_id,
                    "pmv": building_data.get("PMV", 0.0),
                    "ppd": building_data.get("PPD", 5.0),
                    "timestamp": timestamp
                }
                await self.mqtt_service.publish(f"vbms/buildings/{building_id}/comfort", comfort_data)
            
            logger.info(f"Published comprehensive simulation data for step {self.simulation_step}")
            
        except Exception as e:
            logger.error(f"Error publishing simulation data: {e}")
    
    async def _store_simulation_data(self, results: Dict[str, Dict[str, float]]):
        """Store comprehensive simulation data in InfluxDB"""
        try:
            timestamp = datetime.now()
            
            # Store data for both building types
            for building_type in ["optimized", "non_optimized"]:
                if building_type not in results:
                    continue
                    
                building_data = results[building_type]
                building_id = "optimized_building" if building_type == "optimized" else "non_optimized_building"
                
                # Get zone names from simulator
                zone_names = self.dual_fmu_simulator.get_zone_names()
                
                # Store zone temperature data
                for zone_id in zone_names:
                    temp_key = f"{zone_id}_Temp"
                    if temp_key in building_data:
                        await self.influxdb_service.store_zone_thermal_data(
                            zone_id=f"{zone_id}_{building_type}",
                            temperature=building_data[temp_key],
                            setpoint=22.0,  # Default setpoint
                            timestamp=timestamp
                        )
                
                # Store zone energy data
                for zone_id in zone_names:
                    heating_key = f"{zone_id}_Heating_Energy"
                    cooling_key = f"{zone_id}_Cooling_Energy"
                    if heating_key in building_data and cooling_key in building_data:
                        await self.influxdb_service.store_zone_energy_data(
                            zone_id=f"{zone_id}_{building_type}",
                            heating_energy=building_data[heating_key],
                            cooling_energy=building_data[cooling_key],
                            timestamp=timestamp
                        )
                
                # Store building-level energy data
                total_power = building_data.get("Total_Electricity_Power", 0.0)
                lights_energy = building_data.get("Total_Lights_Energy", 0.0)
                equipment_energy = building_data.get("Total_Equipment_Energy", 0.0)
                fan_energy = building_data.get("Total_Fan_Energy", 0.0)
                
                await self.influxdb_service.store_building_energy_data(
                    building_id=building_id,
                    total_electricity_power=total_power,
                    total_lights_energy=lights_energy,
                    total_equipment_energy=equipment_energy,
                    total_fan_energy=fan_energy,
                    timestamp=timestamp
                )
                
                # Store comfort metrics
                pmv = building_data.get("PMV", 0.0)
                ppd = building_data.get("PPD", 5.0)
                
                await self.influxdb_service.store_comfort_metrics(
                    building_id=building_id,
                    pmv=pmv,
                    ppd=ppd,
                    timestamp=timestamp
                )
            
            # Store environmental data (only once, from optimized building)
            if "optimized" in results:
                building_data = results["optimized"]
                outdoor_temp = building_data.get("OutdoorTemp", 20.0)
                global_irradiance = building_data.get("Global_Horizontal_Irradiance", 0.0)
                direct_radiation = building_data.get("Direct_Solar_Radiation", 0.0)
                diffuse_radiation = building_data.get("Diffuse_Solar_Radiation", 0.0)
                
                await self.influxdb_service.store_environmental_data(
                    global_irradiance=global_irradiance,
                    direct_radiation=direct_radiation,
                    diffuse_radiation=diffuse_radiation,
                    outdoor_temp=outdoor_temp,
                    timestamp=timestamp
                )
            
            logger.info(f"Stored comprehensive simulation data for step {self.simulation_step}")
                
        except Exception as e:
            logger.error(f"Error storing simulation data: {e}")
    
    async def get_zone_temperatures(self, hours: int = 24) -> Dict[str, Any]:
        """Get zone temperature data from InfluxDB"""
        try:
            return await self.influxdb_service.query_zone_temperatures(hours)
        except Exception as e:
            logger.error(f"Error retrieving zone temperatures: {e}")
            return {"data": [], "count": 0}
    
    async def get_building_comparison(self, hours: int = 1) -> Dict[str, Any]:
        """Get comprehensive comparison data between optimized and non-optimized buildings"""
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            
            # Get energy data for both buildings
            optimized_energy = await self.influxdb_service.query_energy_consumption(
                building_id="optimized_building", 
                start_time=start_time
            )
            
            non_optimized_energy = await self.influxdb_service.query_energy_consumption(
                building_id="non_optimized_building", 
                start_time=start_time
            )
            
            # Get comfort data for both buildings
            optimized_comfort = await self.influxdb_service.query_comfort_metrics(
                building_id="optimized_building",
                start_time=start_time
            )
            
            non_optimized_comfort = await self.influxdb_service.query_comfort_metrics(
                building_id="non_optimized_building",
                start_time=start_time
            )
            
            # Get temperature data
            temperature_data = await self.influxdb_service.query_zone_temperatures(
                start_time=start_time
            )
            
            # Get environmental data
            environmental_data = await self.influxdb_service.query_environmental_data(
                start_time=start_time
            )
            
            return {
                "optimized_building": {
                    "energy": optimized_energy,
                    "comfort": optimized_comfort,
                    "temperatures": [t for t in temperature_data if not t.get("zone_id", "").endswith("_non_optimized")]
                },
                "non_optimized_building": {
                    "energy": non_optimized_energy,
                    "comfort": non_optimized_comfort,
                    "temperatures": [t for t in temperature_data if t.get("zone_id", "").endswith("_non_optimized")]
                },
                "environmental": environmental_data,
                "hours": hours,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error retrieving building comparison: {e}")
            return {
                "optimized_building": {"energy": {}, "comfort": [], "temperatures": []},
                "non_optimized_building": {"energy": {}, "comfort": [], "temperatures": []},
                "environmental": [],
                "hours": hours,
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_zone_comfort_metrics(self, building_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get comfort metrics for a specific building"""
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            return await self.influxdb_service.query_comfort_metrics(building_id, start_time)
        except Exception as e:
            logger.error(f"Error retrieving comfort metrics for {building_id}: {e}")
            return {"data": [], "count": 0}
    
    async def set_zone_setpoint(self, zone_id: str, setpoint: float):
        """Set temperature setpoint for a zone"""
        try:
            # Publish to MQTT
            await self.mqtt_service.publish_zone_setpoint(zone_id, setpoint)
            
            # Store in InfluxDB
            await self.influxdb_service.write_setpoint_data(zone_id, setpoint)
            
            logger.info(f"Setpoint for {zone_id} set to {setpoint}°C")
            
        except Exception as e:
            logger.error(f"Error setting setpoint for {zone_id}: {e}")
            raise
    
    async def publish_system_status(self):
        """Publish comprehensive system status"""
        try:
            status = {
                "simulation": {
                    "running": self.running,
                    "step": self.simulation_step,
                    "episode": self.episode
                },
                "services": {
                    "mqtt": "online" if self.mqtt_service else "offline",
                    "influxdb": "online" if self.influxdb_service else "offline",
                    "dual_fmu": "online" if self.dual_fmu_simulator else "offline"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await self.mqtt_service.publish_system_status("data_pipeline", "online", status)
            
        except Exception as e:
            logger.error(f"Error publishing system status: {e}")
    
    async def publish_diagnostics(self):
        """Publish system diagnostics"""
        try:
            diagnostics = {
                "data_buffer_size": len(self.data_buffer),
                "simulation_step": self.simulation_step,
                "episode": self.episode,
                "services_status": {
                    "mqtt_connected": self.mqtt_service.is_connected() if self.mqtt_service else False,
                    "influxdb_connected": await self.influxdb_service.health_check() if self.influxdb_service else False
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await self.mqtt_service.publish("vbms/diagnostics", diagnostics)
            
        except Exception as e:
            logger.error(f"Error publishing diagnostics: {e}")
    
    def get_zone_names(self) -> List[str]:
        """Get list of available zone names"""
        if self.dual_fmu_simulator:
            return self.dual_fmu_simulator.get_zone_names()
        return []
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.running = False
            
            if self.mqtt_service:
                await self.mqtt_service.disconnect()
            
            if self.influxdb_service:
                await self.influxdb_service.close()
                
            logger.info("Data Pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
