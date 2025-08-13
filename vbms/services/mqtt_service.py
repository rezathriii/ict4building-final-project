#!/usr/bin/env python3
"""
MQTT Service for VBMS Platform
Enhanced with structured topic hierarchy and data publishing
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

class VBMSTopics:
    """VBMS MQTT Topic Structure"""
    
    # Sensor data topics
    TEMPERATURE = "vbms/sensors/temperature"
    HUMIDITY = "vbms/sensors/humidity" 
    OCCUPANCY = "vbms/sensors/occupancy"
    ENERGY = "vbms/sensors/energy"
    
    # Zone-specific topics
    ZONE_TEMP = "vbms/zones/{zone_id}/temperature"
    ZONE_SETPOINT = "vbms/zones/{zone_id}/setpoint"
    ZONE_HVAC = "vbms/zones/{zone_id}/hvac"
    ZONE_COMFORT = "vbms/zones/{zone_id}/comfort"
    
    # Control topics
    CONTROL_COMMANDS = "vbms/control/commands"
    CONTROL_STATUS = "vbms/control/status"
    
    # Simulation topics
    SIM_STATE = "vbms/simulation/state"
    SIM_METRICS = "vbms/simulation/metrics"
    SIM_REWARDS = "vbms/simulation/rewards"
    
    # System topics
    SYSTEM_STATUS = "vbms/system/status"
    SYSTEM_ALERTS = "vbms/system/alerts"
    SYSTEM_LOGS = "vbms/system/logs"

class MQTTService:
    def __init__(self, broker_host: str = "mqtt", broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = mqtt.Client()
        self.connected = False
        self.message_callbacks: Dict[str, Callable] = {}
        
        # Setup callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
    async def connect(self):
        """Connect to MQTT broker"""
        try:
            logger.info(f"Connecting to MQTT broker: {self.broker_host}:{self.broker_port}")
            
            # Connect in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                self.client.connect, 
                self.broker_host, 
                self.broker_port
            )
            
            # Start network loop
            self.client.loop_start()
            
            # Wait for connection
            await asyncio.sleep(1)
            
            if self.connected:
                logger.info("MQTT connection established")
            else:
                raise ConnectionError("Failed to connect to MQTT broker")
                
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            raise
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for successful connection"""
        if rc == 0:
            self.connected = True
            logger.info("MQTT client connected successfully")
            
            # Subscribe to control topics
            client.subscribe("vbms/control/+")
            client.subscribe("vbms/zones/+/setpoint")
            
        else:
            logger.error(f"MQTT connection failed with code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for disconnection"""
        self.connected = False
        logger.warning("MQTT client disconnected")
    
    def _on_message(self, client, userdata, msg):
        """Callback for received messages"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            logger.info(f"MQTT message received - Topic: {topic}")
            
            # Route message to appropriate callback
            for pattern, callback in self.message_callbacks.items():
                if pattern in topic:
                    callback(topic, payload)
                    
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    async def publish_zone_data(self, zone_id: str, temperature: float, 
                               setpoint: float, comfort_metrics: Dict[str, float]):
        """Publish zone-specific thermal data"""
        timestamp = datetime.now().isoformat()
        
        # Zone temperature
        temp_data = {
            "zone_id": zone_id,
            "temperature": temperature,
            "timestamp": timestamp,
            "unit": "celsius"
        }
        await self.publish(VBMSTopics.ZONE_TEMP.format(zone_id=zone_id), temp_data)
        
        # Zone setpoint
        setpoint_data = {
            "zone_id": zone_id,
            "setpoint": setpoint,
            "timestamp": timestamp,
            "unit": "celsius"
        }
        await self.publish(VBMSTopics.ZONE_SETPOINT.format(zone_id=zone_id), setpoint_data)
        
        # Comfort metrics
        comfort_data = {
            "zone_id": zone_id,
            "pmv": comfort_metrics.get("pmv", 0),
            "ppd": comfort_metrics.get("ppd", 0),
            "timestamp": timestamp
        }
        await self.publish(VBMSTopics.ZONE_COMFORT.format(zone_id=zone_id), comfort_data)
    
    async def publish_simulation_state(self, state_data: Dict[str, Any]):
        """Publish simulation state data"""
        timestamp = datetime.now().isoformat()
        
        payload = {
            "timestamp": timestamp,
            "outdoor_temperature": state_data.get("outdoor_temp"),
            "time_of_day": state_data.get("time_of_day"),
            "simulation_step": state_data.get("step", 0),
            "zone_count": len(state_data.get("zone_temperatures", {})),
            "zone_temperatures": state_data.get("zone_temperatures", {}),
            "energy_consumption": state_data.get("energy_consumption", 0)
        }
        
        await self.publish(VBMSTopics.SIM_STATE, payload)
    
    async def publish_rl_metrics(self, episode: int, reward: float, 
                                comfort_violations: int, energy_cost: float):
        """Publish RL training metrics"""
        timestamp = datetime.now().isoformat()
        
        payload = {
            "timestamp": timestamp,
            "episode": episode,
            "reward": reward,
            "comfort_violations": comfort_violations,
            "energy_cost": energy_cost,
            "avg_reward": reward  # Can be enhanced with running average
        }
        
        await self.publish(VBMSTopics.SIM_REWARDS, payload)
    
    async def publish_simulation_status(self, status: str, details: Dict[str, Any] = None):
        """Publish simulation status updates"""
        timestamp = datetime.now().isoformat()
        
        payload = {
            "timestamp": timestamp,
            "status": status,  # "running", "stopped", "paused", "error"
            "details": details or {}
        }
        
        await self.publish(VBMSTopics.SIM_STATE, payload)
    
    async def publish_system_status(self, component: str, status: str, details: Dict[str, Any] = None):
        """Publish system component status"""
        timestamp = datetime.now().isoformat()
        
        payload = {
            "timestamp": timestamp,
            "component": component,
            "status": status,  # "online", "offline", "warning", "error"
            "details": details or {}
        }
        
        await self.publish(VBMSTopics.SYSTEM_STATUS, payload)
    
    async def publish(self, topic: str, payload: Dict[str, Any]):
        """Publish message to MQTT topic"""
        if not self.connected:
            logger.error("MQTT client not connected")
            return False
        
        try:
            message = json.dumps(payload)
            result = self.client.publish(topic, message, qos=1)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Published to {topic}: {payload}")
                return True
            else:
                logger.error(f"Failed to publish to {topic}")
                return False
                
        except Exception as e:
            logger.error(f"Error publishing MQTT message: {e}")
            return False
    
    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to topic with callback"""
        self.message_callbacks[topic] = callback
        
        if self.connected:
            self.client.subscribe(topic)
            logger.info(f"Subscribed to topic: {topic}")
    
    def is_connected(self) -> bool:
        """Check if MQTT client is connected"""
        return self.connected
    
    async def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            await asyncio.sleep(1)
            logger.info("MQTT client disconnected")
