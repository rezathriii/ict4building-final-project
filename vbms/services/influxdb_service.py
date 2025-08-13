#!/usr/bin/env python3
"""
InfluxDB Service for VBMS Platform
Enhanced with comprehensive data storage and retention policies
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.client.delete_api import DeleteApi

logger = logging.getLogger(__name__)

class VBMSInfluxDB:
    """Enhanced InfluxDB service for VBMS thermal data management"""
    
    def __init__(self, 
                 url: str = "http://influxdb:8086",
                 token: str = "vbms-token",
                 org: str = "vbms"):
        self.url = url
        self.token = token
        self.org = org
        
        # Define buckets for different data types
        self.buckets = {
            "thermal_data": "thermal_data",      # Zone temperatures, setpoints
            "comfort_data": "comfort_data",      # PMV, PPD metrics
            "energy_data": "energy_data",        # Energy consumption
            "control_data": "control_data",      # RL actions, rewards
            "system_data": "system_data"         # System status, alerts
        }
        
        self.client: Optional[InfluxDBClient] = None
        self.write_api = None
        self.query_api = None
        self.delete_api = None
        
    async def initialize(self):
        """Initialize InfluxDB connection and setup buckets"""
        try:
            self.client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org
            )
            
            # Test connection
            health = self.client.health()
            if health.status == "pass":
                self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
                self.query_api = self.client.query_api()
                self.delete_api = self.client.delete_api()
                
                # Setup buckets and retention policies
                await self._setup_buckets()
                
                logger.info("InfluxDB connection established with all buckets")
            else:
                raise ConnectionError("InfluxDB health check failed")
                
        except Exception as e:
            logger.error(f"InfluxDB initialization failed: {e}")
            raise
    
    async def _setup_buckets(self):
        """Setup buckets with retention policies"""
        buckets_api = self.client.buckets_api()
        
        # Define retention policies (in seconds)
        retention_policies = {
            "thermal_data": 86400 * 365,    # 1 year
            "comfort_data": 86400 * 365,    # 1 year  
            "energy_data": 86400 * 365,     # 1 year
            "control_data": 86400 * 90,     # 3 months
            "system_data": 86400 * 30       # 1 month
        }
        
        for bucket_name, retention in retention_policies.items():
            try:
                # Check if bucket exists
                existing_buckets = buckets_api.find_buckets()
                bucket_exists = any(b.name == bucket_name for b in existing_buckets.buckets)
                
                if not bucket_exists:
                    buckets_api.create_bucket(
                        bucket_name=bucket_name,
                        org=self.org,
                        retention_rules=[{
                            "type": "expire",
                            "everySeconds": retention
                        }]
                    )
                    logger.info(f"Created bucket: {bucket_name} with {retention}s retention")
                    
            except Exception as e:
                logger.warning(f"Could not setup bucket {bucket_name}: {e}")
    
    async def store_zone_thermal_data(self, zone_id: str, temperature: float, 
                                     setpoint: float = None, timestamp: Optional[datetime] = None):
        """Store zone thermal data with optional setpoint"""
        if not self.write_api:
            logger.error("InfluxDB write API not available")
            return False
        
        try:
            dt = timestamp or datetime.now()
            
            point = Point("zone_temperature") \
                .tag("zone_id", zone_id) \
                .field("temperature", float(temperature))
            
            if setpoint is not None:
                point = point.field("setpoint", float(setpoint))
                
            point = point.time(dt)
            
            self.write_api.write(bucket=self.buckets["thermal_data"], record=point)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store zone thermal data: {e}")
            return False
    
    async def store_zone_energy_data(self, zone_id: str, heating_energy: float, cooling_energy: float,
                                   timestamp: Optional[datetime] = None):
        """Store zone energy consumption data"""
        if not self.write_api:
            logger.error("InfluxDB write API not available")
            return False
        
        try:
            dt = timestamp or datetime.now()
            
            point = Point("zone_energy") \
                .tag("zone_id", zone_id) \
                .field("heating_energy", float(heating_energy)) \
                .field("cooling_energy", float(cooling_energy)) \
                .time(dt)
            
            self.write_api.write(bucket=self.buckets["energy_data"], record=point)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store zone energy data: {e}")
            return False
    
    async def store_building_energy_data(self, building_id: str, total_electricity_power: float,
                                       total_lights_energy: float, total_equipment_energy: float,
                                       total_fan_energy: float, timestamp: Optional[datetime] = None):
        """Store building-level energy data"""
        if not self.write_api:
            logger.error("InfluxDB write API not available")
            return False
        
        try:
            dt = timestamp or datetime.now()
            
            point = Point("building_energy") \
                .tag("building_id", building_id) \
                .field("total_electricity_power", float(total_electricity_power)) \
                .field("total_lights_energy", float(total_lights_energy)) \
                .field("total_equipment_energy", float(total_equipment_energy)) \
                .field("total_fan_energy", float(total_fan_energy)) \
                .time(dt)
            
            self.write_api.write(bucket=self.buckets["energy_data"], record=point)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store building energy data: {e}")
            return False
    
    async def store_environmental_data(self, global_irradiance: float, direct_radiation: float,
                                     diffuse_radiation: float, outdoor_temp: float,
                                     timestamp: Optional[datetime] = None):
        """Store environmental data"""
        if not self.write_api:
            logger.error("InfluxDB write API not available")
            return False
        
        try:
            dt = timestamp or datetime.now()
            
            point = Point("environmental") \
                .field("global_horizontal_irradiance", float(global_irradiance)) \
                .field("direct_solar_radiation", float(direct_radiation)) \
                .field("diffuse_solar_radiation", float(diffuse_radiation)) \
                .field("outdoor_temperature", float(outdoor_temp)) \
                .time(dt)
            
            self.write_api.write(bucket=self.buckets["thermal_data"], record=point)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store environmental data: {e}")
            return False
    
    async def store_outdoor_temperature(self, temperature: float, timestamp: Optional[datetime] = None):
        """Store outdoor temperature data"""
        if not self.write_api:
            logger.error("InfluxDB write API not available")
            return False
        
        try:
            dt = timestamp or datetime.now()
            
            points = [
                Point("outdoor_temperature")
                .tag("location", "external")
                .field("temperature", float(temperature))
                .time(dt)
            ]
            
            self.write_api.write(bucket=self.buckets["thermal_data"], record=points)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store outdoor temperature data: {e}")
            return False
    
    async def store_comfort_metrics(self, building_id: str, pmv: float, ppd: float,
                                   timestamp: Optional[datetime] = None):
        """Store comfort metrics (PMV/PPD) at building level"""
        if not self.write_api:
            return False
        
        try:
            dt = timestamp or datetime.now()
            
            point = Point("comfort_metrics") \
                .tag("building_id", building_id) \
                .field("pmv", float(pmv)) \
                .field("ppd", float(ppd)) \
                .time(dt)
            
            self.write_api.write(bucket=self.buckets["comfort_data"], record=point)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store comfort metrics: {e}")
            return False
    
    async def store_energy_data(self, total_consumption: float, zone_consumption: Dict[str, float],
                               timestamp: Optional[datetime] = None):
        """Store energy consumption data"""
        if not self.write_api:
            return False
        
        try:
            dt = timestamp or datetime.now()
            points = []
            
            # Total consumption
            points.append(
                Point("energy_consumption")
                .field("total", float(total_consumption))
                .time(dt)
            )
            
            # Zone-specific consumption
            for zone_id, consumption in zone_consumption.items():
                points.append(
                    Point("zone_energy")
                    .tag("zone_id", zone_id)
                    .field("consumption", float(consumption))
                    .time(dt)
                )
            
            self.write_api.write(bucket=self.buckets["energy_data"], record=points)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store energy data: {e}")
            return False
    
    async def store_rl_metrics(self, episode: int, step: int, reward: float, 
                              action: List[float], timestamp: Optional[datetime] = None):
        """Store RL training metrics"""
        if not self.write_api:
            return False
        
        try:
            dt = timestamp or datetime.now()
            
            # Training metrics
            point = Point("rl_training") \
                .field("episode", episode) \
                .field("step", step) \
                .field("reward", float(reward)) \
                .field("action_mean", float(sum(action) / len(action))) \
                .field("action_std", float(sum((x - sum(action)/len(action))**2 for x in action) / len(action))**0.5) \
                .time(dt)
            
            self.write_api.write(bucket=self.buckets["control_data"], record=point)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store RL metrics: {e}")
            return False
    
    async def store_system_status(self, component: str, status: str, 
                                 details: Dict[str, Any] = None,
                                 timestamp: Optional[datetime] = None):
        """Store system component status"""
        if not self.write_api:
            return False
        
        try:
            dt = timestamp or datetime.now()
            
            point = Point("system_status") \
                .tag("component", component) \
                .field("status", status)
            
            # Add details as fields
            if details:
                for key, value in details.items():
                    if isinstance(value, (int, float)):
                        point = point.field(key, value)
                    else:
                        point = point.field(key, str(value))
            
            point = point.time(dt)
            
            self.write_api.write(bucket=self.buckets["system_data"], record=point)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store system status: {e}")
            return False
    
    async def query_zone_temperatures(self, zone_id: Optional[str] = None, 
                                     start_time: Optional[datetime] = None,
                                     end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Query zone temperature data"""
        if not self.query_api:
            return []
        
        try:
            # Default time range - last 24 hours
            if not start_time:
                start_time = datetime.now() - timedelta(hours=24)
            if not end_time:
                end_time = datetime.now()
            
            # Build query
            zone_filter = f'|> filter(fn: (r) => r["zone_id"] == "{zone_id}")' if zone_id else ""
            
            query = f'''
                from(bucket: "{self.buckets["thermal_data"]}")
                |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
                |> filter(fn: (r) => r["_measurement"] == "zone_temperature")
                {zone_filter}
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            result = self.query_api.query(query)
            
            data = []
            for table in result:
                for record in table.records:
                    data.append({
                        "time": record.get_time(),
                        "zone_id": record.values.get("zone_id"),
                        "temperature": record.values.get("temperature"),
                        "setpoint": record.values.get("setpoint")
                    })
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to query zone temperatures: {e}")
            return []
    
    async def query_comfort_metrics(self, building_id: Optional[str] = None,
                                   start_time: Optional[datetime] = None,
                                   end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Query comfort metrics data with optional building filter"""
        if not self.query_api:
            return []
        
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(hours=24)
            if not end_time:
                end_time = datetime.now()
            
            building_filter = f'|> filter(fn: (r) => r["building_id"] == "{building_id}")' if building_id else ""
            
            query = f'''
                from(bucket: "{self.buckets["comfort_data"]}")
                |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
                |> filter(fn: (r) => r["_measurement"] == "comfort_metrics")
                {building_filter}
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            result = self.query_api.query(query)
            
            data = []
            for table in result:
                for record in table.records:
                    data.append({
                        "time": record.get_time(),
                        "building_id": record.values.get("building_id"),
                        "pmv": record.values.get("pmv"),
                        "ppd": record.values.get("ppd")
                    })
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to query comfort metrics: {e}")
            return []
    
    async def query_energy_consumption(self, building_id: Optional[str] = None,
                                      start_time: Optional[datetime] = None,
                                      end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Query energy consumption data with optional building filter"""
        if not self.query_api:
            return {}
        
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(hours=24)
            if not end_time:
                end_time = datetime.now()
            
            # Building filter if specified
            building_filter = f'|> filter(fn: (r) => r["building_id"] == "{building_id}")' if building_id else ""
            
            # Query building energy data
            building_query = f'''
                from(bucket: "{self.buckets["energy_data"]}")
                |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
                |> filter(fn: (r) => r["_measurement"] == "building_energy")
                {building_filter}
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            # Query zone energy data
            zone_query = f'''
                from(bucket: "{self.buckets["energy_data"]}")
                |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
                |> filter(fn: (r) => r["_measurement"] == "zone_energy")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            building_result = self.query_api.query(building_query)
            zone_result = self.query_api.query(zone_query)
            
            building_data = []
            zone_data = []
            
            for table in building_result:
                for record in table.records:
                    building_data.append({
                        "time": record.get_time(),
                        "building_id": record.values.get("building_id"),
                        "total_electricity_power": record.values.get("total_electricity_power"),
                        "total_lights_energy": record.values.get("total_lights_energy"),
                        "total_equipment_energy": record.values.get("total_equipment_energy"),
                        "total_fan_energy": record.values.get("total_fan_energy")
                    })
            
            for table in zone_result:
                for record in table.records:
                    zone_data.append({
                        "time": record.get_time(),
                        "zone_id": record.values.get("zone_id"),
                        "heating_energy": record.values.get("heating_energy"),
                        "cooling_energy": record.values.get("cooling_energy")
                    })
            
            return {
                "building_energy": building_data,
                "zone_energy": zone_data
            }
            
        except Exception as e:
            logger.error(f"Failed to query energy data: {e}")
            return {}
    
    async def query_environmental_data(self, start_time: Optional[datetime] = None,
                                     end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Query environmental data"""
        if not self.query_api:
            return []
        
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(hours=24)
            if not end_time:
                end_time = datetime.now()
            
            query = f'''
                from(bucket: "{self.buckets["thermal_data"]}")
                |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
                |> filter(fn: (r) => r["_measurement"] == "environmental")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            result = self.query_api.query(query)
            
            data = []
            for table in result:
                for record in table.records:
                    data.append({
                        "time": record.get_time(),
                        "outdoor_temperature": record.values.get("outdoor_temperature"),
                        "global_horizontal_irradiance": record.values.get("global_horizontal_irradiance"),
                        "direct_solar_radiation": record.values.get("direct_solar_radiation"),
                        "diffuse_solar_radiation": record.values.get("diffuse_solar_radiation")
                    })
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to query environmental data: {e}")
            return []
    
    async def get_latest_system_status(self) -> Dict[str, Any]:
        """Get latest system component statuses"""
        if not self.query_api:
            return {}
        
        try:
            query = f'''
                from(bucket: "{self.buckets["system_data"]}")
                |> range(start: -1h)
                |> filter(fn: (r) => r["_measurement"] == "system_status")
                |> group(columns: ["component"])
                |> last()
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            result = self.query_api.query(query)
            
            status_data = {}
            for table in result:
                for record in table.records:
                    component = record.values.get("component")
                    if component:
                        status_data[component] = {
                            "time": record.get_time(),
                            "status": record.values.get("status"),
                            "details": {k: v for k, v in record.values.items() 
                                      if k not in ["_time", "_measurement", "component", "status"]}
                        }
            
            return status_data
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {}
    
    async def query_building_data(self, 
                                building_id: str, 
                                start_time: datetime, 
                                end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Query all data for a specific building"""
        if not self.query_api:
            return []
        
        try:
            if end_time is None:
                end_time = datetime.now()
            
            query = f'''
            from(bucket: "{self.buckets['thermal_data']}")
                |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
                |> filter(fn: (r) => r.building_id == "{building_id}")
                |> group(columns: ["_time"])
                |> sort(columns: ["_time"])
            '''
            
            tables = self.query_api.query(query, org=self.org)
            
            results = []
            current_record = {}
            current_time = None
            
            for table in tables:
                for record in table.records:
                    record_time = record.get_time()
                    
                    if current_time != record_time:
                        if current_record:
                            results.append(current_record)
                        current_record = {
                            "timestamp": record_time.isoformat(),
                            "building_id": building_id,
                            "temperatures": {},
                            "comfort": {},
                            "metadata": {}
                        }
                        current_time = record_time
                    
                    field_name = record.get_field()
                    value = record.get_value()
                    
                    # Categorize the field
                    if "temp" in field_name.lower() or field_name in ["outdoor"]:
                        current_record["temperatures"][field_name] = value
                    elif field_name in ["PMV", "PPD"]:
                        current_record["comfort"][field_name] = value
                    else:
                        current_record["metadata"][field_name] = value
            
            if current_record:
                results.append(current_record)
            
            logger.info(f"Retrieved {len(results)} building data records for {building_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to query building data: {e}")
            return []
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data beyond retention period"""
        if not self.delete_api:
            return False
        
        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            
            for bucket_name in self.buckets.values():
                predicate = f'_time < {cutoff_time.isoformat()}Z'
                
                self.delete_api.delete(
                    start=datetime(1970, 1, 1),
                    stop=cutoff_time,
                    predicate=predicate,
                    bucket=bucket_name
                )
                
                logger.info(f"Cleaned up data older than {days_to_keep} days from {bucket_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return False
    
    async def close(self):
        """Close InfluxDB connection"""
        if self.client:
            self.client.close()
            logger.info("InfluxDB connection closed")


# Compatibility alias for existing code
InfluxDBService = VBMSInfluxDB
