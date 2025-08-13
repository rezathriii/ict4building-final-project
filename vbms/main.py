#!/usr/bin/env python3
"""
VBMS Platform - Virtual Building Management System
Enhanced FastAPI application with dual FMU simulation and data pipeline
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta

from services.data_pipeline import VBMSDataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VBMS Platform",
    description="Virtual Building Management System with dual FMU simulation",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
data_pipeline: Optional[VBMSDataPipeline] = None

# Pydantic models
class SimulationConfig(BaseModel):
    start_time: float = 0.0
    end_time: float = 86400.0  # 24 hours
    timestep: float = 300.0    # 5 minutes

class ZoneSetpoint(BaseModel):
    zone_id: str
    setpoint: float
    
class SystemCommand(BaseModel):
    command: str
    parameters: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global data_pipeline
    
    try:
        logger.info("Starting VBMS Platform...")
        
        # Initialize data pipeline
        data_pipeline = VBMSDataPipeline()
        await data_pipeline.initialize()
        
        # Auto-start simulation for live data visualization
        await auto_start_simulation()
        
        logger.info("VBMS Platform started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start VBMS Platform: {e}")
        raise

async def auto_start_simulation():
    """Automatically start simulation and begin data generation"""
    global data_pipeline
    
    try:
        # Start the simulation
        await data_pipeline.start_simulation()
        logger.info("Auto-started thermal simulation")
        
        # Create background task for continuous simulation steps
        asyncio.create_task(continuous_simulation())
        
    except Exception as e:
        logger.error(f"Failed to auto-start simulation: {e}")

async def continuous_simulation():
    """Background task to continuously generate simulation data"""
    global data_pipeline
    
    try:
        # Wait a bit for services to fully stabilize
        await asyncio.sleep(10)
        
        step_count = 0
        while data_pipeline and data_pipeline.running:
            try:
                # Perform simulation step
                await data_pipeline.step_simulation()
                step_count += 1
                
                logger.info(f"Auto-simulation step {step_count} completed")
                
                # Wait 2 seconds between steps for faster testing
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in continuous simulation step {step_count}: {e}")
                await asyncio.sleep(5)  # Wait before retrying
                
    except Exception as e:
        logger.error(f"Continuous simulation task failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up services on shutdown"""
    global data_pipeline
    
    try:
        logger.info("Shutting down VBMS Platform...")
        
        if data_pipeline:
            await data_pipeline.shutdown()
        
        logger.info("VBMS Platform shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "VBMS Platform API",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not data_pipeline:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    # Check service health
    mqtt_status = data_pipeline.mqtt_service.is_connected()
    influx_status = bool(data_pipeline.influxdb_service.client)
    sim_status = data_pipeline.running
    
    health_status = {
        "status": "healthy" if all([mqtt_status, influx_status]) else "degraded",
        "services": {
            "mqtt": "online" if mqtt_status else "offline",
            "influxdb": "online" if influx_status else "offline",
            "simulation": "running" if sim_status else "stopped"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return health_status

@app.get("/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    if not data_pipeline:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    await data_pipeline.publish_system_status()
    
    return {
        "message": "System status published to MQTT",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/simulation/start")
async def start_simulation():
    """Start thermal simulation"""
    if not data_pipeline:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    try:
        await data_pipeline.start_simulation()
        return {
            "message": "Simulation started",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulation/stop")
async def stop_simulation():
    """Stop thermal simulation"""
    if not data_pipeline:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    try:
        await data_pipeline.stop_simulation()
        return {
            "message": "Simulation stopped",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulation/reset")
async def reset_simulation():
    """Reset thermal simulation"""
    if not data_pipeline:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    try:
        await data_pipeline.reset_simulation()
        return {
            "message": "Simulation reset",
            "episode": data_pipeline.episode,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulation/step")
async def perform_simulation_step():
    """Perform a single simulation step"""
    if not data_pipeline:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    try:
        result = await data_pipeline.step_simulation()
        return {
            "message": "Simulation step completed",
            "step": data_pipeline.simulation_step,
            "timestamp": datetime.now().isoformat(),
            "results": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/zones/temperatures")
async def get_zone_temperatures(hours: int = 24):
    """Get zone temperature data from InfluxDB"""
    if not data_pipeline:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    try:
        data = await data_pipeline.influxdb_service.query_zone_temperatures(
            start_time=datetime.now() - timedelta(hours=hours)
        )
        
        return {
            "data": data,
            "count": len(data),
            "hours": hours,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/zones/{zone_id}/comfort")
async def get_zone_comfort_metrics(zone_id: str, hours: int = 24):
    """Get comfort metrics for a specific building"""
    if not data_pipeline:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    try:
        # Map zone_id to building_id
        building_id = "optimized_building" if "optimized" in zone_id or not zone_id.endswith("_non_opt") else "non_optimized_building"
        
        data = await data_pipeline.influxdb_service.query_comfort_metrics(
            building_id=building_id,
            start_time=datetime.now() - timedelta(hours=hours)
        )
        
        return {
            "building_id": building_id,
            "data": data,
            "count": len(data),
            "hours": hours,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/energy/consumption")
async def get_energy_consumption(building_id: Optional[str] = None, hours: int = 24):
    """Get energy consumption data"""
    if not data_pipeline:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    try:
        data = await data_pipeline.influxdb_service.query_energy_consumption(
            building_id=building_id,
            start_time=datetime.now() - timedelta(hours=hours)
        )
        
        return {
            "building_id": building_id,
            "data": data,
            "hours": hours,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/environment/weather")
async def get_environmental_data(hours: int = 24):
    """Get environmental data (solar radiation, outdoor temperature)"""
    if not data_pipeline:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    try:
        data = await data_pipeline.influxdb_service.query_environmental_data(
            start_time=datetime.now() - timedelta(hours=hours)
        )
        
        return {
            "data": data,
            "count": len(data),
            "hours": hours,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/buildings/{building_id}/energy")
async def get_building_energy(building_id: str, hours: int = 24):
    """Get energy data for a specific building"""
    if not data_pipeline:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    try:
        data = await data_pipeline.influxdb_service.query_energy_consumption(
            building_id=building_id,
            start_time=datetime.now() - timedelta(hours=hours)
        )
        
        return {
            "building_id": building_id,
            "data": data,
            "hours": hours,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/buildings/{building_id}/comfort")
async def get_building_comfort(building_id: str, hours: int = 24):
    """Get comfort metrics for a specific building"""
    if not data_pipeline:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    try:
        data = await data_pipeline.influxdb_service.query_comfort_metrics(
            building_id=building_id,
            start_time=datetime.now() - timedelta(hours=hours)
        )
        
        return {
            "building_id": building_id,
            "data": data,
            "count": len(data),
            "hours": hours,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulation/debug")
async def get_simulation_debug():
    """Debug endpoint to check current simulation data structure"""
    if not data_pipeline or not data_pipeline.dual_fmu_simulator:
        raise HTTPException(status_code=503, detail="Simulation not available")
    
    try:
        # Get current simulation step results
        results = data_pipeline.dual_fmu_simulator.step()
        return {
            "simulation_step": data_pipeline.simulation_step,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get debug data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/zones/{zone_id}/setpoint")
async def set_zone_setpoint(zone_id: str, setpoint_data: ZoneSetpoint):
    """Set temperature setpoint for a zone"""
    if not data_pipeline:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    try:
        # Publish setpoint change to MQTT
        await data_pipeline.mqtt_service.publish(
            f"vbms/zones/{zone_id}/setpoint",
            {"setpoint": setpoint_data.setpoint, "timestamp": datetime.now().isoformat()}
        )
        
        return {
            "message": f"Setpoint for zone {zone_id} set to {setpoint_data.setpoint}°C",
            "zone_id": zone_id,
            "setpoint": setpoint_data.setpoint,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/zones/list")
async def get_zones():
    """Get list of available zones"""
    if not data_pipeline or not data_pipeline.dual_fmu_simulator:
        raise HTTPException(status_code=503, detail="Simulation not available")
    
    try:
        zone_names = data_pipeline.dual_fmu_simulator.get_zone_names()
        # Add both optimized and non-optimized versions
        all_zones = []
        for zone in zone_names:
            all_zones.append(f"{zone}_optimized")
            all_zones.append(f"{zone}_non_optimized")
        
        return {
            "zones": all_zones,
            "base_zones": zone_names,
            "count": len(all_zones),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get zone list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulation/comparison")
async def get_simulation_comparison(hours: int = 1):
    """Get comprehensive comparison data between optimized and non-optimized buildings"""
    if not data_pipeline:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    try:
        start_time = datetime.now() - timedelta(hours=hours)
        
        # Get temperature data
        optimized_temps = await data_pipeline.influxdb_service.query_zone_temperatures(
            start_time=start_time
        )
        
        # Get energy data
        optimized_energy = await data_pipeline.influxdb_service.query_energy_consumption(
            building_id="optimized_building",
            start_time=start_time
        )
        
        non_optimized_energy = await data_pipeline.influxdb_service.query_energy_consumption(
            building_id="non_optimized_building", 
            start_time=start_time
        )
        
        # Get comfort data
        optimized_comfort = await data_pipeline.influxdb_service.query_comfort_metrics(
            building_id="optimized_building",
            start_time=start_time
        )
        
        non_optimized_comfort = await data_pipeline.influxdb_service.query_comfort_metrics(
            building_id="non_optimized_building",
            start_time=start_time
        )
        
        # Get environmental data
        environmental_data = await data_pipeline.influxdb_service.query_environmental_data(
            start_time=start_time
        )
        
        return {
            "optimized_building": {
                "temperatures": [t for t in optimized_temps if not t.get("zone_id", "").endswith("_non_optimized")],
                "energy": optimized_energy,
                "comfort": optimized_comfort
            },
            "non_optimized_building": {
                "temperatures": [t for t in optimized_temps if t.get("zone_id", "").endswith("_non_optimized")],
                "energy": non_optimized_energy,
                "comfort": non_optimized_comfort
            },
            "environmental": environmental_data,
            "hours": hours,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system/command")
async def execute_system_command(command_data: SystemCommand):
    """Execute system command"""
    if not data_pipeline:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    try:
        # Publish command to MQTT
        await data_pipeline.mqtt_service.publish(
            "vbms/system/commands",
            {
                "command": command_data.command,
                "parameters": command_data.parameters or {},
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "message": f"Command '{command_data.command}' executed",
            "command": command_data.command,
            "parameters": command_data.parameters,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Uvicorn server configuration
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
