# Fast FMU Simulation Tool

This tool runs fast FMU simulations to extract complete building performance data for analysis.

## Overview

The VBMS platform runs simulations in real-time (1:1 time ratio) which is perfect for live demonstrations but takes too long to generate complete datasets for analysis. This tool runs the same FMU files at maximum speed to generate complete simulation results quickly.

## Files

### Core Simulation Script
- `fast_simulation.py` - Main script that outputs results to CSV files
- `analyze_results.py` - Analysis tool for processing simulation results

### Docker Environment
- `Dockerfile` - Containerized environment for cross-platform compatibility
- `requirements.txt` - Python dependencies for the simulation tool
- `docker-compose.yml` - Docker Compose configuration for easy deployment
- `run.sh` - Convenience script for running simulations

### Output Data
- `simulation_results/` - CSV output files (created automatically)

## Usage

### Quick Start
1. Navigate to the fast-simulation directory: `cd fast-simulation`
2. Run the simulation: `python3 fast_simulation.py`
3. When prompted, select a configuration:
   - Press **Enter** for default (1 day simulation)
   - Type **1** for 1-day quick test (~288 data points)
   - Type **2** for 1-week analysis (~672 data points)
   - Type **3** for 1-month analysis (~720 data points)
   - Type **4** for 1-year full analysis (~8,760 data points)
4. Wait for simulation to complete (you'll see progress messages)
5. Analyze results: `python3 analyze_results.py`

**Example interaction:**
```
Select configuration (1-4) or press Enter for default (1): 1

Selected: 1_day_5min
Duration: 1.0 days
Step size: 5.0 minutes
Expected data points: 288

2025-08-03 16:45:12,345 - INFO - === Starting Dual FMU Fast Simulation ===
2025-08-03 16:45:12,346 - INFO - --- Simulating Optimized Building ---
2025-08-03 16:45:15,123 - INFO - Simulation completed: 288 time steps
2025-08-03 16:45:15,456 - INFO - ✓ Optimized building simulation completed
2025-08-03 16:45:15,457 - INFO - --- Simulating Non-Optimized Building ---
2025-08-03 16:45:18,234 - INFO - ✓ Non-optimized building simulation completed

✓ Simulation completed successfully!
Results saved in: simulation_results/1_day_5min_20250803_164512
```

### Direct Python Execution
```bash
cd fast-simulation
python fast_simulation.py          # Interactive CSV export
python analyze_results.py          # Analyze generated results
```

### Using Convenience Script
```bash
cd fast-simulation
./run.sh                           # Interactive menu
```

### Docker Execution
```bash
cd fast-simulation
docker-compose up                   # Run in container
```

## Simulation Configurations

When you run the simulation, you'll be prompted to choose:

1. **1 day, 5-minute steps** (~288 data points)
   - **Duration**: 24 hours
   - **Use case**: Quick testing and validation
   - **Time to complete**: ~30 seconds
   - **File size**: ~50 KB per building

2. **1 week, 15-minute steps** (~672 data points)
   - **Duration**: 7 days  
   - **Use case**: Weekly patterns analysis
   - **Time to complete**: ~1 minute
   - **File size**: ~100 KB per building

3. **1 month, 1-hour steps** (~720 data points)
   - **Duration**: 30 days
   - **Use case**: Monthly performance analysis
   - **Time to complete**: ~1-2 minutes
   - **File size**: ~150 KB per building

4. **1 year, 1-hour steps** (~8,760 data points)
   - **Duration**: 365 days
   - **Use case**: Annual energy signatures and full performance analysis
   - **Time to complete**: ~5-10 minutes
   - **File size**: ~1-2 MB per building

**Recommendation**: Start with option 1 (default) for testing, then use option 4 for comprehensive analysis.

## Output Variables

All 40 FMU output variables are extracted:

**Temperature Outputs:**
- OutdoorTemp
- B1SW_Temp, B1SE_Temp, B1NW_Temp, B1NE_Temp, B1Corr_Temp
- B2SW_Temp, B2SE_Temp, B2NW_Temp, B2NE_Temp, B2Corr_Temp

**Energy Outputs:**
- Heating Energy: B1SW_Heating_Energy, B1SE_Heating_Energy, etc.
- Cooling Energy: B1SW_Cooling_Energy, B1SE_Cooling_Energy, etc.
- Total_Electricity_Power, Total_Lights_Energy, Total_Equipment_Energy, Total_Fan_Energy

**Environmental Outputs:**
- Global_Horizontal_Irradiance, Direct_Solar_Radiation, Diffuse_Solar_Radiation

**Comfort Metrics:**
- PMV (Predicted Mean Vote)
- PPD (Predicted Percentage Dissatisfied)

## Output Format

### CSV Files
Each simulation generates two CSV files in `simulation_results/[config]_[timestamp]/`:
- `optimized_building_[timestamp].csv` - Results from optimized building FMU
- `non_optimized_building_[timestamp].csv` - Results from non-optimized building FMU

**CSV Structure:**
- `timestamp` - ISO format timestamp
- `simulation_time` - Seconds from simulation start  
- All 40 FMU output variables with their values

### Analysis Results
After running `analyze_results.py`, you'll get:
- `simulation_analysis_report_[timestamp].txt` - Comprehensive text report
- `energy_comparison.png` - Energy consumption comparison chart
- `temperature_profile.png` - Temperature profiles over time

## Performance

- **Speed**: 100-1000x faster than real-time simulation
- **1 year simulation**: Completes in minutes instead of 365 days
- **Data volume**: ~8,760 data points for 1-year hourly simulation
- **File sizes**: Typically 1-10 MB for CSV files, similar for database

## Integration with Main VBMS

This tool complements the main VBMS system:
- **Main VBMS**: Real-time demo mode (perfect for presentations)
- **Fast Simulation**: Complete dataset generation (perfect for analysis)

Use the fast simulation tools to generate comprehensive datasets for:
- Energy signature analysis
- Annual performance reports
- Building optimization studies
- Comparative analysis between optimized/non-optimized buildings
