# EnergyPlus Direct Simulation

This directory contains the IDF and EPW files needed to run EnergyPlus simulations directly.

## Files Structure
```
inputs/
├── optimized_office_building_clean.idf     # Optimized building model (ExternalInterface removed)
├── not_optimized_office_building_clean.idf # Non-optimized building model (ExternalInterface removed)
├── optimized_office_building.idf           # Original files (for reference)
├── not_optimized_office_building.idf       # Original files (for reference)
└── paris_2005.epw                          # Paris weather data

outputs/                                     # EnergyPlus output files will be generated here
```

## EnergyPlus Commands

Run these commands from the `fast-simulation` directory after installing EnergyPlus 24.1.0:

### Optimized Building Simulation:
```bash
energyplus -w inputs/paris_2005.epw -d outputs/optimized inputs/optimized_office_building_clean.idf
```

### Non-Optimized Building Simulation:
```bash
energyplus -w inputs/paris_2005.epw -d outputs/not_optimized inputs/not_optimized_office_building_clean.idf
```

## What Was Fixed

The original IDF files contained `ExternalInterface` objects that were used for FMU export. These have been removed from the `*_clean.idf` files to allow normal EnergyPlus simulation without requiring socket configuration files.

## Output Files

After running the simulations, you'll find these key files in the outputs directories:

- `outputs/optimized/eplusout.csv` - All simulation variables for optimized building
- `outputs/not_optimized/eplusout.csv` - All simulation variables for non-optimized building

The CSV files will contain the 40+ variables you specified, including:
- Zone temperatures
- HVAC power consumption  
- Heating/cooling energy
- Lighting and equipment energy
- Outdoor conditions
- And many more building performance metrics

## Notes

- Each simulation will create its own subdirectory in `outputs/`
- The `-w` flag specifies the weather file
- The `-d` flag specifies the output directory
- Make sure EnergyPlus 24.1.0 is installed and in your PATH
- The cleaned IDF files have had ExternalInterface objects removed to prevent socket.cfg errors
