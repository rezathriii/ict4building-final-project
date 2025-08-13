#!/usr/bin/env python3
"""
Extract simulation data from EnergyPlus SQLite database and create CSV files
with the requested variables. Updated to handle different variable indices between buildings.
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime

def get_variable_mapping(db_path):
    """Get variable mapping for the specific database"""
    conn = sqlite3.connect(db_path)
    
    # Query to get all variable indices and names
    query = """
    SELECT ReportDataDictionaryIndex, Name, KeyValue, Units 
    FROM ReportDataDictionary;
    """
    
    variables = pd.read_sql_query(query, conn)
    conn.close()
    
    # Create mapping based on what's available
    mapping = {}
    
    # Outdoor temperature
    outdoor_temp = variables[variables['Name'] == 'Site Outdoor Air Drybulb Temperature']
    if not outdoor_temp.empty:
        mapping['OutdoorTemp'] = (outdoor_temp.iloc[0]['ReportDataDictionaryIndex'], 'Site Outdoor Air Drybulb Temperature')
    
    # Solar radiation
    direct_solar = variables[variables['Name'] == 'Site Direct Solar Radiation Rate per Area']
    if not direct_solar.empty:
        mapping['Global_Horizontal_Irradiance'] = (direct_solar.iloc[0]['ReportDataDictionaryIndex'], 'Site Direct Solar Radiation Rate per Area')
        mapping['Direct_Solar_Radiation'] = (direct_solar.iloc[0]['ReportDataDictionaryIndex'], 'Site Direct Solar Radiation Rate per Area')
    
    diffuse_solar = variables[variables['Name'] == 'Site Diffuse Solar Radiation Rate per Area']
    if not diffuse_solar.empty:
        mapping['Diffuse_Solar_Radiation'] = (diffuse_solar.iloc[0]['ReportDataDictionaryIndex'], 'Site Diffuse Solar Radiation Rate per Area')
    
    # Zone temperatures
    zones = [
        ('B1SW_Temp', 'BLOCK1:OFFICEXSWX1F'),
        ('B1SE_Temp', 'BLOCK1:OFFICEXSEX1F'),
        ('B1NW_Temp', 'BLOCK1:OFFICEXNWX1F'),
        ('B1NE_Temp', 'BLOCK1:OFFICEXNEX1F'),
        ('B1Corr_Temp', 'BLOCK1:CORRIDORX1F'),
        ('B2SW_Temp', 'BLOCK2:OFFICEXSWX2F'),
        ('B2SE_Temp', 'BLOCK2:OFFICEXSEX2F'),
        ('B2NW_Temp', 'BLOCK2:OFFICEXNWX2F'),
        ('B2NE_Temp', 'BLOCK2:OFFICEXNEX2F'),
        ('B2Corr_Temp', 'BLOCK2:CORRIDORX2F'),
    ]
    
    for var_name, zone in zones:
        zone_temp = variables[(variables['Name'] == 'Zone Operative Temperature') & (variables['KeyValue'] == zone)]
        if not zone_temp.empty:
            mapping[var_name] = (zone_temp.iloc[0]['ReportDataDictionaryIndex'], f'Zone Operative Temperature {zone}')
    
    # Heating energy - use Zone Ideal Loads Supply Air Total Heating Rate
    heating_zones = [
        ('B1SW_Heating_Energy', 'BLOCK1:OFFICEXSWX1F IDEAL LOADS AIR'),
        ('B1SE_Heating_Energy', 'BLOCK1:OFFICEXSEX1F IDEAL LOADS AIR'),
        ('B1NW_Heating_Energy', 'BLOCK1:OFFICEXNWX1F IDEAL LOADS AIR'),
        ('B1NE_Heating_Energy', 'BLOCK1:OFFICEXNEX1F IDEAL LOADS AIR'),
        ('B1Corr_Heating_Energy', 'BLOCK1:CORRIDORX1F IDEAL LOADS AIR'),
        ('B2SW_Heating_Energy', 'BLOCK2:OFFICEXSWX2F IDEAL LOADS AIR'),
        ('B2SE_Heating_Energy', 'BLOCK2:OFFICEXSEX2F IDEAL LOADS AIR'),
        ('B2NW_Heating_Energy', 'BLOCK2:OFFICEXNWX2F IDEAL LOADS AIR'),
        ('B2NE_Heating_Energy', 'BLOCK2:OFFICEXNEX2F IDEAL LOADS AIR'),
        ('B2Corr_Heating_Energy', 'BLOCK2:CORRIDORX2F IDEAL LOADS AIR'),
    ]
    
    for var_name, zone in heating_zones:
        heating = variables[(variables['Name'] == 'Zone Ideal Loads Supply Air Total Heating Rate') & (variables['KeyValue'] == zone)]
        if not heating.empty:
            mapping[var_name] = (heating.iloc[0]['ReportDataDictionaryIndex'], f'Zone Ideal Loads Supply Air Total Heating Rate {zone}')
    
    # Cooling energy - use Zone Ideal Loads Supply Air Total Cooling Rate
    cooling_zones = [
        ('B1SW_Cooling_Energy', 'BLOCK1:OFFICEXSWX1F IDEAL LOADS AIR'),
        ('B1SE_Cooling_Energy', 'BLOCK1:OFFICEXSEX1F IDEAL LOADS AIR'),
        ('B1NW_Cooling_Energy', 'BLOCK1:OFFICEXNWX1F IDEAL LOADS AIR'),
        ('B1NE_Cooling_Energy', 'BLOCK1:OFFICEXNEX1F IDEAL LOADS AIR'),
        ('B1Corr_Cooling_Energy', 'BLOCK1:CORRIDORX1F IDEAL LOADS AIR'),
        ('B2SW_Cooling_Energy', 'BLOCK2:OFFICEXSWX2F IDEAL LOADS AIR'),
        ('B2SE_Cooling_Energy', 'BLOCK2:OFFICEXSEX2F IDEAL LOADS AIR'),
        ('B2NW_Cooling_Energy', 'BLOCK2:OFFICEXNWX2F IDEAL LOADS AIR'),
        ('B2NE_Cooling_Energy', 'BLOCK2:OFFICEXNEX2F IDEAL LOADS AIR'),
        ('B2Corr_Cooling_Energy', 'BLOCK2:CORRIDORX2F IDEAL LOADS AIR'),
    ]
    
    for var_name, zone in cooling_zones:
        cooling = variables[(variables['Name'] == 'Zone Ideal Loads Supply Air Total Cooling Rate') & (variables['KeyValue'] == zone)]
        if not cooling.empty:
            mapping[var_name] = (cooling.iloc[0]['ReportDataDictionaryIndex'], f'Zone Ideal Loads Supply Air Total Cooling Rate {zone}')
    
    # Electricity
    total_elec = variables[variables['Name'] == 'Facility Total Electricity Demand Rate']
    if not total_elec.empty:
        mapping['Total_Electricity_Power'] = (total_elec.iloc[0]['ReportDataDictionaryIndex'], 'Facility Total Electricity Demand Rate')
    
    lights_elec = variables[variables['Name'] == 'InteriorLights:Electricity']
    if not lights_elec.empty:
        mapping['Total_Lights_Energy'] = (lights_elec.iloc[0]['ReportDataDictionaryIndex'], 'InteriorLights:Electricity')
    
    equip_elec = variables[variables['Name'] == 'InteriorEquipment:Electricity']
    if not equip_elec.empty:
        mapping['Total_Equipment_Energy'] = (equip_elec.iloc[0]['ReportDataDictionaryIndex'], 'InteriorEquipment:Electricity')
    
    # Fan energy - use first available zone
    fan_energy = variables[variables['Name'] == 'Zone Ventilation Fan Electricity Energy']
    if not fan_energy.empty:
        mapping['Total_Fan_Energy'] = (fan_energy.iloc[0]['ReportDataDictionaryIndex'], 'Zone Ventilation Fan Electricity Energy')
    
    # PMV and PPD - use first available zone
    pmv = variables[variables['Name'] == 'Zone Thermal Comfort Fanger Model PMV']
    if not pmv.empty:
        mapping['PMV'] = (pmv.iloc[0]['ReportDataDictionaryIndex'], 'Zone Thermal Comfort Fanger Model PMV')
    
    ppd = variables[variables['Name'] == 'Zone Thermal Comfort Fanger Model PPD']
    if not ppd.empty:
        mapping['PPD'] = (ppd.iloc[0]['ReportDataDictionaryIndex'], 'Zone Thermal Comfort Fanger Model PPD')
    
    return mapping

def create_csv_from_sql(db_path, output_path, simulation_name):
    """Extract data from SQLite database and create CSV file"""
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    
    # Get variable mapping for this specific database
    variable_mapping = get_variable_mapping(db_path)
    
    # Get time data
    time_query = """
    SELECT TimeIndex, Month, Day, Hour, Minute, Interval, IntervalType, SimulationDays
    FROM Time 
    ORDER BY TimeIndex;
    """
    time_df = pd.read_sql_query(time_query, conn)
    
    # Create main dataframe with time information
    df = pd.DataFrame()
    df['TimeIndex'] = time_df['TimeIndex']
    df['DateTime'] = time_df.apply(lambda row: f"2005-{row['Month']:02d}-{row['Day']:02d} {row['Hour']:02d}:{row['Minute']:02d}", axis=1)
    df['Month'] = time_df['Month']
    df['Day'] = time_df['Day']
    df['Hour'] = time_df['Hour']
    df['Minute'] = time_df['Minute']
    
    # Extract each variable
    for var_name, (var_index, description) in variable_mapping.items():
        try:
            # Query the specific variable data
            data_query = f"""
            SELECT rd.TimeIndex, rd.Value
            FROM ReportData rd
            WHERE rd.ReportDataDictionaryIndex = {var_index}
            ORDER BY rd.TimeIndex;
            """
            
            var_data = pd.read_sql_query(data_query, conn)
            
            if not var_data.empty:
                # Merge with main dataframe
                df = df.merge(var_data.rename(columns={'Value': var_name}), 
                            on='TimeIndex', how='left')
                print(f"✓ Extracted {var_name}: {len(var_data)} records")
            else:
                print(f"✗ No data found for {var_name} (index {var_index})")
                df[var_name] = 0  # Add column with zeros if no data
                
        except Exception as e:
            print(f"✗ Error extracting {var_name}: {e}")
            df[var_name] = 0  # Add column with zeros if error
    
    # Convert heating/cooling power (W) to energy (J) 
    # Assuming 10-minute timesteps (600 seconds)
    timestep_seconds = 600  # 10 minutes
    
    energy_vars = [var for var in df.columns if 'Heating_Energy' in var or 'Cooling_Energy' in var]
    for var in energy_vars:
        if var in df.columns:
            df[var] = df[var] * timestep_seconds  # Convert W to J
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✓ Saved {simulation_name} data to {output_path}")
    print(f"  - {len(df)} timesteps")
    print(f"  - {len(df.columns)} variables")
    
    # Display sample data
    print(f"\nFirst 5 rows of {simulation_name} data:")
    print(df.head())
    
    # Display summary statistics for key variables
    key_vars = ['OutdoorTemp', 'B1SW_Temp', 'B1SW_Heating_Energy', 'B1SW_Cooling_Energy', 'Total_Electricity_Power']
    available_key_vars = [var for var in key_vars if var in df.columns]
    if available_key_vars:
        print(f"\nSummary statistics for key variables in {simulation_name}:")
        print(df[available_key_vars].describe())
    
    conn.close()
    return df

def main():
    """Main function to process both simulations"""
    print("Extracting EnergyPlus simulation data with updated variable mapping...")
    print("=" * 70)
    
    # Process optimized building
    optimized_db = "outputs/optimized/eplusout.sql"
    optimized_csv = "outputs/optimized_building_results.csv"
    
    if os.path.exists(optimized_db):
        print("\n1. Processing OPTIMIZED building simulation...")
        df_opt = create_csv_from_sql(optimized_db, optimized_csv, "Optimized")
    else:
        print(f"✗ Database not found: {optimized_db}")
    
    # Process non-optimized building
    not_optimized_db = "outputs/not_optimized/eplusout.sql"
    not_optimized_csv = "outputs/not_optimized_building_results.csv"
    
    if os.path.exists(not_optimized_db):
        print("\n2. Processing NOT OPTIMIZED building simulation...")
        df_not_opt = create_csv_from_sql(not_optimized_db, not_optimized_csv, "Not Optimized")
    else:
        print(f"✗ Database not found: {not_optimized_db}")
    
    print("\n" + "=" * 70)
    print("✓ Data extraction completed!")
    print(f"✓ Optimized results: {optimized_csv}")
    print(f"✓ Not optimized results: {not_optimized_csv}")
    
    # Check file sizes
    if os.path.exists(optimized_csv):
        size_opt = os.path.getsize(optimized_csv) / 1024 / 1024  # MB
        print(f"  - Optimized CSV size: {size_opt:.1f} MB")
    
    if os.path.exists(not_optimized_csv):
        size_not_opt = os.path.getsize(not_optimized_csv) / 1024 / 1024  # MB
        print(f"  - Not optimized CSV size: {size_not_opt:.1f} MB")

if __name__ == "__main__":
    main()
