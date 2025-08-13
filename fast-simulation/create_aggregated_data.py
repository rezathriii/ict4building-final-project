#!/usr/bin/env python3
"""
Create time-aggregated data (hourly, daily, weekly) from the detailed simulation CSV files.
"""

import pandas as pd
import os
from datetime import datetime
import numpy as np

def aggregate_simulation_data(csv_path, output_prefix):
    """Create hourly, daily, and weekly aggregated data from simulation CSV"""
    
    print(f"\nProcessing {csv_path}...")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Fix 24:00 timestamps (convert to 00:00 of next day)
    df['DateTime'] = df['DateTime'].str.replace(' 24:00', ' 00:00')
    
    # Convert DateTime to pandas datetime
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Fix dates that were originally 24:00 (add one day)
    mask = df['Hour'] == 0  # These were originally 24:00
    df.loc[mask, 'DateTime'] = df.loc[mask, 'DateTime'] + pd.Timedelta(days=1)
    
    df.set_index('DateTime', inplace=True)
    
    # Get all numeric columns (excluding time-related columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove time-related columns from aggregation
    time_cols = ['TimeIndex', 'Month', 'Day', 'Hour', 'Minute']
    numeric_cols = [col for col in numeric_cols if col not in time_cols]
    
    print(f"  - Found {len(numeric_cols)} variables to aggregate")
    print(f"  - Time range: {df.index.min()} to {df.index.max()}")
    print(f"  - Total records: {len(df)}")
    
    # 1. HOURLY AGGREGATION
    print("  - Creating hourly aggregation...")
    hourly_df = df[numeric_cols].resample('H').agg({
        # For temperatures: use mean
        **{col: 'mean' for col in numeric_cols if 'Temp' in col or 'PMV' in col or 'PPD' in col},
        # For energy variables: use sum (since they are per timestep)
        **{col: 'sum' for col in numeric_cols if 'Energy' in col},
        # For power and radiation: use mean
        **{col: 'mean' for col in numeric_cols if 'Power' in col or 'Irradiance' in col or 'Radiation' in col}
    })
    
    # Add time information back
    hourly_df['Year'] = hourly_df.index.year
    hourly_df['Month'] = hourly_df.index.month
    hourly_df['Day'] = hourly_df.index.day
    hourly_df['Hour'] = hourly_df.index.hour
    hourly_df['DayOfYear'] = hourly_df.index.dayofyear
    hourly_df['WeekOfYear'] = hourly_df.index.isocalendar().week
    
    # Save hourly data
    hourly_path = f"{output_prefix}_hourly.csv"
    hourly_df.to_csv(hourly_path)
    print(f"    ✓ Saved hourly data: {hourly_path} ({len(hourly_df)} records)")
    
    # 2. DAILY AGGREGATION
    print("  - Creating daily aggregation...")
    daily_df = df[numeric_cols].resample('D').agg({
        # For temperatures: use mean
        **{col: 'mean' for col in numeric_cols if 'Temp' in col or 'PMV' in col or 'PPD' in col},
        # For energy variables: use sum
        **{col: 'sum' for col in numeric_cols if 'Energy' in col},
        # For power and radiation: use mean
        **{col: 'mean' for col in numeric_cols if 'Power' in col or 'Irradiance' in col or 'Radiation' in col}
    })
    
    # Add time information back
    daily_df['Year'] = daily_df.index.year
    daily_df['Month'] = daily_df.index.month
    daily_df['Day'] = daily_df.index.day
    daily_df['DayOfYear'] = daily_df.index.dayofyear
    daily_df['WeekOfYear'] = daily_df.index.isocalendar().week
    
    # Save daily data
    daily_path = f"{output_prefix}_daily.csv"
    daily_df.to_csv(daily_path)
    print(f"    ✓ Saved daily data: {daily_path} ({len(daily_df)} records)")
    
    # 3. WEEKLY AGGREGATION
    print("  - Creating weekly aggregation...")
    weekly_df = df[numeric_cols].resample('W').agg({
        # For temperatures: use mean
        **{col: 'mean' for col in numeric_cols if 'Temp' in col or 'PMV' in col or 'PPD' in col},
        # For energy variables: use sum
        **{col: 'sum' for col in numeric_cols if 'Energy' in col},
        # For power and radiation: use mean
        **{col: 'mean' for col in numeric_cols if 'Power' in col or 'Irradiance' in col or 'Radiation' in col}
    })
    
    # Add time information back
    weekly_df['Year'] = weekly_df.index.year
    weekly_df['Month'] = weekly_df.index.month
    weekly_df['WeekOfYear'] = weekly_df.index.isocalendar().week
    weekly_df['WeekStart'] = weekly_df.index.to_period('W').start_time
    weekly_df['WeekEnd'] = weekly_df.index.to_period('W').end_time
    
    # Save weekly data
    weekly_path = f"{output_prefix}_weekly.csv"
    weekly_df.to_csv(weekly_path)
    print(f"    ✓ Saved weekly data: {weekly_path} ({len(weekly_df)} records)")
    
    return hourly_df, daily_df, weekly_df

def create_summary_statistics(hourly_df, daily_df, weekly_df, output_prefix):
    """Create summary statistics for aggregated data"""
    
    summary_path = f"{output_prefix}_aggregation_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write(f"Time Aggregation Summary for {output_prefix}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Original data aggregated into:\n")
        f.write(f"- Hourly records: {len(hourly_df)}\n")
        f.write(f"- Daily records: {len(daily_df)}\n")
        f.write(f"- Weekly records: {len(weekly_df)}\n\n")
        
        # Sample statistics for key variables
        key_vars = ['OutdoorTemp', 'B1SW_Temp', 'B1SW_Heating_Energy', 'B1SW_Cooling_Energy', 'Total_Electricity_Power']
        available_vars = [var for var in key_vars if var in hourly_df.columns]
        
        if available_vars:
            f.write("Sample Statistics for Key Variables:\n")
            f.write("-" * 40 + "\n\n")
            
            for var in available_vars:
                f.write(f"{var}:\n")
                f.write(f"  Hourly - Mean: {hourly_df[var].mean():.2f}, Std: {hourly_df[var].std():.2f}\n")
                f.write(f"  Daily  - Mean: {daily_df[var].mean():.2f}, Std: {daily_df[var].std():.2f}\n")
                f.write(f"  Weekly - Mean: {weekly_df[var].mean():.2f}, Std: {weekly_df[var].std():.2f}\n\n")
        
        # Energy consumption summary
        energy_vars = [col for col in hourly_df.columns if 'Energy' in col]
        if energy_vars:
            f.write("Energy Consumption Summary (Annual Totals in MJ):\n")
            f.write("-" * 50 + "\n\n")
            
            for var in energy_vars:
                annual_mj = daily_df[var].sum() / 1e6  # Convert J to MJ
                f.write(f"  {var}: {annual_mj:.1f} MJ\n")
    
    print(f"    ✓ Saved summary: {summary_path}")

def main():
    """Main function to create aggregated data"""
    print("Creating time-aggregated data from simulation CSV files...")
    print("=" * 70)
    
    # Process optimized building
    optimized_csv = "optimized_building_results.csv"
    if os.path.exists(optimized_csv):
        print("\n1. Processing OPTIMIZED building data...")
        hourly_opt, daily_opt, weekly_opt = aggregate_simulation_data(
            optimized_csv, 
            "optimized_building"
        )
        create_summary_statistics(hourly_opt, daily_opt, weekly_opt, "optimized_building")
    else:
        print(f"✗ File not found: {optimized_csv}")
    
    # Process non-optimized building
    not_optimized_csv = "not_optimized_building_results.csv"
    if os.path.exists(not_optimized_csv):
        print("\n2. Processing NOT OPTIMIZED building data...")
        hourly_not_opt, daily_not_opt, weekly_not_opt = aggregate_simulation_data(
            not_optimized_csv, 
            "not_optimized_building"
        )
        create_summary_statistics(hourly_not_opt, daily_not_opt, weekly_not_opt, "not_optimized_building")
    else:
        print(f"✗ File not found: {not_optimized_csv}")
    
    print("\n" + "=" * 70)
    print("✓ Time aggregation completed!")
    print("\nFiles created:")
    
    # List all created files
    aggregated_files = [
        "optimized_building_hourly.csv",
        "optimized_building_daily.csv", 
        "optimized_building_weekly.csv",
        "optimized_building_aggregation_summary.txt",
        "not_optimized_building_hourly.csv",
        "not_optimized_building_daily.csv",
        "not_optimized_building_weekly.csv", 
        "not_optimized_building_aggregation_summary.txt"
    ]
    
    for file in aggregated_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / 1024 / 1024
            print(f"  ✓ {file} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {file} (not created)")
    
    print(f"\nAggregation Rules Applied:")
    print(f"  - Temperature variables (Temp, PMV, PPD): Mean values")
    print(f"  - Energy variables (*Energy): Sum values (cumulative)")
    print(f"  - Power/Radiation variables: Mean values")
    print(f"  - Time periods: Hourly (8,760 records), Daily (365 records), Weekly (~52 records)")

if __name__ == "__main__":
    main()
