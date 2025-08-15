#!/usr/bin/env python3
"""
Analyze simulation results from optimized and non-optimized building CSV files.
Create comprehensive tables for LaTeX report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_analyze_data():
    """Load both CSV files and perform comprehensive analysis."""
    
    # Load data
    not_opt_path = Path("outputs/not_optimized/not_optimized_building_results.csv")
    opt_path = Path("outputs/optimized/optimized_building_results.csv")
    
    print("Loading simulation data...")
    df_not_opt = pd.read_csv(not_opt_path)
    df_opt = pd.read_csv(opt_path)
    
    print(f"Not optimized building data: {len(df_not_opt)} records")
    print(f"Optimized building data: {len(df_opt)} records")
    
    # Convert DateTime column (handle 24:00 format)
    df_not_opt['DateTime'] = pd.to_datetime(df_not_opt['DateTime'], format='mixed', errors='coerce')
    df_opt['DateTime'] = pd.to_datetime(df_opt['DateTime'], format='mixed', errors='coerce')
    
    return df_not_opt, df_opt

def calculate_energy_summary(df_not_opt, df_opt):
    """Calculate energy consumption summary statistics."""
    
    # Define energy columns for heating and cooling
    heating_cols = [col for col in df_not_opt.columns if 'Heating_Energy' in col]
    cooling_cols = [col for col in df_not_opt.columns if 'Cooling_Energy' in col]
    
    # Calculate total heating and cooling energy
    df_not_opt['Total_Heating'] = df_not_opt[heating_cols].sum(axis=1)
    df_not_opt['Total_Cooling'] = df_not_opt[cooling_cols].sum(axis=1)
    df_not_opt['Total_HVAC'] = df_not_opt['Total_Heating'] + df_not_opt['Total_Cooling']
    
    df_opt['Total_Heating'] = df_opt[heating_cols].sum(axis=1)
    df_opt['Total_Cooling'] = df_opt[cooling_cols].sum(axis=1)
    df_opt['Total_HVAC'] = df_opt['Total_Heating'] + df_opt['Total_Cooling']
    
    # Annual totals (convert from J to kWh: divide by 3.6e6)
    annual_summary = {
        'Building Type': ['Not Optimized', 'Optimized'],
        'Annual Heating (kWh)': [
            df_not_opt['Total_Heating'].sum() / 3.6e6,
            df_opt['Total_Heating'].sum() / 3.6e6
        ],
        'Annual Cooling (kWh)': [
            df_not_opt['Total_Cooling'].sum() / 3.6e6,
            df_opt['Total_Cooling'].sum() / 3.6e6
        ],
        'Annual HVAC (kWh)': [
            df_not_opt['Total_HVAC'].sum() / 3.6e6,
            df_opt['Total_HVAC'].sum() / 3.6e6
        ],
        'Annual Lighting (kWh)': [
            df_not_opt['Total_Lights_Energy'].sum() / 3.6e6,
            df_opt['Total_Lights_Energy'].sum() / 3.6e6
        ],
        'Annual Equipment (kWh)': [
            df_not_opt['Total_Equipment_Energy'].sum() / 3.6e6,
            df_opt['Total_Equipment_Energy'].sum() / 3.6e6
        ],
        'Annual Fan (kWh)': [
            df_not_opt['Total_Fan_Energy'].sum() / 3.6e6,
            df_opt['Total_Fan_Energy'].sum() / 3.6e6
        ]
    }
    
    # Calculate total energy consumption
    total_not_opt = (df_not_opt['Total_HVAC'].sum() + 
                    df_not_opt['Total_Lights_Energy'].sum() + 
                    df_not_opt['Total_Equipment_Energy'].sum() + 
                    df_not_opt['Total_Fan_Energy'].sum()) / 3.6e6
    
    total_opt = (df_opt['Total_HVAC'].sum() + 
                df_opt['Total_Lights_Energy'].sum() + 
                df_opt['Total_Equipment_Energy'].sum() + 
                df_opt['Total_Fan_Energy'].sum()) / 3.6e6
    
    annual_summary['Total Energy (kWh)'] = [total_not_opt, total_opt]
    
    # Calculate energy savings
    energy_savings = total_not_opt - total_opt
    energy_savings_percent = (energy_savings / total_not_opt) * 100
    
    return pd.DataFrame(annual_summary), energy_savings, energy_savings_percent

def calculate_comfort_summary(df_not_opt, df_opt):
    """Calculate thermal comfort summary statistics."""
    
    comfort_summary = {
        'Building Type': ['Not Optimized', 'Optimized'],
        'Average PMV': [
            df_not_opt['PMV'].mean(),
            df_opt['PMV'].mean()
        ],
        'PMV Std Dev': [
            df_not_opt['PMV'].std(),
            df_opt['PMV'].std()
        ],
        'Average PPD (%)': [
            df_not_opt['PPD'].mean(),
            df_opt['PPD'].mean()
        ],
        'PPD Std Dev (%)': [
            df_not_opt['PPD'].std(),
            df_opt['PPD'].std()
        ],
        'Min PMV': [
            df_not_opt['PMV'].min(),
            df_opt['PMV'].min()
        ],
        'Max PMV': [
            df_not_opt['PMV'].max(),
            df_opt['PMV'].max()
        ],
        'Comfort Hours (PMV ±0.5)': [
            len(df_not_opt[abs(df_not_opt['PMV']) <= 0.5]),
            len(df_opt[abs(df_opt['PMV']) <= 0.5])
        ]
    }
    
    # Calculate comfort percentage
    total_hours = len(df_not_opt)
    comfort_summary['Comfort Percentage (%)'] = [
        (comfort_summary['Comfort Hours (PMV ±0.5)'][0] / total_hours) * 100,
        (comfort_summary['Comfort Hours (PMV ±0.5)'][1] / total_hours) * 100
    ]
    
    return pd.DataFrame(comfort_summary)

def calculate_zone_performance(df_not_opt, df_opt):
    """Calculate zone-wise performance comparison."""
    
    # Zone temperature columns
    temp_cols_not_opt = [col for col in df_not_opt.columns if '_Temp' in col and 'Outdoor' not in col]
    temp_cols_opt = [col for col in df_opt.columns if '_Temp' in col and 'Outdoor' not in col]
    
    # Heating energy columns
    heating_cols = [col for col in df_not_opt.columns if 'Heating_Energy' in col]
    
    zone_data = []
    
    for i, (temp_col, heat_col) in enumerate(zip(temp_cols_not_opt, heating_cols)):
        if temp_col in df_opt.columns and heat_col in df_opt.columns:
            zone_name = temp_col.replace('_Temp', '')
            
            zone_data.append({
                'Zone': zone_name,
                'Avg Temp Not Opt (°C)': df_not_opt[temp_col].mean(),
                'Avg Temp Opt (°C)': df_opt[temp_col].mean(),
                'Temp Std Not Opt (°C)': df_not_opt[temp_col].std(),
                'Temp Std Opt (°C)': df_opt[temp_col].std(),
                'Annual Heating Not Opt (kWh)': df_not_opt[heat_col].sum() / 3.6e6,
                'Annual Heating Opt (kWh)': df_opt[heat_col].sum() / 3.6e6,
                'Heating Savings (%)': ((df_not_opt[heat_col].sum() - df_opt[heat_col].sum()) / df_not_opt[heat_col].sum()) * 100 if df_not_opt[heat_col].sum() > 0 else 0
            })
    
    return pd.DataFrame(zone_data)

def generate_latex_tables(energy_df, comfort_df, zone_df, energy_savings, energy_savings_percent):
    """Generate LaTeX table code for the report."""
    
    latex_output = []
    
    # Energy Performance Table
    latex_output.append("% Energy Performance Comparison Table")
    latex_output.append("\\begin{table}[H]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{Annual Energy Consumption Comparison}")
    latex_output.append("\\label{tab:energy_comparison}")
    latex_output.append("\\begin{tabular}{|l|c|c|c|}")
    latex_output.append("\\hline")
    latex_output.append("\\textbf{Energy Category} & \\textbf{Not Optimized (kWh)} & \\textbf{Optimized (kWh)} & \\textbf{Savings (\\%)} \\\\")
    latex_output.append("\\hline")
    
    for idx, row in energy_df.iterrows():
        if idx == 0:  # Not optimized row
            continue
        not_opt_val = energy_df.iloc[0]
        opt_val = row
        
        for col in ['Annual Heating (kWh)', 'Annual Cooling (kWh)', 'Annual HVAC (kWh)', 
                   'Annual Lighting (kWh)', 'Annual Equipment (kWh)', 'Total Energy (kWh)']:
            if col in energy_df.columns:
                savings_pct = ((not_opt_val[col] - opt_val[col]) / not_opt_val[col]) * 100 if not_opt_val[col] > 0 else 0
                category = col.replace('Annual ', '').replace(' (kWh)', '')
                latex_output.append(f"{category} & {not_opt_val[col]:.0f} & {opt_val[col]:.0f} & {savings_pct:.1f}\\% \\\\")
                latex_output.append("\\hline")
    
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table}")
    latex_output.append("")
    
    # Comfort Performance Table
    latex_output.append("% Thermal Comfort Comparison Table")
    latex_output.append("\\begin{table}[H]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{Thermal Comfort Performance Comparison}")
    latex_output.append("\\label{tab:comfort_comparison}")
    latex_output.append("\\begin{tabular}{|l|c|c|}")
    latex_output.append("\\hline")
    latex_output.append("\\textbf{Comfort Metric} & \\textbf{Not Optimized} & \\textbf{Optimized} \\\\")
    latex_output.append("\\hline")
    
    comfort_metrics = [
        ('Average PMV', 'Average PMV', '.3f'),
        ('Average PPD (%)', 'Average PPD (%)', '.1f'),
        ('PMV Standard Deviation', 'PMV Std Dev', '.3f'),
        ('PPD Standard Deviation (%)', 'PPD Std Dev (%)', '.1f'),
        ('Comfort Hours (PMV ±0.5)', 'Comfort Hours (PMV ±0.5)', '.0f'),
        ('Comfort Percentage (\\%)', 'Comfort Percentage (%)', '.1f')
    ]
    
    for display_name, col_name, fmt in comfort_metrics:
        if col_name in comfort_df.columns:
            not_opt_val = comfort_df.iloc[0][col_name]
            opt_val = comfort_df.iloc[1][col_name]
            formatted_not_opt = format(not_opt_val, fmt)
            formatted_opt = format(opt_val, fmt)
            latex_output.append(f"{display_name} & {formatted_not_opt} & {formatted_opt} \\\\")
            latex_output.append("\\hline")
    
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table}")
    latex_output.append("")
    
    # Zone Performance Summary Table (top 5 zones)
    latex_output.append("% Zone Performance Summary Table")
    latex_output.append("\\begin{table}[H]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{Zone-wise Performance Summary (Top 5 Zones by Energy Savings)}")
    latex_output.append("\\label{tab:zone_performance}")
    latex_output.append("\\begin{tabular}{|l|c|c|c|c|}")
    latex_output.append("\\hline")
    latex_output.append("\\textbf{Zone} & \\textbf{Avg Temp Not Opt (°C)} & \\textbf{Avg Temp Opt (°C)} & \\textbf{Heating Savings (\\%)} & \\textbf{Annual Savings (kWh)} \\\\")
    latex_output.append("\\hline")
    
    # Sort zones by heating savings percentage and take top 5
    zone_sorted = zone_df.nlargest(5, 'Heating Savings (%)')
    
    for _, row in zone_sorted.iterrows():
        annual_savings = row['Annual Heating Not Opt (kWh)'] - row['Annual Heating Opt (kWh)']
        latex_output.append(f"{row['Zone']} & {row['Avg Temp Not Opt (°C)']:.1f} & {row['Avg Temp Opt (°C)']:.1f} & {row['Heating Savings (%)']:.1f}\\% & {annual_savings:.0f} \\\\")
        latex_output.append("\\hline")
    
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table}")
    
    return "\n".join(latex_output)

def main():
    """Main analysis function."""
    print("Starting simulation results analysis...")
    
    # Load data
    df_not_opt, df_opt = load_and_analyze_data()
    
    # Calculate summaries
    energy_summary, energy_savings, energy_savings_percent = calculate_energy_summary(df_not_opt, df_opt)
    comfort_summary = calculate_comfort_summary(df_not_opt, df_opt)
    zone_summary = calculate_zone_performance(df_not_opt, df_opt)
    
    # Print results
    print("\n" + "="*60)
    print("ENERGY PERFORMANCE SUMMARY")
    print("="*60)
    print(energy_summary.round(2))
    print(f"\nTotal Energy Savings: {energy_savings:.0f} kWh ({energy_savings_percent:.1f}%)")
    
    print("\n" + "="*60)
    print("THERMAL COMFORT SUMMARY")
    print("="*60)
    print(comfort_summary.round(3))
    
    print("\n" + "="*60)
    print("ZONE PERFORMANCE SUMMARY")
    print("="*60)
    print(zone_summary.round(2))
    
    # Generate LaTeX tables
    latex_tables = generate_latex_tables(energy_summary, comfort_summary, zone_summary, 
                                        energy_savings, energy_savings_percent)
    
    # Save LaTeX tables to file
    with open("simulation_results_latex_tables.txt", "w") as f:
        f.write(latex_tables)
    
    print("\n" + "="*60)
    print("LaTeX tables saved to: simulation_results_latex_tables.txt")
    print("="*60)
    
    return energy_summary, comfort_summary, zone_summary, latex_tables

if __name__ == "__main__":
    main()
