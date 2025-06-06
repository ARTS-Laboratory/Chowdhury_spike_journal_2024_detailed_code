# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:35:34 2025
@author: chypuja
"""
import pandas as pd

# Input and output paths
input_file = "data/9_full_sensor_spike_data_processed.csv"
output_file = "data/9_full_sensor_spike_data_with_resistance.csv"

# Load header separately
header_rows = pd.read_csv(input_file, nrows=1, header=None)
column_names = pd.read_csv(input_file, skiprows=1, nrows=1).columns

# Load the actual data
D1 = pd.read_csv(input_file, skiprows=1)

# Extract TDS (Voltage) columns — every 4th column starting from index 4
C_cols = [D1.iloc[:, i] for i in range(4, 37, 4)]

# Voltage to resistance conversion
def voltage_to_resistance(V):
    I_mA = (V / 2.3) * 3 + 3   # current in mA
    I_A = I_mA / 1000          # convert to A
    R = V / I_A                # resistance in ohms
    return R

# Compute resistance and add to dataframe
for i, C in enumerate(C_cols, start=1):
    R = voltage_to_resistance(C)
    D1[f'R{i}'] = R

# Combine header rows and data
# Write the first two header rows manually before data
with open(output_file, 'w', newline='') as f:
    # Write both header rows
    header_rows.to_csv(f, index=False, header=False)
    # Write data with column names
    D1.to_csv(f, index=False)
