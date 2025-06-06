# Chowdhury_spike_journal_2024
# Code for paper "Wireless Sensor Network for Distributed Real-Time Soil Saturation Monitoring in Levees"
This folder contains code, data, and plots for kriging and clustering for different axies as like XY, XZ, and YZ.

* **data**: This folder has raw/original data, processed data, and combined data:
  * processed data: `9_full_sensor_spike_data_processed.csv`
  * original/raw data:`9_full_sensor_spike_original.xlsx`
  * new data after adding resistance values: `9_full_sensor_spike_data_with_resistance.csv`


* **plots**: This folder saves all plots after running the code.

* **plot_2D_9spike_resistance_data_all.py**: This code is used for 2D plotting for all parameter in one plot.
* **processed_resistance.py**: This code is used to process the voltage to resistance and save the new data to the original CSV file again.
* **krig_clus_resistance_conductivity_XY.py**: This is a 2D kriging and clustering code for the XY plane.
* **krig_clus_resistance_conductivity_XZ.py**: This is a 2D kriging and clustering code for the XZ plane.
* **krig_clus_resistance_conductivity_YZ.py**: This is a 2D kriging and clustering code for the YZ plane.

