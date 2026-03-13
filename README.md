# 3D Polar Vortex Visualization

This repository contains the Python scripts and generated interactive Plotly figures to visualize the 3D structure of Sudden Stratospheric Warming (SSW) events.

**Background:**  
These figures were originally generated as part of my BSc thesis: *Impact of the Stratospheric Polar Vortex on Temperature and Precipitation in the Northern
Hemisphere Climate*. Supervisor: [Cristina Peña-Ortiz](https://orcid.org/0000-0002-5451-8521).

## How it works

The included script (`3d_SSWvortex_plev_ght_anom_1yr.py`) processes ERA5 daily geopotential data to construct a 3D animated timeline of the Geopotential Height Anomalies. The event onset (Day 0) is classified automatically by identifying at least 10 consecutive days where the zonal wind at 10hPa (60°N) is negative (`u < 0`).

## Viewing the Events

You can view the interactive timeline directly via GitHub Pages:  
[https://oscarmtr.github.io/SSWs-vortex-plot/](https://oscarmtr.github.io/SSWs-vortex-plot/)

**Data**  
Plotted events were calculated using the geopotential average from 2000 to 2024.

## License

Content is licensed under CC BY-NC-SA 4.0. The source code is licensed under GNU AGPLv3.
