

import numpy as np
import xarray as xr
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from datetime import timedelta
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION — modify these parameters for your use case
# ============================================================

# File paths (Update these to point to your local ERA5 NetCDF files)
# The U file should contain daily zonal wind (u) at 10hPa.
# The Z file should contain daily geopotential (z) at multiple pressure levels.
U_NC_FILE  = "path/to/your/era5.u.10hPa.day.nc"
Z_NC_FILE  = "path/to/your/era5.z.allhPa.day.lev.nc"

# Output HTML file name
OUTPUT_HTML = "3d.ght.anon.animation.ssw.html"

# Search period for Day 0 (the event onset date)
# Day 0 is defined as the first day where zonal wind (U) < 0 at the reference latitude.
DATE_START = "YYYY-MM-DD"
DATE_END   = "YYYY-MM-DD"

# Animation window (days relative to Day 0)
DAYS_BEFORE = 10
DAYS_AFTER  = 20

# Percentile threshold to filter anomalies (0.8 = show only the 20% most extreme values)
PERCENTIL = 0.8

# Reference latitude for zonal wind reversal detection (typically 60°N)
LAT_REF = 60.0

# Event title for the plot
EVENTO = "SSW YYYY-YYYY"


# ============================================================
# PART A: Detect Day 0 (first day with u < 0 at reference lat)
# ============================================================

print("Loading zonal wind file...")
# xarray loads lazily — data is not read into memory until needed
ds_u = xr.open_dataset(U_NC_FILE)

# Detect the name of the wind variable (commonly 'u' or 'uwnd')
u_var = "u" if "u" in ds_u else list(ds_u.data_vars)[0]
u_data = ds_u[u_var]

# Select latitude closest to reference and calculate zonal mean (over longitude)
lat_coord = "latitude" if "latitude" in u_data.dims else "lat"
lon_coord = "longitude" if "longitude" in u_data.dims else "lon"
time_coord = "time" if "time" in u_data.dims else "valid_time"

u_60 = u_data.sel({lat_coord: LAT_REF}, method="nearest").mean(dim=lon_coord)

# Convert to pandas to easily filter the time period
u_series = u_60.to_series()
u_series.index = pd.to_datetime(u_series.index)

mask = (u_series.index >= DATE_START) & (u_series.index <= DATE_END)
u_period = u_series[mask]

# Find the first day with u < 0 for at least 10 consecutive days
day0 = None
for i in range(len(u_period) - 10):
    if (u_period.iloc[i:i+10] < 0).all():
        day0 = u_period.index[i].date()
        break

if day0 is None:
    raise ValueError(f"Could not find 10 consecutive days with u < 0 in the period {DATE_START} - {DATE_END}")

print(f"✓ Day 0 detected (start of 10-day negative wind): {day0}")

ds_u.close()

# Generate sequence of dates for the animation
selected_dates = pd.date_range(
    start=day0 - timedelta(days=DAYS_BEFORE),
    end=day0   + timedelta(days=DAYS_AFTER),
    freq="1D"
)
print(f"✓ Selected dates: {selected_dates[0].date()} → {selected_dates[-1].date()} ({len(selected_dates)} days)")


# ============================================================
# PART B: Load geopotential and calculate anomalies
# ============================================================

print("\nLoading geopotential file (lazy)...")
ds_z = xr.open_dataset(Z_NC_FILE)

# Detect variable and dimension names
z_var   = "z"  if "z"   in ds_z else list(ds_z.data_vars)[0]
lat_dim = "lat"  if "lat"  in ds_z.dims else "latitude"
lon_dim = "lon"  if "lon"  in ds_z.dims else "longitude"
lev_dim = "plev" if "plev" in ds_z.dims else "level"
time_dim = "time" if "time" in ds_z.dims else "valid_time"

z_data = ds_z[z_var]

# Convert pressure to hPa if it is in Pa
levels_raw = z_data[lev_dim].values
if levels_raw.max() > 2000:
    levels_hpa = levels_raw / 100.0
    z_data = z_data.assign_coords({lev_dim: levels_hpa})
else:
    levels_hpa = levels_raw

lats = z_data[lat_dim].values
lons = z_data[lon_dim].values

# Filter for Northern Hemisphere (0-90°N)
z_nh = z_data.sel({lat_dim: slice(90, 0) if lats[0] > lats[-1] else slice(0, 90)})
lats_nh = z_nh[lat_dim].values

print(f"✓ Resolution: {len(lons)} lon × {len(lats_nh)} lat × {len(levels_hpa)} levels")

# Convert geopotential to geopotential height [m]
print("Converting to geopotential height...")
z_gh = z_nh / 9.80665  # [m²/s²] → [m]

# Deseasonalize (calculate climatology)
# xarray handles the broadcasting automatically in a memory-efficient way
print("Calculating anomalies (climatology)...")
z_clim = z_gh.mean(dim=time_dim)          # temporal mean by position
z_anom = z_gh - z_clim                    # anomaly calculation

# Apply logarithmic weighting by pressure level
print("Applying log10(level) weighting...")
log_weights = xr.DataArray(
    np.log10(levels_hpa + 1),
    coords={lev_dim: z_anom[lev_dim]},
    dims=[lev_dim]
)
z_weighted = z_anom * log_weights

# Select only the dates required for the animation
print("Selecting animation dates...")
z_sel = z_weighted.sel({time_dim: selected_dates}, method="nearest")

# Load only the selected dates into RAM (avoids loading the entire dataset)
print("Loading selected data into RAM...")
z_sel = z_sel.load()
ds_z.close()
print(f"✓ Array loaded: {z_sel.shape} — {z_sel.nbytes / 1e6:.1f} MB")

# Calculate global color limits based on the loaded data
print("Calculating global color limits...")
all_values = z_sel.values.ravel()
all_values = all_values[~np.isnan(all_values)]
max_abs = np.ceil(np.percentile(np.abs(all_values), 99) / 500) * 500
global_min = -max_abs
global_max =  max_abs
print(f"✓ Color range: [{global_min:.0f}, {global_max:.0f}] m")

# Pre-calculate polar coordinates for the 3D projection
print("Pre-calculating polar coordinates...")
LON_RAD = np.deg2rad(lons)              # (nlon,)
LAT_NH  = lats_nh                       # (nlat_nh,)
R       = 90.0 - LAT_NH                 # distance to pole (degrees)

# Broadcasting to create a full 2D grid of coordinates
X_POLAR = R[:, None] * np.cos(LON_RAD[None, :])   # (nlat, nlon)
Y_POLAR = R[:, None] * np.sin(LON_RAD[None, :])   # (nlat, nlon)


# ============================================================
# PART C: Map outlines in polar coordinates
# ============================================================

print("\nGenerating coastline outlines...")
try:
    # Try using Cartopy to get accurate coastline shapes
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import MultiLineString
    
    coastlines = cfeature.COASTLINE.geometries()
    x_country, y_country, z_country = [], [], []
    
    for geom in coastlines:
        if hasattr(geom, "geoms"):
            coords_list = [list(g.coords) for g in geom.geoms]
        else:
            coords_list = [list(geom.coords)]
        
        for coords in coords_list:
            coords = np.array(coords)
            lat_c, lon_c = coords[:, 1], coords[:, 0]
            
            # Filter for Northern Hemisphere only
            mask_nh = lat_c >= 0
            if mask_nh.sum() < 2:
                continue
                
            lat_c, lon_c = lat_c[mask_nh], lon_c[mask_nh]
            
            # Convert to polar coordinates
            r_c = 90.0 - lat_c
            x_c = r_c * np.cos(np.deg2rad(lon_c))
            y_c = r_c * np.sin(np.deg2rad(lon_c))
            
            x_country.extend(x_c.tolist() + [None])
            y_country.extend(y_c.tolist() + [None])
            z_country.extend([1200.0] * len(x_c) + [None])

except Exception:
    # Fallback if Cartopy is not available: draw a simple circle at 60°N
    print("  (cartopy not available, using reference circle instead)")
    theta = np.linspace(0, 2 * np.pi, 360)
    r_60 = 90 - 60
    x_country = (r_60 * np.cos(theta)).tolist()
    y_country = (r_60 * np.sin(theta)).tolist()
    z_country = [1200.0] * 360

country_trace = go.Scatter3d(
    x=x_country, y=y_country, z=z_country,
    mode="lines",
    line=dict(color="black", width=1.5),
    name="Coastlines",
    showlegend=False
)

# Colorscale RdBu reversed (negative anomalies = blue, positive = red)
COLORSCALE = [
    [0.0,  "#053061"], [0.1,  "#2166ac"], [0.2,  "#4393c3"],
    [0.3,  "#92c5de"], [0.4,  "#d1e5f0"], [0.5,  "#f7f7f7"],
    [0.6,  "#fddbc7"], [0.7,  "#f4a582"], [0.8,  "#d6604d"],
    [0.9,  "#b2182b"], [1.0,  "#67001f"]
]


# ============================================================
# PART D: Generate animation frames
# ============================================================

print("\nGenerating animation frames...")

def make_frame(date, show_colorbar=False):
    """Generates a 3D scatter trace for a given date."""
    diff_days = (date.date() - day0).days
    diff_text = f"+{diff_days}" if diff_days > 0 else str(diff_days)
    
    # Select timestep — shape: (nlat_nh, nlon, nlevels)
    z_t = z_sel.sel({time_dim: date}, method="nearest")
    
    # Transpose to (nlevels, nlat, nlon) to easily iterate by level
    dims = list(z_t.dims)
    lev_ax  = dims.index(lev_dim)
    lat_ax  = dims.index(lat_dim)
    lon_ax  = dims.index(lon_dim)
    
    z_np = np.transpose(z_t.values, axes=[lev_ax, lat_ax, lon_ax])
    
    # Calculate anomaly relative to the mean of each level (vectorized)
    lev_mean = np.nanmean(z_np, axis=(1, 2), keepdims=True)  # (nlev,1,1)
    z_anom_t = z_np - lev_mean                                 # (nlev,nlat,nlon)
    
    # Calculate threshold percentile per level (vectorized)
    abs_anom = np.abs(z_anom_t)
    thresholds = np.nanpercentile(abs_anom.reshape(len(levels_hpa), -1),
                                   PERCENTIL * 100, axis=1)   # (nlev,)
    
    # Mask to keep only the points exceeding the threshold
    mask = abs_anom >= thresholds[:, None, None]  # (nlev,nlat,nlon)
    
    # Broadcast coordinates to match the 3D shape before masking
    X_all = np.broadcast_to(X_POLAR[None, :, :], z_anom_t.shape)
    Y_all = np.broadcast_to(Y_POLAR[None, :, :], z_anom_t.shape)
    LEV_all = np.broadcast_to(levels_hpa[:, None, None], z_anom_t.shape)
    
    # Apply mask and flatten data for Plotly
    x_pts   = X_all[mask]
    y_pts   = Y_all[mask]
    z_pts   = LEV_all[mask]
    col_pts = z_anom_t[mask]
    
    trace = go.Scatter3d(
        x=x_pts, y=y_pts, z=z_pts,
        mode="markers",
        marker=dict(
            color=col_pts,
            size=8,
            opacity=1,
            cmin=global_min,
            cmax=global_max,
            colorscale=COLORSCALE,
            showscale=show_colorbar,
            colorbar=dict(
                title=dict(text="m", side="right", font=dict(size=12)),
                len=0.5,
                thickness=20,
                x=0.95,
                y=0.50,
                tickvals=list(range(int(global_min), int(global_max)+1, 500)),
                tickfont=dict(size=12)
            )
        ),
        name=str(date.date()),
        showlegend=False
    )
    return trace, diff_text


# Create initial frame
init_trace, init_diff_text = make_frame(selected_dates[0], show_colorbar=True)
print(f"  Frame 1/{len(selected_dates)} ✓", end="\r")

# Generate all frames
frames = []
for i, date in enumerate(selected_dates):
    trace, diff_text = make_frame(date, show_colorbar=True)
    frames.append(go.Frame(
        name=str(date.date()),
        data=[trace, country_trace],
        layout=go.Layout(
            title=dict(text=(
                f"{EVENTO} GHT anomalies (p={1-PERCENTIL:.1f})"
                f"<br><span style='font-size:14px'>Day: {diff_text}</span>"
            ))
        )
    ))
    print(f"  Frame {i+1}/{len(selected_dates)} ✓", end="\r")

print(f"\n✓ {len(frames)} frames generated")


# ============================================================
# PART E: Build final Plotly figure
# ============================================================

print("\nBuilding Plotly figure...")

# Define slider steps
slider_steps = [
    dict(
        label=str(d.date()),
        method="animate",
        args=[[str(d.date())], dict(
            mode="immediate",
            frame=dict(duration=500, redraw=True),
            transition=dict(duration=200)
        )]
    )
    for d in selected_dates
]

fig = go.Figure(
    data=[init_trace, country_trace],
    frames=frames,
    layout=go.Layout(
        title=dict(
            text=(f"{EVENTO} — GHT anomalies (p={1-PERCENTIL:.1f})"
                  f"<br><span style='font-size:14px'>Day: {init_diff_text}</span>"),
            font=dict(size=16, family="Georgia, serif", color="#1a1a2e"),
            x=0.5, xanchor="center", y=0.97, yanchor="top"
        ),
        scene=dict(
            xaxis=dict(title="", showgrid=True, gridcolor="rgba(200,200,200,0.3)",
                       zeroline=False, showticklabels=False,
                       showbackground=False, dtick=30),
            yaxis=dict(title="", showgrid=True, gridcolor="rgba(200,200,200,0.3)",
                       zeroline=False, showticklabels=False,
                       showbackground=False, dtick=30),
            zaxis=dict(title="Pressure (hPa)", autorange="reversed",
                       type="log", showgrid=True,
                       gridcolor="rgba(200,200,200,0.3)",
                       showbackground=False,
                       title_font=dict(size=11)),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            bgcolor="rgba(0,0,0,0)",
            domain=dict(x=[0.0, 0.95], y=[0.08, 1.0])
        ),
        paper_bgcolor="white",
        plot_bgcolor="rgba(0,0,0,0)",
        updatemenus=[dict(
            type="buttons",
            direction="left",
            showactive=True,
            x=0.05, y=0.02,
            xanchor="left", yanchor="bottom",
            pad=dict(r=10, t=0),
            font=dict(size=11),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=600, redraw=True),
                        transition=dict(duration=250),
                        fromcurrent=True,
                        mode="immediate"
                    )]
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate"
                    )]
                )
            ]
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(
                prefix="", font=dict(size=12, color="#333"),
                xanchor="center"
            ),
            pad=dict(t=15, b=20),
            x=0.08, len=0.85,
            xanchor="left",
            y=0.02, yanchor="top",
            font=dict(size=12),
            steps=slider_steps
        )],
        margin=dict(l=0, r=0, t=60, b=50)
    )
)


# ============================================================
# PART F: Save to optimized HTML
# ============================================================

print(f"\nSaving HTML to: {OUTPUT_HTML}")

# JavaScript to ensure the plot fits the full viewport when opened
RESIZE_SCRIPT = """
(function() {
  var gd = document.getElementById('{plot_id}');
  var container = gd.parentElement;

  // CSS for fullscreen
  document.body.style.margin = '0';
  document.body.style.padding = '0';
  document.body.style.overflow = 'hidden';
  document.body.style.background = 'white';
  container.style.width = '100vw';
  container.style.height = '100vh';

  function resize() {
    Plotly.relayout(gd, { width: window.innerWidth, height: window.innerHeight });
  }

  window.addEventListener('resize', resize);
  setTimeout(resize, 100);
})();
"""

pio.write_html(
    fig,
    file=OUTPUT_HTML,
    include_plotlyjs=True,
    full_html=True,
    auto_open=False,
    auto_play=False,
    post_script=RESIZE_SCRIPT,
    config={"responsive": True}
)

import os
size_mb = os.path.getsize(OUTPUT_HTML) / 1e6
print(f"\n{'='*50}")
print(f"✓ HTML saved: {size_mb:.1f} MB")
print(f"✓ Day 0: {day0}")
print(f"✓ Frames: {len(frames)}")
print(f"✓ Color range: [{global_min:.0f}, {global_max:.0f}] m")
print(f"{'='*50}")
