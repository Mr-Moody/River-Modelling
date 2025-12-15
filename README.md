# Rivers, Meanders, and Oxbow Lakes

![River Evolution Simulation Result](Figures/30_years_sped_up.gif)

Modelling the change in topology of rivers over time due to bank erosion and sediment transport.

**Authors:** Thomas Moody, Tara Kasayapanand, Andrew Lau

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Unity Simulation](#unity-simulation)
   - [Installation](#installation)
   - [Setup](#setup)
   - [Usage](#usage)
   - [Exporting Results](#exporting-results)
4. [Python Analysis](#python-analysis)
   - [Setup](#python-setup)
   - [Analysing Results](#analysing-results)

---

## Project Overview

This project models and simulates river morphodynamics, how rivers change their shape, width, and bed elevation over time due to water flow and sediment transport. The simulation is implemented in Unity for real-time visualisation and physics-based modeling, with results exported to CSV format for detailed analysis in Python.

The simulation models:
- Water flow dynamics using shallow water equations.
- Sediment transport and bed erosion/deposition.
- Bank migration and erosion.
- Long-term river evolution over days, months, or years.


---

## Project Structure

```
River-Modelling/
├── River-Unity/                 # Unity simulation project
│   ├── Assets/
│   │   ├── Scripts/             # Core simulation scripts
│   │   │   ├── RiverMeshPhysicsSolver.cs
│   │   │   ├── LongTermSimulationController.cs
│   │   │   ├── MonteCarloSimulationManager.cs
│   │   │   └── ...
│   │   ├── Scenes/              # Unity scenes
│   │   └── Resources/           # Initial geometry data
│   └── Results/                 # Exported simulation results
│       ├── summary_statistics.csv
│       └── Results_YYYYMMDD_HHMMSS/
│
├── csv_files/                    # Input geometry data
│   ├── Jurua_19871012.csv
│   └── Jurua_20170710.csv
│
├── Objects/                      # Python mesh objects
│   ├── MeshGrid.py
│   └── MeshPoint.py
│
├── data_extraction.ipynb         # Extract geometry from CSV
├── monte_carlo_analysis.ipynb    # Monte Carlo analysis
├── prediction_analysis.ipynb     # Prediction comparison
├── cline_analysis.py             # Centerline analysis module
├── visualisation.py              # 3D visualisation
├── simulation.py                 # Python simulation framework
├── physics.py                    # Physics solver
└── requirements.txt              # Python dependencies
```


---

## Unity Simulation

### Installation

#### Requirements
- **Unity 6000.1.15f1** (or more recent stable version).

#### Steps

1. **Install Unity Hub**
   - Download from [unity.com](https://unity.com/download).
   - Install Unity Editor version 6000.1.15f1.

2. **Open the Project**
   - Open Unity Hub.
   - Click "Add" and select the `River-Unity` folder.
   - Click "Open" to load the project in Unity Editor.

3. **Verify Installation**
   - Unity should open with the project.
   - Check the Console for any import errors.
   - Navigate to `Assets/Scenes/SampleScene.unity` to see the main scene.

### Setup

#### Initial Configuration

1. **Open the Main Scene**
   - In Unity, open `Assets/Scenes/SampleScene.unity`.

2. **Configure the Simulation Controller**
   - In the Hierarchy, locate the GameObject with the `LongTermSimulationController` component.
      > Or find the `MonteCarloSimulationManager` for Monte Carlo simulations.
   - Check the Inspector panel for configuration options.

3. **Set Export Path**
   - By default, results export to `E:\UCL\River-Modelling\River-Unity\Results`.
   - You can change this in the script's `exportBasePath` field.

4. **Load Initial Geometry**
   - Initial river geometry can be loaded from CSV files.
   - Place CSV files in `Assets/Resources/` folder.
   - Use the `CSVGeometryLoader` component to load geometry.

#### Key Configuration Parameters

**Simulation Parameters:**
- `NumCrossSections`: Number of cross-sections along the river.
- `WidthResolution`: Resolution of the width dimension.
- `TimeStep`: Time step size (in days for long-term simulations).
- `TotalDays`: Total simulation duration.

**Physics Parameters:**
- `TransportCoefficient`: Sediment transport coefficient.
- `CriticalShear`: Critical shear stress for sediment motion.
- `BankErosionRate`: Rate of bank erosion.
- `BankCriticalShear`: Critical shear for bank erosion.

**Export Settings:**
- `autoExportOnComplete`: Automatically export results when simulation finishes.
- `exportFileName`: Base filename for exported CSV files (timestamp appended automatically).

### Usage

#### Running a Single Simulation

1. **Start the Simulation**
   - Press Play in Unity Editor.
   - The simulation will run in real-time.

2. **Monitor Progress**
   - Watch the 3D visualisation of the river evolution.
   - Check the Console for log messages.
   - UI displays current simulation time and progress.
   - Can disable rendering in UI to reduce simulation time.

3. **Stop**
   - Click "Stop" button in the UI.
   - The simulation will export results automatically if `autoExportOnComplete` is enabled.
       > Stopping a simulation run before time length is completed will still output final river topology.

#### Running Monte Carlo Simulations

1. **Configure Monte Carlo Parameters**
   - Select the GameObject with `MonteCarloSimulationManager`.
   - Set number of realisations.
   - Configure parameter ranges for uncertainty analysis.
   - Enable "Store Full Ensemble" if you need detailed riveer data for each stage.

2. **Run Ensemble**
   - Press Play.
   - The simulation will run multiple realisations with varying parameters.
   - Progress is shown in the UI.

3. **Results**
   - Results are automatically exported to timestamped folders.
   - Summary statistics are saved to `summary_statistics.csv`.
   - Full field data (if enabled) is saved for each realisation.

#### Visual Controls

- **Camera**: Use the `CameraController` to navigate the 3D view (WASD - horizontal, Space/Shift - vertical, C - lock/unlock cursor).
- **Heatmaps**: Toggle velocity and erosion heatmaps using shaders.

### Exporting Results

#### Automatic Export

Results are automatically exported when:
- Simulation completes successfully.
- Simulation is manually stopped (if configured).
- Each Monte Carlo realisation finishes (for ensemble runs).

#### Export Format

Results are saved as CSV files with the following naming convention:
- Single simulations: `RiverSimulation_YYYY_DDMMYYYY_HHMMSS.csv`.
- Monte Carlo: `Results_YYYYMMDD_HHMMSS/` folder containing:
  - `summary_statistics.csv` - Aggregated statistics for all realisations.
  - `stage_N_bed_elevation.csv` - Bed elevation for realisation N.
  - `stage_N_velocity_u.csv` - U-component of velocity.
  - `stage_N_velocity_v.csv` - V-component of velocity.
  - `stage_N_water_depth.csv` - Water depth field.
  - `stage_N_cell_type.csv` - Cell type classification.

#### CSV File Structure

River geometry CSV files follow this format:
```
cline_x,cline_y,cross_section_index,left_bank_x,left_bank_y,right_bank_x,right_bank_y
```

Monte Carlo field data CSVs are 2D grids with:
- Rows: Cross-section indices.
- Columns: Width resolution indices.
- Values: Physical quantities (elevation, velocity, depth, etc.).

---

## Python Analysis

### Python Setup

#### Requirements

Install required Python packages:

```bash
pip install numpy matplotlib pyvista>=0.40.0 pandas seaborn scipy
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Note:** `cline_analysis.py` must be in the same directory as the analysis notebooks.

#### Project Structure

The Python analysis scripts are located in the root directory:
- `data_extraction.ipynb` - Extract and process river geometry from CSV files.
- `monte_carlo_analysis.ipynb` - Analyze Monte Carlo simulation results.
- `prediction_analysis.ipynb` - Compare predicted vs. actual river geometry.
- `cline_analysis.py` - Centerline analysis.

### Analysing Results

#### 1. Extracting Geometry Data

Open `data_extraction.ipynb`:

```python
# Load CSV files
fnames = ['Jurua_19871012.csv', 'Jurua_20170710.csv']
csv_dir = './csv_files/'

# Extract geometry data
# This will parse centerline, bank positions, curvature, width, etc.
```

**Outputs:**
- Centerline coordinates.
- Left and right bank positions.
- River width.
- Curvature measurements.
- Migration indices.

#### 2. Analysing Monte Carlo Results

Open `monte_carlo_analysis.ipynb`:

```python
# Load summary statistics
df = pd.read_csv('River-Unity/Results/summary_statistics.csv')

# Analyse transport coefficient vs. erosion rate
transport_coeff = df["transport_coefficient_K"]
max_erosion_rate = df["max_erosion_rate"]

# Create correlation plots
plt.scatter(transport_coeff, max_erosion_rate)
```

**Key Analyses:**
- Parameter sensitivity analysis.
- Correlation between physical parameters.
- Statistical distributions of outcomes.
- Uncertainty quantification.

#### 3. Prediction vs. Actual Comparison

Open `prediction_analysis.ipynb`:

```python
# Load actual and predicted geometry
df_actual = pd.read_csv('csv_files/Jurua_20170710.csv')
df_predicted = pd.read_csv('River-Unity/Results/RiverSimulation_2017_YYYYMMDD_HHMMSS.csv')
df_initial = pd.read_csv('csv_files/Jurua_19871012.csv')

# Compare geometries
# Calculate metrics: width change, migration distance, etc.
```

**Metrics:**
- Width change accuracy.
- Migration distance error.
- Centerline position comparison.
- Bank position errors.

#### 4. Visualising Results

Use `visualisation.py` for 3D visualization:

```python
from visualisation import visualiseSimulation

# Visualise a single frame
visualiseSimulation(filename="simulation_frames.pkl", frame_index=0)

# Animate through all frames
visualiseSimulation(filename="simulation_frames.pkl", animate=True)
```

**Features:**
- 3D mesh visualisation (Matplotlib 3D or GPU-accelerated with PyVista).
- Velocity field visualisation.
- Erosion/deposition heatmaps.
- Animated evolution through time.

#### Example Analysis Workflow

1. **Load Simulation Results**
   ```python
   import pandas as pd
   results = pd.read_csv('River-Unity/Results/RiverSimulation_2017_20251214_205254.csv')
   ```

2. **Compare with Observed Data**
   ```python
   observed = pd.read_csv('csv_files/Jurua_20170710.csv')
   # Calculate differences
   ```

3. **Generate Visualisations**
   ```python
   import matplotlib.pyplot as plt
   # Plot width changes, migration, etc.
   ```

4. **Statistical Analysis**
   ```python
   from scipy import stats
   # Perform correlation analysis, hypothesis testing, etc.
   ```

---

## License

This project is licensed under the **MIT License** - see the separate `LICENSE` file for details.

**Copyright** © 2025 Thomas Moody, Tara Kasayapanand, and Andrew Lau.

---
