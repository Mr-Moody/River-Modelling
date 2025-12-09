# Monte Carlo Implementation Guide

## What It Does
- Treats **transport coefficient (K)** in the Exner equation as uncertain: `K ~ LogNormal(μ, σ)`
- Adds **3D Perlin noise** to initial bed elevation with random seeds
- Runs N realizations to produce ensemble statistics with confidence intervals

---

## Step 1: Add New Files
Copy these to your Unity project:
- `MonteCarloSimulationManager.cs`
- `MonteCarloStateSnapshot.cs`

---

## Step 2: Modify `RiverMeshPhysicsSolver.cs`

**A) Change the transport coefficient field (line ~22):**

```csharp
// REPLACE THIS:
public readonly double transportCoefficient;

// WITH THIS:
private readonly double _baseTransportCoefficient;

public double transportCoefficient
{
    get
    {
        if (MonteCarloParameters.UseMonteCarloTransportCoefficient)
            return MonteCarloParameters.CurrentTransportCoefficient;
        return _baseTransportCoefficient;
    }
}
```

**B) In constructor (line ~58), change:**
```csharp
// REPLACE:
this.transportCoefficient = transportCoefficient;

// WITH:
this._baseTransportCoefficient = transportCoefficient;
```

---

## Step 3: Unity Setup

1. Create empty GameObject named "MonteCarloManager"
2. Add `MonteCarloSimulationManager` component
3. Drag your `SimulationController` to its `simulationController` field
4. Configure parameters in Inspector:
   - `Num Realizations`: 100
   - `Transport Coefficient Mu`: -2.3 (gives mean K ≈ 0.1)
   - `Transport Coefficient Sigma`: 0.3

---

## Step 4: Run

```csharp
// Start ensemble
monteCarloManager.StartEnsembleSimulation();

// Results available via event
monteCarloManager.OnEnsembleCompleted += (ensemble) => {
    Debug.Log($"Mean: {ensemble.EnsembleMeanBedElevation}");
    Debug.Log($"95% CI: [{ensemble.BedElevation95CI_Lower}, {ensemble.BedElevation95CI_Upper}]");
};
```

---

## Output Example
```
Realizations: 100
Ensemble Mean Bed Elevation: -0.0234
95% CI: [-0.0512, 0.0089]
```
