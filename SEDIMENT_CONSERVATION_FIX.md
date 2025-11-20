# Sediment Conservation and Peak Growth Fix

## Problems Identified

### 1. **No Sediment Conservation**
- The Exner equation was using `np.gradient` which doesn't properly handle boundary conditions
- Boundaries were not enforcing zero-flux conditions, allowing sediment to be created/destroyed
- This caused unbounded growth of peaks

### 2. **Missing Bed Slope Effects**
- Sediment transport didn't account for bed slope
- Sediment should transport easier downhill than uphill (gravity effect)
- Without slope effects, peaks could grow without resistance

### 3. **Numerical Instability**
- High transport coefficient (0.1) combined with no slope effects caused instability
- Hard clipping bounds (-10 to 10m) trapped sediment when limits were reached
- No slope-dependent transport reduction on steep slopes

### 4. **Boundary Condition Issues**
- `np.gradient` doesn't respect zero-flux boundary conditions
- Flux at boundaries wasn't explicitly set to zero
- This violated sediment conservation

## Solutions Implemented

### 1. **Zero-Flux Boundary Conditions**
- Explicitly set sediment flux to zero at all boundaries
- Use finite differences with proper boundary handling
- Ensures sediment cannot cross boundaries (closed system)
- **Result**: Sediment is now conserved (total volume should remain constant)

### 2. **Bed Slope Effects in Sediment Transport**
- Added slope-dependent transport reduction
- Steep slopes reduce transport (prevents instability)
- Sediment flux direction now includes slope component (30% slope, 70% flow)
- **Result**: Peaks are eroded by gravity, preventing unbounded growth

### 3. **Reduced Transport Coefficient and Stability Limits**
- Reduced maximum change rate from 0.1 to 0.02 m/s
- Added slope factor that reduces transport on steep slopes
- Soft bounds instead of hard clipping
- **Result**: More stable evolution, prevents numerical blow-up

### 4. **Improved Boundary Handling**
- Replaced `np.gradient` with explicit finite differences
- Proper one-sided differences at boundaries
- Zero flux explicitly enforced at corners
- **Result**: Accurate divergence calculation that conserves sediment

## Key Changes

### `sediment_flux()` function
- Now accepts bed elevation `h` parameter
- Computes bed slope and applies slope factor
- Reduces transport on steep slopes to prevent instability

### `compute_sediment_flux_vector()` function
- Now includes slope effects in flux direction
- Combines flow direction (70%) with slope direction (30%)
- Sediment preferentially moves downhill

### `exner_equation()` function
- Explicit zero-flux boundary conditions
- Finite differences instead of `np.gradient`
- Proper boundary handling for sediment conservation
- Reduced change rate limits for stability

## Sediment Conservation

**Is there a fixed amount of sediment?**
- **YES** - With the fixes, sediment is now conserved
- The total sediment volume (sum of bed elevations × area) should remain constant
- Zero-flux boundaries ensure no sediment enters or leaves the domain
- Use `diagnose_sediment.py` to verify conservation

## Expected Behavior After Fix

1. **Peaks should stop growing unboundedly**
   - Slope effects erode peaks
   - Steep slopes have reduced transport
   - Gravity pulls sediment downhill

2. **Sediment is conserved**
   - Total sediment volume remains constant
   - No creation or destruction at boundaries
   - Redistribution only (erosion in some areas, deposition in others)

3. **More stable evolution**
   - Reduced change rates prevent numerical instability
   - Slope-dependent transport prevents blow-up
   - Smooth, physical bed evolution

## Testing

Run the diagnostic tool to check sediment conservation:
```python
python diagnose_sediment.py
```

This will:
- Calculate total sediment volume for each frame
- Show if sediment is being created/destroyed
- Report bed elevation statistics
- Warn if hitting bounds

## Recommended Parameters

For stable simulations:
- `transport_coefficient = 0.01-0.05` (reduced from 0.1)
- `dt = 0.001-0.01` (smaller time steps for stability)
- `max_change_rate = 0.02` (reduced from 0.1)
- Monitor sediment conservation using diagnostic tool

## Next Steps

1. Run simulation with new parameters
2. Check sediment conservation using `diagnose_sediment.py`
3. Verify peaks no longer grow unboundedly
4. Adjust parameters if needed for desired behavior

