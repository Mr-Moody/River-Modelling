# River Bank Implementation Design

## Recommended Architecture

### 1. Physics Layer (Single Grid)
- **Keep single grid** for physics calculations
- Add a **cell type mask** to distinguish:
  - `FLUID` cells (river channel)
  - `BANK_LEFT` cells (left bank)
  - `BANK_RIGHT` cells (right bank)
  - `BED` cells (river bed - currently all cells)

### 2. Visualization Layer (Separate Meshes)
- **Bed mesh**: Current MeshGrid for river bed
- **Left bank mesh**: Separate MeshGrid for left bank
- **Right bank mesh**: Separate MeshGrid for right bank
- These are **visualization-only** - physics runs on single grid

## Implementation Strategy

### Option A: Extend Current Grid (Recommended)
1. Add bank cells to the physics grid (extend width)
2. Mark bank cells as non-fluid in boundary conditions
3. Create separate visualization meshes for banks
4. Apply no-slip boundary conditions at bank-fluid interface

### Option B: Keep Grid Size, Use Mask
1. Keep current grid size
2. Use a mask to mark bank regions within the grid
3. Apply boundary conditions at masked cells
4. Create separate visualization meshes

## Key Considerations

1. **Boundary Conditions**: Banks should have no-slip walls (u=0, v=0)
2. **Sediment Transport**: No sediment transport in bank cells
3. **Bed Evolution**: Banks can erode but at different rates than bed
4. **Visualization**: Banks need separate meshes for proper 3D rendering

## Benefits of This Approach

- ✅ Minimal changes to existing physics solver
- ✅ Maintains grid-based efficiency
- ✅ Clean separation of physics and visualization
- ✅ Easy to adjust bank geometry
- ✅ Can add bank erosion later

