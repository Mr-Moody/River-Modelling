using UnityEngine;
using System;

/// <summary>
/// Stores a snapshot of the solver state for Monte Carlo analysis.
/// </summary>
[System.Serializable]
public class MonteCarloStateSnapshot
{
    public double[,] BedElevation { get; private set; }
    public double[,] VelocityU { get; private set; }
    public double[,] VelocityV { get; private set; }
    public double[,] WaterDepth { get; private set; }
    public int[,] CellType { get; private set; }
    
    public int NumCrossSections { get; private set; }
    public int WidthResolution { get; private set; }
    
    // Summary statistics
    public double MeanBedElevation { get; private set; }
    public double StdBedElevation { get; private set; }
    public double MaxVelocity { get; private set; }
    public double MeanWaterDepth { get; private set; }
    
    public MonteCarloStateSnapshot(RiverMeshPhysicsSolver solver)
    {
        NumCrossSections = solver.numCrossSections;
        WidthResolution = solver.widthResolution;
        
        // Deep copy arrays
        BedElevation = (double[,])solver.h.Clone();
        VelocityU = (double[,])solver.u.Clone();
        VelocityV = (double[,])solver.v.Clone();
        WaterDepth = (double[,])solver.waterDepth.Clone();
        CellType = (int[,])solver.cellType.Clone();
        
        // Compute summary statistics
        ComputeStatistics(solver);
    }
    
    private void ComputeStatistics(RiverMeshPhysicsSolver solver)
    {
        double sumH = 0, sumH2 = 0, sumD = 0;
        double maxV = 0;
        int count = 0;
        
        for (int i = 0; i < NumCrossSections; i++)
        {
            for (int w = 0; w < WidthResolution; w++)
            {
                if (solver.cellType[i, w] == RiverCellType.FLUID)
                {
                    sumH += solver.h[i, w];
                    sumH2 += solver.h[i, w] * solver.h[i, w];
                    sumD += solver.waterDepth[i, w];
                    
                    double vel = solver.GetVelocityMagnitude(i, w);
                    if (vel > maxV) maxV = vel;
                    
                    count++;
                }
            }
        }
        
        if (count > 0)
        {
            MeanBedElevation = sumH / count;
            StdBedElevation = Math.Sqrt(sumH2 / count - MeanBedElevation * MeanBedElevation);
            MeanWaterDepth = sumD / count;
            MaxVelocity = maxV;
        }
    }
}
