using UnityEngine;
using System;
using System.Collections.Generic;

/// <summary>
/// Diagnostic tool to check if bank migration is working and provide recommendations.
/// Attach this to any GameObject and call CheckBankMigration() from the Inspector context menu.
/// </summary>
public class BankMigrationDiagnostics : MonoBehaviour
{
    [Header("References")]
    public SimulationController simulationController;
    
    [ContextMenu("Check Bank Migration Status")]
    public void CheckBankMigrationStatus()
    {
        if (simulationController == null)
        {
            simulationController = FindFirstObjectByType<SimulationController>();
            if (simulationController == null)
            {
                Debug.LogError("[BankMigrationDiagnostics] SimulationController not found!");
                return;
            }
        }
        
        RiverMeshPhysicsSolver solver = simulationController.GetSolver();
        if (solver == null)
        {
            Debug.LogError("[BankMigrationDiagnostics] Solver not initialized! Start the simulation first.");
            return;
        }
        
        Debug.Log("═══════════════════════════════════════════════════════════");
        Debug.Log("[BankMigrationDiagnostics] BANK MIGRATION DIAGNOSTICS");
        Debug.Log("═══════════════════════════════════════════════════════════");
        
        // Check configuration
        Debug.Log($"\n[Configuration]");
        Debug.Log($"  EnableHorizontalBankMigration: {simulationController.EnableHorizontalBankMigration}");
        Debug.Log($"  UpdateRiverGeometry: {simulationController.UpdateRiverGeometry}");
        Debug.Log($"  Bank Migration Threshold: {simulationController.BankMigrationThreshold}");
        Debug.Log($"  Bank Erosion Rate: {simulationController.BankErosionRate}");
        Debug.Log($"  Bank Critical Shear: {simulationController.BankCriticalShear}");
        Debug.Log($"  Transport Coefficient: {simulationController.TransportCoefficient}");
        Debug.Log($"  Time Step: {simulationController.TimeStep}");
        
        // Check if horizontal migration is enabled
        if (!simulationController.EnableHorizontalBankMigration)
        {
            Debug.LogWarning("\n⚠️  WARNING: EnableHorizontalBankMigration is DISABLED!");
            Debug.LogWarning("   → Enable this checkbox to see horizontal bank migration.");
        }
        
        // Analyze bank erosion
        double[,] tau = solver.ComputeShearStress();
        double[,] bankErosion = solver.ComputeBankErosion(tau);
        
        int numCrossSections = solver.numCrossSections;
        int widthResolution = solver.widthResolution;
        
        double maxBankErosion = 0.0;
        double maxCumulativeErosion = 0.0;
        double totalCumulativeErosion = 0.0;
        int bankEdgeCells = 0;
        int migratingCells = 0;
        int aboveThresholdCells = 0;
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                if (solver.cellType[i, w] != RiverCellType.FLUID)
                {
                    // Check if bank edge
                    bool isEdge = (w > 0 && solver.cellType[i, w - 1] == RiverCellType.FLUID) ||
                                  (w < widthResolution - 1 && solver.cellType[i, w + 1] == RiverCellType.FLUID);
                    
                    if (isEdge)
                    {
                        bankEdgeCells++;
                        double erosion = bankErosion[i, w];
                        double cumulative = solver.GetCumulativeBankErosion(i, w);
                        
                        if (erosion > maxBankErosion) maxBankErosion = erosion;
                        if (cumulative > maxCumulativeErosion) maxCumulativeErosion = cumulative;
                        totalCumulativeErosion += cumulative;
                        
                        if (solver.IsBankMigrating(i, w)) migratingCells++;
                        if (cumulative >= solver.bankMigrationThreshold) aboveThresholdCells++;
                    }
                }
            }
        }
        
        Debug.Log($"\n[Bank Erosion Analysis]");
        Debug.Log($"  Bank Edge Cells: {bankEdgeCells}");
        Debug.Log($"  Max Bank Erosion Rate: {maxBankErosion:E6} m/s");
        Debug.Log($"  Max Cumulative Erosion: {maxCumulativeErosion:F6} m");
        Debug.Log($"  Average Cumulative Erosion: {(bankEdgeCells > 0 ? totalCumulativeErosion / bankEdgeCells : 0):F6} m");
        Debug.Log($"  Cells Above 50% Threshold: {migratingCells}");
        Debug.Log($"  Cells Above Full Threshold: {aboveThresholdCells}");
        Debug.Log($"  Migration Threshold: {solver.bankMigrationThreshold:F6} m");
        
        // Calculate time to reach threshold
        if (maxBankErosion > 0 && solver.bankMigrationThreshold > 0)
        {
            double timeToThreshold = solver.bankMigrationThreshold / maxBankErosion;
            double stepsToThreshold = timeToThreshold / simulationController.TimeStep;
            Debug.Log($"\n[Time Estimates]");
            Debug.Log($"  Time to reach threshold (at max rate): {timeToThreshold:F2} seconds");
            Debug.Log($"  Steps to reach threshold: {stepsToThreshold:F0} steps");
            Debug.Log($"  At {simulationController.IterationsPerFrame} iter/frame: {stepsToThreshold / simulationController.IterationsPerFrame:F0} frames");
        }
        
        // Check shear stress
        double maxTau = 0.0;
        double maxTauExcess = 0.0;
        int cellsWithExcessTau = 0;
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                if (solver.cellType[i, w] != RiverCellType.FLUID)
                {
                    double tauValue = tau[i, w];
                    double tauExcess = Math.Max(tauValue - simulationController.BankCriticalShear, 0.0);
                    
                    if (tauValue > maxTau) maxTau = tauValue;
                    if (tauExcess > maxTauExcess) maxTauExcess = tauExcess;
                    if (tauExcess > 0) cellsWithExcessTau++;
                }
            }
        }
        
        Debug.Log($"\n[Shear Stress Analysis]");
        Debug.Log($"  Max Shear Stress: {maxTau:F6} Pa");
        Debug.Log($"  Max Excess Shear (tau - critical): {maxTauExcess:F6} Pa");
        Debug.Log($"  Bank Critical Shear: {simulationController.BankCriticalShear:F6} Pa");
        Debug.Log($"  Cells with Excess Shear: {cellsWithExcessTau}");
        
        if (maxTauExcess == 0)
        {
            Debug.LogWarning("\n⚠️  WARNING: No excess shear stress detected!");
            Debug.LogWarning("   → Bank erosion requires: tau > BankCriticalShear");
            Debug.LogWarning("   → Try reducing BankCriticalShear or increasing flow velocity");
        }
        
        // Recommendations
        Debug.Log($"\n[Recommendations]");
        
        List<string> recommendations = new List<string>();
        
        if (!simulationController.EnableHorizontalBankMigration)
        {
            recommendations.Add("✓ Enable 'EnableHorizontalBankMigration' checkbox");
        }
        
        if (maxTauExcess == 0)
        {
            recommendations.Add("✓ Reduce 'BankCriticalShear' (try 0.001 or lower)");
            recommendations.Add("✓ Increase initial water velocity or depth");
        }
        
        if (maxCumulativeErosion < solver.bankMigrationThreshold * 0.1)
        {
            recommendations.Add("✓ Reduce 'BankMigrationThreshold' (try 0.0001 or lower)");
            recommendations.Add("✓ Increase 'BankErosionRate' (try 200-500)");
            recommendations.Add("✓ Increase 'TransportCoefficient' (try 0.5-1.0)");
        }
        
        if (simulationController.TimeStep < 0.05f)
        {
            recommendations.Add("✓ Consider increasing 'TimeStep' slightly (0.05-0.1) for faster visible changes");
        }
        
        if (recommendations.Count == 0)
        {
            Debug.Log("  ✓ Configuration looks good! Bank migration should be working.");
            Debug.Log("  → If still not seeing movement, try running simulation longer.");
        }
        else
        {
            foreach (string rec in recommendations)
            {
                Debug.Log($"  {rec}");
            }
        }
        
        Debug.Log("\n═══════════════════════════════════════════════════════════");
    }
    
    [ContextMenu("Print Current Bank Positions")]
    public void PrintCurrentBankPositions()
    {
        if (simulationController == null)
        {
            simulationController = FindFirstObjectByType<SimulationController>();
        }
        
        RiverMeshPhysicsSolver solver = simulationController?.GetSolver();
        if (solver == null)
        {
            Debug.LogError("[BankMigrationDiagnostics] Solver not available!");
            return;
        }
        
        Debug.Log("[BankMigrationDiagnostics] Current Bank Edge Positions:");
        for (int i = 0; i < Mathf.Min(10, solver.numCrossSections); i++) // Print first 10
        {
            var (left, right) = solver.GetBankEdges(i);
            if (left >= 0 && right >= 0)
            {
                double leftCumulative = solver.GetCumulativeBankErosion(i, left);
                double rightCumulative = solver.GetCumulativeBankErosion(i, right);
                Debug.Log($"  CrossSection {i}: Left={left} (cum={leftCumulative:F6}), Right={right} (cum={rightCumulative:F6})");
            }
        }
    }
}
