using UnityEngine;
using System;
using System.IO;
using System.Text;
using System.Globalization;

/// <summary>
/// Utility class for exporting Monte Carlo simulation results to CSV files.
/// </summary>
public static class MonteCarloCSVExporter
{
    /// <summary>
    /// Creates a results folder with timestamp in the specified base path.
    /// </summary>
    /// <param name="basePath">Base path for results folder. If empty, defaults to E:\UCL\River-Modelling\River-Unity\Results.</param>
    /// <returns>Full path to the created results folder.</returns>
    public static string CreateResultsFolder(string basePath = "")
    {
        try
        {
            if (string.IsNullOrEmpty(basePath))
            {
                // Default to E:\UCL\River-Modelling\River-Unity\Results
                basePath = @"E:\UCL\River-Modelling\River-Unity\Results";
            }
            
            // Ensure base directory exists
            if (!Directory.Exists(basePath))
            {
                Directory.CreateDirectory(basePath);
                Debug.Log($"[MonteCarloCSVExporter] Created base results directory: {basePath}");
            }
            
            // Create folder name with timestamp
            DateTime now = DateTime.Now;
            string folderName = $"Results_{now:yyyyMMdd_HHmmss}";
            string fullPath = Path.Combine(basePath, folderName);
            
            // Create directory if it doesn't exist
            if (!Directory.Exists(fullPath))
            {
                Directory.CreateDirectory(fullPath);
                Debug.Log($"[MonteCarloCSVExporter] Created results folder: {fullPath}");
            }
            else
            {
                Debug.LogWarning($"[MonteCarloCSVExporter] Results folder already exists: {fullPath}");
            }
            
            return fullPath;
        }
        catch (Exception e)
        {
            Debug.LogError($"[MonteCarloCSVExporter] Failed to create results folder: {e.Message}");
            return string.Empty;
        }
    }
    
    /// <summary>
    /// Exports full field data for a single Monte Carlo realization to CSV files.
    /// </summary>
    /// <param name="result">The realization result containing the data.</param>
    /// <param name="folderPath">Path to the results folder.</param>
    /// <param name="stageNumber">Stage number (1-indexed) for file naming.</param>
    /// <returns>True if export succeeded, false otherwise.</returns>
    public static bool ExportFullFieldData(MonteCarloRealizationResult result, string folderPath, int stageNumber)
    {
        if (result == null)
        {
            Debug.LogError($"[MonteCarloCSVExporter] Cannot export stage {stageNumber} - result is null");
            return false;
        }
        
        if (string.IsNullOrEmpty(folderPath))
        {
            Debug.LogError($"[MonteCarloCSVExporter] Cannot export stage {stageNumber} - folder path is empty");
            return false;
        }
        
        try
        {
            // Get data from snapshot if available, otherwise from final fields
            MonteCarloStateSnapshot snapshot = result.Snapshot;
            
            if (snapshot != null)
            {
                // Export from snapshot (most complete data)
                ExportBedElevation(snapshot.BedElevation, snapshot.NumCrossSections, snapshot.WidthResolution, folderPath, stageNumber);
                ExportVelocityU(snapshot.VelocityU, snapshot.NumCrossSections, snapshot.WidthResolution, folderPath, stageNumber);
                ExportVelocityV(snapshot.VelocityV, snapshot.NumCrossSections, snapshot.WidthResolution, folderPath, stageNumber);
                ExportVelocityMagnitude(snapshot.VelocityU, snapshot.VelocityV, snapshot.NumCrossSections, snapshot.WidthResolution, folderPath, stageNumber);
                ExportWaterDepth(snapshot.WaterDepth, snapshot.NumCrossSections, snapshot.WidthResolution, folderPath, stageNumber);
                ExportCellType(snapshot.CellType, snapshot.NumCrossSections, snapshot.WidthResolution, folderPath, stageNumber);
            }
            else if (result.FinalBedElevation != null)
            {
                // Export from final fields (limited data - only what's available)
                int numCrossSections = result.FinalBedElevation.GetLength(0);
                int widthResolution = result.FinalBedElevation.GetLength(1);
                
                ExportBedElevation(result.FinalBedElevation, numCrossSections, widthResolution, folderPath, stageNumber);
                
                // Export velocity magnitude if available
                if (result.FinalVelocityMagnitude != null)
                {
                    ExportVelocityMagnitude(result.FinalVelocityMagnitude, numCrossSections, widthResolution, folderPath, stageNumber);
                }
                
                Debug.LogWarning($"[MonteCarloCSVExporter] Stage {stageNumber}: Only partial data exported (no snapshot available). Enable 'Store Full Ensemble' for complete data.");
            }
            else
            {
                Debug.LogWarning($"[MonteCarloCSVExporter] Stage {stageNumber}: No full field data available for export. Enable 'Store Full Ensemble' in MonteCarloSimulationManager.");
                return false;
            }
            
            Debug.Log($"[MonteCarloCSVExporter] Exported full field data for stage {stageNumber} to {folderPath}");
            return true;
        }
        catch (Exception e)
        {
            Debug.LogError($"[MonteCarloCSVExporter] Failed to export stage {stageNumber}: {e.Message}");
            return false;
        }
    }
    
    /// <summary>
    /// Exports bed elevation field to CSV.
    /// </summary>
    private static void ExportBedElevation(double[,] data, int numCrossSections, int widthResolution, string folderPath, int stageNumber)
    {
        string filePath = Path.Combine(folderPath, $"bed_elevation_stage_{stageNumber}.csv");
        Export2DArray(data, numCrossSections, widthResolution, filePath, "bed_elevation");
    }
    
    /// <summary>
    /// Exports U velocity component to CSV.
    /// </summary>
    private static void ExportVelocityU(double[,] data, int numCrossSections, int widthResolution, string folderPath, int stageNumber)
    {
        string filePath = Path.Combine(folderPath, $"velocity_u_stage_{stageNumber}.csv");
        Export2DArray(data, numCrossSections, widthResolution, filePath, "velocity_u");
    }
    
    /// <summary>
    /// Exports V velocity component to CSV.
    /// </summary>
    private static void ExportVelocityV(double[,] data, int numCrossSections, int widthResolution, string folderPath, int stageNumber)
    {
        string filePath = Path.Combine(folderPath, $"velocity_v_stage_{stageNumber}.csv");
        Export2DArray(data, numCrossSections, widthResolution, filePath, "velocity_v");
    }
    
    /// <summary>
    /// Exports velocity magnitude to CSV.
    /// </summary>
    private static void ExportVelocityMagnitude(double[,] u, double[,] v, int numCrossSections, int widthResolution, string folderPath, int stageNumber)
    {
        // Calculate magnitude from U and V components
        double[,] magnitude = new double[numCrossSections, widthResolution];
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                magnitude[i, w] = Math.Sqrt(u[i, w] * u[i, w] + v[i, w] * v[i, w]);
            }
        }
        
        string filePath = Path.Combine(folderPath, $"velocity_magnitude_stage_{stageNumber}.csv");
        Export2DArray(magnitude, numCrossSections, widthResolution, filePath, "velocity_magnitude");
    }
    
    /// <summary>
    /// Exports velocity magnitude from pre-computed array.
    /// </summary>
    private static void ExportVelocityMagnitude(double[,] magnitude, int numCrossSections, int widthResolution, string folderPath, int stageNumber)
    {
        string filePath = Path.Combine(folderPath, $"velocity_magnitude_stage_{stageNumber}.csv");
        Export2DArray(magnitude, numCrossSections, widthResolution, filePath, "velocity_magnitude");
    }
    
    /// <summary>
    /// Exports water depth field to CSV.
    /// </summary>
    private static void ExportWaterDepth(double[,] data, int numCrossSections, int widthResolution, string folderPath, int stageNumber)
    {
        string filePath = Path.Combine(folderPath, $"water_depth_stage_{stageNumber}.csv");
        Export2DArray(data, numCrossSections, widthResolution, filePath, "water_depth");
    }
    
    /// <summary>
    /// Exports cell type field to CSV.
    /// </summary>
    private static void ExportCellType(int[,] data, int numCrossSections, int widthResolution, string folderPath, int stageNumber)
    {
        string filePath = Path.Combine(folderPath, $"cell_type_stage_{stageNumber}.csv");
        
        var culture = CultureInfo.InvariantCulture;
        var sb = new StringBuilder();
        
        // Header row: cross_section, width_0, width_1, ..., width_N
        sb.Append("cross_section");
        for (int w = 0; w < widthResolution; w++)
        {
            sb.Append($",width_{w}");
        }
        sb.AppendLine();
        
        // Data rows
        for (int i = 0; i < numCrossSections; i++)
        {
            sb.Append(i.ToString(culture));
            for (int w = 0; w < widthResolution; w++)
            {
                sb.Append($",{data[i, w].ToString(culture)}");
            }
            sb.AppendLine();
        }
        
        File.WriteAllText(filePath, sb.ToString());
    }
    
    /// <summary>
    /// Exports a 2D double array to CSV in grid format.
    /// </summary>
    private static void Export2DArray(double[,] data, int numCrossSections, int widthResolution, string filePath, string fieldName)
    {
        if (data == null)
        {
            Debug.LogWarning($"[MonteCarloCSVExporter] Cannot export {fieldName} - data array is null");
            return;
        }
        
        var culture = CultureInfo.InvariantCulture;
        var sb = new StringBuilder();
        
        // Header row: cross_section, width_0, width_1, ..., width_N
        sb.Append("cross_section");
        for (int w = 0; w < widthResolution; w++)
        {
            sb.Append($",width_{w}");
        }
        sb.AppendLine();
        
        // Data rows
        for (int i = 0; i < numCrossSections; i++)
        {
            sb.Append(i.ToString(culture));
            for (int w = 0; w < widthResolution; w++)
            {
                sb.Append($",{data[i, w].ToString("F6", culture)}");
            }
            sb.AppendLine();
        }
        
        File.WriteAllText(filePath, sb.ToString());
    }
    
    /// <summary>
    /// Exports summary statistics for all stages to a single CSV file.
    /// </summary>
    /// <param name="ensemble">The Monte Carlo ensemble containing all realization results.</param>
    /// <param name="folderPath">Path to the results folder.</param>
    /// <returns>True if export succeeded, false otherwise.</returns>
    public static bool ExportSummaryStatistics(MonteCarloEnsemble ensemble, string folderPath)
    {
        if (ensemble == null)
        {
            Debug.LogError("[MonteCarloCSVExporter] Cannot export summary - ensemble is null");
            return false;
        }
        
        if (string.IsNullOrEmpty(folderPath))
        {
            Debug.LogError("[MonteCarloCSVExporter] Cannot export summary - folder path is empty");
            return false;
        }
        
        try
        {
            string filePath = Path.Combine(folderPath, "summary_statistics.csv");
            var culture = CultureInfo.InvariantCulture;
            var sb = new StringBuilder();
            
            // Header row
            sb.AppendLine("stage,transport_coefficient_K,perlin_seed,mean_bed_elevation,std_bed_elevation,min_bed_elevation,max_bed_elevation,mean_velocity,max_erosion_rate,mean_water_depth,max_velocity");
            
            // Get all realizations
            var realizations = ensemble.GetAllRealizations();
            
            // Sort by stage number (should already be sorted, but ensure it)
            realizations.Sort((a, b) => a.RealizationIndex.CompareTo(b.RealizationIndex));
            
            // Write data rows
            foreach (var result in realizations)
            {
                int stage = result.RealizationIndex + 1; // 1-indexed for display
                
                // Get additional stats from snapshot if available
                double meanWaterDepth = double.NaN;
                double maxVelocity = double.NaN;
                
                if (result.Snapshot != null)
                {
                    meanWaterDepth = result.Snapshot.MeanWaterDepth;
                    maxVelocity = result.Snapshot.MaxVelocity;
                }
                
                // Format values, using "NaN" for missing data
                string meanWaterDepthStr = double.IsNaN(meanWaterDepth) ? "NaN" : meanWaterDepth.ToString("F6", culture);
                string maxVelocityStr = double.IsNaN(maxVelocity) ? "NaN" : maxVelocity.ToString("F6", culture);
                
                sb.AppendLine($"{stage}," +
                             $"{result.TransportCoefficient.ToString("F6", culture)}," +
                             $"{result.PerlinSeed}," +
                             $"{result.MeanBedElevation.ToString("F6", culture)}," +
                             $"{result.StdBedElevation.ToString("F6", culture)}," +
                             $"{result.MinBedElevation.ToString("F6", culture)}," +
                             $"{result.MaxBedElevation.ToString("F6", culture)}," +
                             $"{result.MeanVelocity.ToString("F6", culture)}," +
                             $"{result.MaxErosionRate.ToString("F6", culture)}," +
                             $"{meanWaterDepthStr}," +
                             $"{maxVelocityStr}");
            }
            
            File.WriteAllText(filePath, sb.ToString());
            Debug.Log($"[MonteCarloCSVExporter] Exported summary statistics to {filePath} ({realizations.Count} stages)");
            return true;
        }
        catch (Exception e)
        {
            Debug.LogError($"[MonteCarloCSVExporter] Failed to export summary statistics: {e.Message}");
            return false;
        }
    }
}
