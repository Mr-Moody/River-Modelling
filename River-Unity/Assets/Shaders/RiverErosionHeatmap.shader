Shader "Custom/RiverErosionHeatmap"
{
    Properties
    {
        _Color ("Base Color", Color) = (1,1,1,1)
        _MaxErosionRate ("Max Erosion Rate", Float) = 1.0
        _MinErosionRate ("Min Erosion Rate", Float) = -1.0
        _ErosionThreshold ("Erosion Threshold", Float) = 0.001
        _Glossiness ("Smoothness", Range(0,1)) = 0.5
        _Metallic ("Metallic", Range(0,1)) = 0.0
        [Toggle] _ShowDeposition ("Show Deposition", Float) = 1
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 200

        CGPROGRAM
        // Physically based Standard lighting model, and enable shadows on all light types
        #pragma surface surf Standard fullforwardshadows

        // Use shader model 3.0 target, to get nicer looking lighting
        #pragma target 3.0

        struct Input
        {
            float2 uv_MainTex;
            float4 vertexColor : COLOR;
        };

        half _Glossiness;
        half _Metallic;
        fixed4 _Color;
        float _MaxErosionRate;
        float _MinErosionRate;
        float _ErosionThreshold;
        float _ShowDeposition;

        // Heatmap color function: Blue (deposition/no change) to Red (high erosion)
        // Input: normalized erosion value (0 = no erosion, 1 = max erosion)
        // For deposition (positive dh_dt), we can show as blue/cyan
        // For erosion (negative dh_dt), we show as yellow/red
        fixed3 GetErosionHeatmapColor(float normalizedErosion, float isDeposition)
        {
            // Ensure value is in 0-1 range
            normalizedErosion = saturate(normalizedErosion);
            
            fixed3 color;
            
            if (_ShowDeposition > 0.5 && isDeposition > 0.5)
            {
                // Deposition case: Blue to Cyan gradient (bed rising)
                // More deposition = brighter cyan/blue
                color = lerp(fixed3(0.0, 0.0, 0.5), fixed3(0.0, 1.0, 1.0), normalizedErosion);
            }
            else
            {
                // Erosion case: Green -> Yellow -> Red gradient (bed lowering)
                // Blue (0.0) -> Cyan (0.25) -> Green (0.5) -> Yellow (0.75) -> Red (1.0)
                if (normalizedErosion < 0.25)
                {
                    // Blue to Cyan (low erosion)
                    float t = normalizedErosion / 0.25;
                    color = lerp(fixed3(0.0, 0.0, 1.0), fixed3(0.0, 1.0, 1.0), t);
                }
                else if (normalizedErosion < 0.5)
                {
                    // Cyan to Green (moderate erosion)
                    float t = (normalizedErosion - 0.25) / 0.25;
                    color = lerp(fixed3(0.0, 1.0, 1.0), fixed3(0.0, 1.0, 0.0), t);
                }
                else if (normalizedErosion < 0.75)
                {
                    // Green to Yellow (high erosion)
                    float t = (normalizedErosion - 0.5) / 0.25;
                    color = lerp(fixed3(0.0, 1.0, 0.0), fixed3(1.0, 1.0, 0.0), t);
                }
                else
                {
                    // Yellow to Red (very high erosion - bank movement)
                    float t = (normalizedErosion - 0.75) / 0.25;
                    color = lerp(fixed3(1.0, 1.0, 0.0), fixed3(1.0, 0.0, 0.0), t);
                }
            }
            
            return color;
        }

        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            // Read erosion data from vertex color
            // Red channel: normalized erosion rate (0-1, where 1 = max erosion)
            // Green channel: flag for deposition (1.0 = deposition, 0.0 = erosion)
            // Blue channel: bank movement indicator (1.0 = active bank migration, 0.0 = no migration)
            float normalizedErosion = saturate(IN.vertexColor.r);
            float isDeposition = IN.vertexColor.g;
            float bankMovement = IN.vertexColor.b;
            
            // If bank movement is active, enhance the red color to show bank migration
            fixed3 heatmapColor = GetErosionHeatmapColor(normalizedErosion, isDeposition);
            
            // Enhance color for active bank migration (make it more red/orange)
            if (bankMovement > 0.5)
            {
                // Mix with bright red/orange to highlight bank movement areas
                heatmapColor = lerp(heatmapColor, fixed3(1.0, 0.3, 0.0), 0.5);
            }
            
            // Apply base color tint if needed
            fixed3 finalColor = heatmapColor * _Color.rgb;
            
            o.Albedo = finalColor;
            o.Metallic = _Metallic;
            o.Smoothness = _Glossiness;
            o.Alpha = _Color.a;
        }
        ENDCG
    }
    FallBack "Diffuse"
}

