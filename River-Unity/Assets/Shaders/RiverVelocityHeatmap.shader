Shader "Custom/RiverVelocityHeatmap"
{
    Properties
    {
        _Color ("Base Color", Color) = (1,1,1,1)
        _MaxVelocity ("Max Velocity", Float) = 1.0
        _MinVelocity ("Min Velocity", Float) = 0.0
        _Glossiness ("Smoothness", Range(0,1)) = 0.5
        _Metallic ("Metallic", Range(0,1)) = 0.0
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
        float _MaxVelocity;
        float _MinVelocity;

        // Heatmap color function: Blue (0 velocity) to Red (high velocity)
        // Input is already normalized to 0-1 range
        fixed3 GetHeatmapColor(float normalizedVel)
        {
            // Ensure value is in 0-1 range
            normalizedVel = saturate(normalizedVel);
            
            // Create heatmap gradient: Blue -> Cyan -> Green -> Yellow -> Red
            // Blue (0.0) -> Cyan (0.25) -> Green (0.5) -> Yellow (0.75) -> Red (1.0)
            fixed3 color;
            
            if (normalizedVel < 0.25)
            {
                // Blue to Cyan
                float t = normalizedVel / 0.25;
                color = lerp(fixed3(0.0, 0.0, 1.0), fixed3(0.0, 1.0, 1.0), t);
            }
            else if (normalizedVel < 0.5)
            {
                // Cyan to Green
                float t = (normalizedVel - 0.25) / 0.25;
                color = lerp(fixed3(0.0, 1.0, 1.0), fixed3(0.0, 1.0, 0.0), t);
            }
            else if (normalizedVel < 0.75)
            {
                // Green to Yellow
                float t = (normalizedVel - 0.5) / 0.25;
                color = lerp(fixed3(0.0, 1.0, 0.0), fixed3(1.0, 1.0, 0.0), t);
            }
            else
            {
                // Yellow to Red
                float t = (normalizedVel - 0.75) / 0.25;
                color = lerp(fixed3(1.0, 1.0, 0.0), fixed3(1.0, 0.0, 0.0), t);
            }
            
            return color;
        }

        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            // Read normalized velocity from vertex color (stored in red channel)
            // The velocity magnitude is already normalized to 0-1 range in C# code
            float normalizedVelocity = saturate(IN.vertexColor.r);
            
            // Get heatmap color based on normalized velocity (0-1 range)
            // We pass the normalized value directly to GetHeatmapColor
            // The function will handle the color mapping
            fixed3 heatmapColor = GetHeatmapColor(normalizedVelocity);
            
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

