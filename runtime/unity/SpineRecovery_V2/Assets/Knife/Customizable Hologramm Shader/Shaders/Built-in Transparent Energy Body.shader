Shader "Knife/Built-in/Transparent Energy Body"
{
    Properties
    {
        [HDR] _CenterColor("Center Color", Color) = (0.02, 0.35, 1.0, 1)
        [HDR] _LeftColor("Left Color", Color) = (1.0, 0.48, 0.0, 1)
        [HDR] _RightColor("Right Color", Color) = (0.0, 1.0, 0.12, 1)
        _BodyAlpha("Body Alpha", Range(0, 1)) = 0.07
        _FresnelPower("Edge Sharpness", Range(0.5, 8)) = 2.4
        _FresnelIntensity("Edge Intensity", Range(0, 8)) = 3.2
        _GridScale("Network Scale", Range(1, 80)) = 18
        _GridWidth("Network Width", Range(0.005, 0.15)) = 0.035
        _GridIntensity("Network Intensity", Range(0, 5)) = 1.5
        _DotSize("Node Size", Range(0.01, 0.25)) = 0.08
        _DotIntensity("Node Intensity", Range(0, 8)) = 2.5
        _ScanDensity("Scan Density", Range(1, 100)) = 28
        _ScanSpeed("Scan Speed", Range(-5, 5)) = 0.45
        _ScanIntensity("Scan Intensity", Range(0, 4)) = 0.7
        _ColorSplit("Color Split", Range(0.01, 2)) = 0.32
        _PulseSpeed("Pulse Speed", Range(0, 8)) = 1.2
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Transparent"
            "Queue" = "Transparent+20"
            "IgnoreProjector" = "True"
        }

        Pass
        {
            Name "EnergyBody"
            Tags { "LightMode" = "ForwardBase" }

            Blend SrcAlpha One
            ZWrite Off
            ZTest LEqual
            Cull Back

            CGPROGRAM
            #pragma target 3.0
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma multi_compile_instancing
            #include "UnityCG.cginc"

            half4 _CenterColor;
            half4 _LeftColor;
            half4 _RightColor;
            half _BodyAlpha;
            half _FresnelPower;
            half _FresnelIntensity;
            half _GridScale;
            half _GridWidth;
            half _GridIntensity;
            half _DotSize;
            half _DotIntensity;
            half _ScanDensity;
            half _ScanSpeed;
            half _ScanIntensity;
            half _ColorSplit;
            half _PulseSpeed;

            struct Attributes
            {
                float4 positionOS : POSITION;
                half3 normalOS : NORMAL;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionOS : TEXCOORD0;
                float3 positionWS : TEXCOORD1;
                half3 normalWS : TEXCOORD2;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            Varyings Vert(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                output.positionCS = UnityObjectToClipPos(input.positionOS);
                output.positionWS = mul(unity_ObjectToWorld, input.positionOS).xyz;
                output.positionOS = input.positionOS.xyz;
                output.normalWS = UnityObjectToWorldNormal(input.normalOS);
                return output;
            }

            // A triangular lattice. Three families of lines make the surface read
            // like a connected low-poly network without geometry-shader support.
            half TriangleGrid(float2 p, out half nodes)
            {
                const half sin60 = 0.8660254h;
                half3 axes = half3(p.x, p.x * 0.5h + p.y * sin60,
                                  p.x * 0.5h - p.y * sin60);
                half3 lineDistance = abs(frac(axes + 0.5h) - 0.5h);
                half3 aa = max(fwidth(axes), 0.001h);
                half3 lines = 1.0h - smoothstep(_GridWidth, _GridWidth + aa, lineDistance);

                half2 cell = abs(frac(p + 0.5h) - 0.5h);
                half nodeDistance = length(cell);
                nodes = 1.0h - smoothstep(_DotSize, _DotSize + max(fwidth(nodeDistance), 0.002h), nodeDistance);
                return saturate(max(lines.x, max(lines.y, lines.z)));
            }

            half Hash31(float3 p)
            {
                p = frac(p * 0.1031h);
                p += dot(p, p.yzx + 33.33h);
                return frac((p.x + p.y) * p.z);
            }

            half4 Frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(input);
                half3 n = normalize(input.normalWS);
                half3 viewDir = normalize(_WorldSpaceCameraPos.xyz - input.positionWS);
                half fresnel = pow(saturate(1.0h - abs(dot(n, viewDir))), _FresnelPower);

                // Triplanar projection avoids requiring good UVs on the FBX.
                half3 weights = pow(abs(n), 4.0h);
                weights /= max(weights.x + weights.y + weights.z, 0.001h);
                float3 p = input.positionOS * _GridScale;
                half nodeX, nodeY, nodeZ;
                half gridX = TriangleGrid(p.yz, nodeX);
                half gridY = TriangleGrid(p.xz, nodeY);
                half gridZ = TriangleGrid(p.xy, nodeZ);
                half grid = dot(half3(gridX, gridY, gridZ), weights);
                half nodes = dot(half3(nodeX, nodeY, nodeZ), weights);

                half split = max(_ColorSplit, 0.001h);
                half leftWeight = saturate(-input.positionOS.x / split);
                half rightWeight = saturate(input.positionOS.x / split);
                half3 color = lerp(_CenterColor.rgb, _LeftColor.rgb, leftWeight);
                color = lerp(color, _RightColor.rgb, rightWeight);

                half scanWave = sin((input.positionOS.y * _ScanDensity - _Time.y * _ScanSpeed) * 6.2831853h);
                half scan = pow(saturate(scanWave * 0.5h + 0.5h), 14.0h);
                half sparkle = step(0.965h, Hash31(floor(p * 1.7h) + floor(_Time.y * 0.6h)));
                half pulse = 0.82h + 0.18h * sin(_Time.y * _PulseSpeed + input.positionOS.y * 5.0h);

                half energy = _BodyAlpha;
                energy += fresnel * _FresnelIntensity;
                energy += grid * _GridIntensity;
                energy += nodes * (_DotIntensity + sparkle * 2.0h);
                energy += scan * _ScanIntensity;
                energy *= pulse;

                // Alpha stays bounded while HDR RGB carries the bloom energy.
                half alpha = saturate(_BodyAlpha + fresnel * 0.72h + grid * 0.42h + nodes * 0.75h + scan * 0.2h);
                return half4(color * energy, alpha);
            }
            ENDCG
        }
    }

    FallBack Off
}
