Shader "Hidden/UniGLTF/NormalMapExporter"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            sampler2D _MainTex;

            fixed4 frag (v2f i) : SV_Target
            {
                half4 col = tex2D(_MainTex, i.uv);

                // Convert from Unity's compressed normal representation when
                // X is packed into alpha. A regular linear texture assigned to
                // a normal property has opaque alpha and must instead decode
                // X/Y directly; using UnpackNormal on DXT5nm platforms would
                // turn a flat (0.5, 0.5) normal into +X.
                half3 normal;
                if (col.a >= 0.999h)
                {
                    normal.xy = col.xy * 2 - 1;
                    normal.z = sqrt(1 - saturate(dot(normal.xy, normal.xy)));
                }
                else
                {
                    normal = UnpackNormal(col);
                }

                col.xyz = (normal + 1) * 0.5;
                col.w = 1;

                return col;
            }
            ENDCG
        }
    }
}
