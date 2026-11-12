Shader "Music/Beat Arc Dissolve"
{
    Properties
    {
        [HDR] _Color ("Arc Color", Color) = (2.2, 0.35, 4, 1)
        _NoiseTex ("Dissolve Noise", 2D) = "white" {}
        _Dissolve ("Dissolve", Range(0, 1)) = 0
        _EdgeWidth ("Bright Edge Width", Range(0.001, 0.25)) = 0.08
        [HDR] _EdgeColor ("Edge Color", Color) = (4, 1.4, 5, 1)
    }
    SubShader
    {
        Tags { "Queue"="Transparent+60" "RenderType"="Transparent" }
        Blend SrcAlpha One
        Cull Off
        ZWrite Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _NoiseTex;
            float4 _NoiseTex_ST;
            fixed4 _Color;
            fixed4 _EdgeColor;
            float _Dissolve;
            float _EdgeWidth;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _NoiseTex);
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float noise = tex2D(_NoiseTex, i.uv).r;
                float visible = step(_Dissolve, noise);
                float edge = saturate(1.0 - abs(noise - _Dissolve) / max(_EdgeWidth, 0.001));
                clip(visible - 0.01);
                fixed3 color = _Color.rgb + _EdgeColor.rgb * edge;
                return fixed4(color, _Color.a * visible);
            }
            ENDCG
        }
    }
}
