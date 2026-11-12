Shader "RingBridge/RecordedSprite"
{
    Properties
    {
        [PerRendererData] _MainTex ("Sprite", 2D) = "white" {}
        _Color ("Tint", Color) = (1,1,1,1)
        _WhiteMask ("Use alpha as white star", Float) = 0
        _UseArc ("Angular progress", Float) = 0
        _ArcStart ("Clockwise start", Float) = 0
        _ArcSweep ("Clockwise sweep", Float) = 90
        _Progress ("Progress", Range(0,1)) = 1
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" "CanUseSpriteAtlas"="True" }
        Cull Off ZWrite Off Lighting Off
        Blend One OneMinusSrcAlpha
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            sampler2D _MainTex;
            fixed4 _Color;
            float _WhiteMask, _UseArc, _ArcStart, _ArcSweep, _Progress;
            struct Input { float4 vertex:POSITION; float2 uv:TEXCOORD0; fixed4 color:COLOR; };
            struct Output { float4 position:SV_POSITION; float2 uv:TEXCOORD0; fixed4 color:COLOR; float2 local:TEXCOORD1; };
            Output vert(Input v)
            {
                Output o; o.position=UnityObjectToClipPos(v.vertex); o.uv=v.uv;
                o.color=v.color*_Color; o.local=v.vertex.xy; return o;
            }
            fixed4 frag(Output i):SV_Target
            {
                if (_UseArc > .5 && _Progress < .9999)
                {
                    float angle=atan2(-i.local.y,i.local.x)*57.2957795;
                    float delta=fmod(angle-_ArcStart+720,360);
                    if(delta>355) delta=0;
                    clip(_Progress*_ArcSweep-delta);
                }
                fixed4 c=tex2D(_MainTex,i.uv);
                #ifndef UNITY_COLORSPACE_GAMMA
                c.rgb=LinearToGammaSpace(c.rgb);
                #endif
                c.rgb=lerp(c.rgb,fixed3(1,1,1),_WhiteMask);
                c*=i.color; c.rgb*=c.a; return c;
            }
            ENDCG
        }
    }
}
