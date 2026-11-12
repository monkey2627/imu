Shader "TaikoSelf/RecordedSprite"
{
    Properties
    {
        [PerRendererData] _MainTex ("Sprite", 2D) = "white" {}
        _Color ("Tint", Color) = (1,1,1,1)
        _WhiteMask ("White alpha artwork", Float) = 0
        _UseArc ("Angular clip", Float) = 0
        _ArcStart ("Clockwise start", Float) = 0
        _ArcSweep ("Sweep", Float) = 90
        _Progress ("Progress", Range(0,1)) = 1
        _AngleOffset ("Sprite rotation clockwise", Float) = 0
        _BackgroundScale ("Extend background edges", Float) = 1
        _HoldProgressArc ("Gold hold progress", Float) = 0
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
            float _WhiteMask, _UseArc, _ArcStart, _ArcSweep, _Progress, _AngleOffset, _BackgroundScale;
            float _HoldProgressArc;
            struct Input { float4 vertex:POSITION; float2 uv:TEXCOORD0; fixed4 color:COLOR; };
            struct Output { float4 position:SV_POSITION; float2 uv:TEXCOORD0; fixed4 color:COLOR; float2 local:TEXCOORD1; };
            Output vert(Input v)
            {
                Output o; o.position=UnityObjectToClipPos(v.vertex); o.uv=v.uv;
                o.color=v.color*_Color; o.local=v.vertex.xy; return o;
            }
            fixed4 frag(Output i):SV_Target
            {
                if (_HoldProgressArc > .5)
                {
                    // Same clockwise design coordinates as the lift/return arcs.
                    // Distance to an arc gives rounded ends without texture blur.
                    const float radius = 1.24;
                    float start = radians(_ArcStart);
                    float end = radians(_ArcStart + saturate(_Progress) * _ArcSweep);
                    float finish = radians(_ArcStart + _ArcSweep);
                    float2 a = radius * float2(cos(start), -sin(start));
                    float2 b = radius * float2(cos(end), -sin(end));
                    float2 c = radius * float2(cos(finish), -sin(finish));
                    float delta = fmod(degrees(atan2(-i.local.y, i.local.x)) - _ArcStart + 720, 360);
                    float radial = abs(length(i.local) - radius);
                    float track = delta <= _ArcSweep ? radial : min(length(i.local-a), length(i.local-c));
                    float distance = delta <= saturate(_Progress)*_ArcSweep ? radial : min(length(i.local-a), length(i.local-b));
                    float lit = step(.00001, _Progress);
                    float core = (1-smoothstep(.013, .023, distance))*lit;
                    float glow = exp(-pow(distance/.065, 2))*.55*lit;
                    float band = exp(-pow(distance/.032, 2))*.85*lit;
                    float head = (1-smoothstep(.022, .045, length(i.local-b)))*lit;
                    float rail = (1-smoothstep(.006, .013, track))*.22;
                    float alpha = max(rail, max(glow, max(band, max(core, head))));
                    fixed3 color = lerp(fixed3(1,.72,.18), fixed3(1,.98,.83), max(core, head));
                    fixed4 arc = fixed4(color, alpha)*i.color;
                    arc.rgb *= arc.a;
                    return arc;
                }
                if(_UseArc>.5)
                {
                    float angle=atan2(-i.local.y,i.local.x)*57.2957795+_AngleOffset;
                    float delta=fmod(angle-_ArcStart+720,360);
                    if(delta>359.5) delta=0;
                    clip(_Progress*_ArcSweep-delta);
                }
                float2 uv=(i.uv-.5)*_BackgroundScale+.5;
                fixed4 c=tex2D(_MainTex,uv);
                // Fade extended top/bottom edges into their corner colour instead
                // of stretching the baked drum or extruding the landscape stripes.
                float outsideY=max(-uv.y,uv.y-1);
                fixed4 edge=tex2D(_MainTex,float2(.02,saturate(uv.y)));
                c=lerp(c,edge,smoothstep(0,.08,max(0,outsideY)));
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
