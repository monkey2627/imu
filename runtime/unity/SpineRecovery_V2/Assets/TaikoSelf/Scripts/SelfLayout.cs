// Auto-generated from 敲鼓 · 正面鼓面 · 自主节奏. Design px: 844x390, origin top-left, y down.
using System.Collections.Generic;
using UnityEngine;

namespace TaikoSelf
{
    public struct LayerInfo
    {
        public float x;
        public float y;
        public int order;
        public LayerInfo(float x, float y, int order) { this.x = x; this.y = y; this.order = order; }
    }

    public struct SparkleInfo
    {
        public float phase;
        public float period;
        public SparkleInfo(float phase, float period) { this.phase = phase; this.period = period; }
    }

    public static class SelfLayout
    {
        public const float DrumX = 422.0f, DrumY = 215.0f;
        /// <summary>SVG angles (deg, clockwise from +x) of each progress arc.</summary>
        public const float LiftStart = 179.0f, LiftSweep = 62.0f;
        public const float HoldStart = 259.0f, HoldSweep = 102.0f;
        public const float ReturnStart = 19.0f, ReturnSweep = 142.0f;

        public static readonly Dictionary<string, LayerInfo> Layers = new Dictionary<string, LayerInfo>
        {
            { "ts_bg", new LayerInfo(422.0f, 195.0f, 0) },
            { "ts_sparkle_0", new LayerInfo(96.0f, 100.0f, 1) },
            { "ts_sparkle_1", new LayerInfo(232.0f, 150.0f, 1) },
            { "ts_sparkle_2", new LayerInfo(612.0f, 40.0f, 1) },
            { "ts_sparkle_3", new LayerInfo(790.0f, 80.0f, 1) },
            { "ts_sparkle_4", new LayerInfo(812.0f, 250.0f, 1) },
            { "ts_star_glow", new LayerInfo(676.0f, 162.0f, 1) },
            { "ts_meteor", new LayerInfo(267.0f, 46.0f, 2) },
            { "ts_lanterns", new LayerInfo(449.5f, 231.5f, 3) },
            { "ts_spin_ring", new LayerInfo(422.0f, 215.0f, 4) },
            { "ts_bases", new LayerInfo(422.0f, 215.0f, 10) },
            { "ts_lift_fill", new LayerInfo(422.0f, 215.0f, 11) },
            { "ts_hold_fill", new LayerInfo(422.0f, 215.0f, 11) },
            { "ts_ret_fill", new LayerInfo(422.0f, 215.0f, 11) },
            { "ts_lift_prog", new LayerInfo(422.0f, 215.0f, 12) },
            { "ts_ret_comet", new LayerInfo(422.0f, 215.0f, 15) },
            { "ts_ret_clip", new LayerInfo(422.0f, 215.0f, 0) },
            { "ts_ret_dot", new LayerInfo(531.7f, 252.8f, 18) },
            { "ts_core_glow", new LayerInfo(422.0f, 215.0f, 20) },
            { "ts_core", new LayerInfo(422.0f, 215.0f, 21) },
            { "ts_sheen", new LayerInfo(422.0f, 215.0f, 22) },
            { "ts_const", new LayerInfo(748.5f, 224.5f, 30) },
            { "ts_cap_lift", new LayerInfo(422.0f, 30.5f, 40) },
            { "ts_cap_hold", new LayerInfo(422.0f, 30.5f, 40) },
            { "ts_cap_ret", new LayerInfo(422.0f, 30.5f, 40) },
            { "ts_hud_pause", new LayerInfo(40.0f, 30.0f, 50) },
        };

        /// <summary>Hold stars (保持), lit one every 0.4 s.</summary>
        public static readonly Vector2[] Stars =
        {
            new Vector2(417.0f, 101.1f),
            new Vector2(450.5f, 104.6f),
            new Vector2(481.6f, 117.8f),
            new Vector2(507.4f, 139.5f),
            new Vector2(525.7f, 167.7f),
            new Vector2(535.0f, 200.1f)
        };

        /// <summary>Notes along the return arc; NoteAngles are their SVG angles.</summary>
        public static readonly Vector2[] Notes =
        {
            new Vector2(504.4f, 264.2f),
            new Vector2(477.7f, 293.2f),
            new Vector2(441.7f, 309.0f),
            new Vector2(402.3f, 309.0f),
            new Vector2(366.3f, 293.2f),
            new Vector2(339.6f, 264.2f)
        };
        public static readonly float[] NoteAngles = { 30.83f, 54.50f, 78.17f, 101.83f, 125.50f, 149.17f };

        /// <summary>Star dust dropped from the return arc as the dot passes; DustVariant picks ts_dust_N.</summary>
        public static readonly Vector2[] Dust =
        {
            new Vector2(525.4f, 258.0f),
            new Vector2(521.9f, 283.3f),
            new Vector2(499.5f, 295.9f),
            new Vector2(486.0f, 317.7f),
            new Vector2(460.5f, 320.2f),
            new Vector2(439.4f, 334.7f),
            new Vector2(415.1f, 326.8f),
            new Vector2(389.9f, 331.7f),
            new Vector2(370.8f, 314.6f),
            new Vector2(345.8f, 309.0f),
            new Vector2(335.1f, 285.7f),
            new Vector2(314.5f, 270.5f)
        };
        public static readonly float[] DustAngles = { 22.55f, 34.38f, 46.22f, 58.05f, 69.88f, 81.72f, 93.55f, 105.38f, 117.22f, 129.05f, 140.88f, 152.72f };
        public static readonly int[] DustVariant = { 2, 1, 0, 3, 0, 1, 2, 1, 0, 3, 0, 1 };

        public static readonly SparkleInfo[] Sparkles =
        {
            new SparkleInfo(-0.00f, 2.40f),
            new SparkleInfo(0.60f, 2.40f),
            new SparkleInfo(1.20f, 2.40f),
            new SparkleInfo(1.80f, 2.40f),
            new SparkleInfo(0.90f, 2.40f)
        };

        public const float MeteorCenterOffsetX = 27.0f;
        public const float MeteorCenterOffsetY = -10.0f;
    }
}
