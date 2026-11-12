// Auto-generated from 圆环 C · 臀桥三段. Design px: 844x390, origin top-left, y down.
using System.Collections.Generic;

namespace RingBridge
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

    public static class BridgeLayout
    {
        public const int PoseFrames = 13;

        /// <summary>Sprite name -> design-space center of its default placement and sorting order.</summary>
        public static readonly Dictionary<string, LayerInfo> Layers = new Dictionary<string, LayerInfo>
        {
            { "rb_bg", new LayerInfo(422.0f, 195.0f, 0) },
            { "rb_sparkle_0", new LayerInfo(96.0f, 100.0f, 1) },
            { "rb_sparkle_1", new LayerInfo(232.0f, 150.0f, 1) },
            { "rb_sparkle_2", new LayerInfo(612.0f, 40.0f, 1) },
            { "rb_sparkle_3", new LayerInfo(790.0f, 80.0f, 1) },
            { "rb_sparkle_4", new LayerInfo(812.0f, 250.0f, 1) },
            { "rb_star_glow", new LayerInfo(676.0f, 162.0f, 1) },
            { "rb_meteor", new LayerInfo(267.0f, 46.0f, 2) },
            { "rb_lanterns", new LayerInfo(449.5f, 231.5f, 3) },
            { "rb_spin_ring", new LayerInfo(422.0f, 215.0f, 4) },
            { "rb_lift_fill", new LayerInfo(422.0f, 215.0f, 10) },
            { "rb_lift_static", new LayerInfo(422.0f, 215.0f, 11) },
            { "rb_lift_track", new LayerInfo(422.0f, 215.0f, 12) },
            { "rb_lift_prog", new LayerInfo(422.0f, 215.0f, 13) },
            { "rb_lift_done", new LayerInfo(422.0f, 215.0f, 14) },
            { "rb_lift_head", new LayerInfo(286.3f, 206.7f, 15) },
            { "rb_lift_ink", new LayerInfo(355.9f, 153.6f, 16) },
            { "rb_hold_fill", new LayerInfo(422.0f, 215.0f, 20) },
            { "rb_hold_static", new LayerInfo(422.0f, 215.0f, 21) },
            { "rb_hold_track", new LayerInfo(422.0f, 215.0f, 22) },
            { "rb_hold_prog", new LayerInfo(422.0f, 215.0f, 23) },
            { "rb_hold_done", new LayerInfo(422.0f, 215.0f, 24) },
            { "rb_hold_head", new LayerInfo(430.3f, 79.3f, 25) },
            { "rb_hold_ink", new LayerInfo(487.9f, 153.6f, 26) },
            { "rb_lower_fill", new LayerInfo(422.0f, 215.0f, 30) },
            { "rb_lower_static", new LayerInfo(422.0f, 215.0f, 31) },
            { "rb_lower_track", new LayerInfo(422.0f, 215.0f, 32) },
            { "rb_lower_prog", new LayerInfo(422.0f, 215.0f, 33) },
            { "rb_lower_done", new LayerInfo(422.0f, 215.0f, 34) },
            { "rb_lower_head", new LayerInfo(557.7f, 223.3f, 35) },
            { "rb_lower_ink", new LayerInfo(422.9f, 309.2f, 36) },
            { "rb_disc", new LayerInfo(422.0f, 215.0f, 40) },
            { "rb_pose_00", new LayerInfo(422.0f, 215.0f, 41) },
            { "rb_pose_01", new LayerInfo(422.0f, 215.0f, 41) },
            { "rb_pose_02", new LayerInfo(422.0f, 215.0f, 41) },
            { "rb_pose_03", new LayerInfo(422.0f, 215.0f, 41) },
            { "rb_pose_04", new LayerInfo(422.0f, 215.0f, 41) },
            { "rb_pose_05", new LayerInfo(422.0f, 215.0f, 41) },
            { "rb_pose_06", new LayerInfo(422.0f, 215.0f, 41) },
            { "rb_pose_07", new LayerInfo(422.0f, 215.0f, 41) },
            { "rb_pose_08", new LayerInfo(422.0f, 215.0f, 41) },
            { "rb_pose_09", new LayerInfo(422.0f, 215.0f, 41) },
            { "rb_pose_10", new LayerInfo(422.0f, 215.0f, 41) },
            { "rb_pose_11", new LayerInfo(422.0f, 215.0f, 41) },
            { "rb_pose_12", new LayerInfo(422.0f, 215.0f, 41) },
            { "rb_rays", new LayerInfo(424.0f, 195.0f, 42) },
            { "rb_hud_pause", new LayerInfo(40.0f, 30.0f, 60) },
        };

        /// <summary>Background sparkle twinkle timing, in the same order as rb_sparkle_N.</summary>
        public static readonly SparkleInfo[] Sparkles =
        {
            new SparkleInfo(-0.00f, 2.40f),
            new SparkleInfo(0.60f, 2.40f),
            new SparkleInfo(1.20f, 2.40f),
            new SparkleInfo(1.80f, 2.40f),
            new SparkleInfo(0.90f, 2.40f)
        };
    }
}
