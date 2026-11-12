using UnityEngine;

namespace RingSkinD
{
    /// <summary>
    /// Converts design-space pixels (844x390, origin top-left, y down) to world units.
    /// 1 design px = 0.01 unit; screen center (422, 195) maps to world (0, 0).
    /// </summary>
    public static class DesignSpace
    {
        public const float PxToUnit = 0.01f;
        public const float CenterX = 422f;
        public const float CenterY = 195f;

        public static Vector3 ToWorld(float x, float y)
        {
            return new Vector3((x - CenterX) * PxToUnit, -(y - CenterY) * PxToUnit, 0f);
        }

        /// <summary>Ring center in design px is (422, 215).</summary>
        public static Vector3 RingCenter
        {
            get { return ToWorld(422f, 215f); }
        }

        /// <summary>Time source for all demo animations. Replace with music time in production.</summary>
        public static float Now
        {
            get { return Time.timeSinceLevelLoad; }
        }
    }

    public static class Ease
    {
        public static float OutQuad(float t)
        {
            t = Mathf.Clamp01(t);
            return 1f - (1f - t) * (1f - t);
        }

        public static float InCubic(float t)
        {
            t = Mathf.Clamp01(t);
            return t * t * t;
        }
    }

    public static class SpriteAlpha
    {
        public static void Set(SpriteRenderer r, float a)
        {
            if (r == null) return;
            Color c = r.color;
            c.a = a;
            r.color = c;
        }
    }
}
