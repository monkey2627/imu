using UnityEngine;

namespace RingSkinD
{
    /// <summary>
    /// 5 s LED charge for one quadrant: rows light up inner to outer every 1.25 s,
    /// the whole zone flashes once when full, then all rows release back to unlit.
    /// Diagonal quadrants share the same offset (pair A = melody + drums, pair B = harmony + bass, 3 s later).
    /// </summary>
    public class LedChargeQuadrant : MonoBehaviour
    {
        [Tooltip("LED rows, inner to outer")]
        public SpriteRenderer[] rows = new SpriteRenderer[4];
        public SpriteRenderer flash;

        public float period = 6f;
        public float offset = 0f;
        public float unlitAlpha = 0.18f;
        public float rowFadeIn = 0.15f;
        public float[] rowLitAt = { 1.25f, 2.5f, 3.75f, 5f };
        public float flashStart = 5.15f;
        public float flashRise = 0.1f;
        public float flashFade = 0.5f;
        public float releaseStart = 5.64f;

        void Update()
        {
            float t = Mathf.Repeat(DesignSpace.Now - offset, period);
            float release = t > releaseStart ? Mathf.Clamp01((t - releaseStart) / (period - releaseStart)) : 0f;

            for (int i = 0; i < rows.Length; i++)
            {
                float a = unlitAlpha;
                if (i < rowLitAt.Length && t >= rowLitAt[i])
                    a = Mathf.Lerp(unlitAlpha, 1f, Mathf.Clamp01((t - rowLitAt[i]) / rowFadeIn));
                if (release > 0f)
                    a = Mathf.Lerp(1f, unlitAlpha, release);
                SpriteAlpha.Set(rows[i], a);
            }

            if (flash != null)
            {
                float dt = t - flashStart;
                float f = 0f;
                if (dt >= 0f && dt < flashRise) f = dt / flashRise;
                else if (dt >= flashRise) f = 1f - Mathf.Clamp01((dt - flashRise) / flashFade);
                SpriteAlpha.Set(flash, f);
            }
        }
    }
}
