using UnityEngine;

namespace RingSkinD
{
    /// <summary>Horizontal floor lines rushing from the horizon (y 292) to the bottom edge (y 390), fading in with depth.</summary>
    public class FloorLines : MonoBehaviour
    {
        public SpriteRenderer[] lines;
        public float period = 2f;
        public float stagger = 0.4f;
        public float fromY = 292f;
        public float toY = 390f;
        public float groupAlpha = 0.55f;

        void Update()
        {
            if (lines == null) return;
            for (int i = 0; i < lines.Length; i++)
            {
                SpriteRenderer l = lines[i];
                if (l == null) continue;
                float p = Mathf.Repeat(DesignSpace.Now + i * stagger, period) / period;
                float y = Mathf.Lerp(fromY, toY, Ease.InCubic(p));
                float lineAlpha = p < 0.2f ? Mathf.Lerp(0f, 0.9f, p / 0.2f) : Mathf.Lerp(0.9f, 1f, (p - 0.2f) / 0.8f);
                float depthMask = Mathf.Clamp01((y - 290f) / 100f);
                l.transform.localPosition = DesignSpace.ToWorld(DesignSpace.CenterX, y);
                SpriteAlpha.Set(l, groupAlpha * depthMask * lineAlpha);
            }
        }
    }
}
