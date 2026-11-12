using UnityEngine;

namespace RingBridge
{
    /// <summary>Breathing star: alpha 0.2 &lt;-&gt; 1 and scale 0.6 &lt;-&gt; 1.1, ease in-out, looping.</summary>
    [RequireComponent(typeof(SpriteRenderer))]
    public class Twinkle : MonoBehaviour
    {
        public float period = 2.4f;
        public float phase = 0f;
        public float minAlpha = 0.2f;
        public float maxAlpha = 1f;
        public float minScale = 0.6f;
        public float maxScale = 1.1f;

        SpriteRenderer sr;
        Vector3 baseScale;

        void Awake()
        {
            sr = GetComponent<SpriteRenderer>();
            baseScale = transform.localScale;
        }

        void Update()
        {
            float p = Mathf.Repeat(DesignSpace.Now + phase, period) / period;
            // 0 -> 0.5 rises, 0.5 -> 1 falls
            float k = p < 0.5f ? Ease.InOutSine(p * 2f) : Ease.InOutSine((1f - p) * 2f);
            SpriteAlpha.Set(sr, Mathf.Lerp(minAlpha, maxAlpha, k));
            float s = Mathf.Lerp(minScale, maxScale, k);
            transform.localScale = new Vector3(baseScale.x * s, baseScale.y * s, 1f);
        }
    }
}
