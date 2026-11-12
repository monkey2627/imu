using UnityEngine;

namespace RingSkinD
{
    /// <summary>Square-wave blink: full alpha for the first half of the period, dim for the second half.</summary>
    [RequireComponent(typeof(SpriteRenderer))]
    public class BlinkLight : MonoBehaviour
    {
        public float period = 1f;
        public float phase = 0f;
        public float dimAlpha = 0.15f;

        SpriteRenderer sr;

        void Awake()
        {
            sr = GetComponent<SpriteRenderer>();
        }

        void Update()
        {
            float p = Mathf.Repeat(DesignSpace.Now + phase, period) / period;
            SpriteAlpha.Set(sr, p < 0.5f ? 1f : dimAlpha);
        }
    }
}
