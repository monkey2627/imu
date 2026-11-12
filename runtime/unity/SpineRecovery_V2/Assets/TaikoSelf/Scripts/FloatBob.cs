using UnityEngine;

namespace TaikoSelf
{
    /// <summary>Gentle up-and-down float (sky lanterns): rises 'amplitudePx' design px and returns, ease in-out.</summary>
    public class FloatBob : MonoBehaviour
    {
        public float period = 6f;
        public float amplitudePx = 8f;
        public float phase = 0f;

        Vector3 basePos;

        void Awake()
        {
            basePos = transform.localPosition;
        }

        void Update()
        {
            float p = Mathf.Repeat(DesignSpace.Now + phase, period) / period;
            float k = p < 0.5f ? Ease.InOutSine(p * 2f) : Ease.InOutSine((1f - p) * 2f);
            transform.localPosition = basePos + new Vector3(0f, amplitudePx * k * DesignSpace.PxToUnit, 0f);
        }
    }
}
