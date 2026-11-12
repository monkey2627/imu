using UnityEngine;

namespace RingSkinD
{
    /// <summary>Judge ring color-split copy that jitters briefly every 2 s. Cyan copy uses sign = -1.</summary>
    public class ChromaGlitch : MonoBehaviour
    {
        public float period = 2f;
        public float sign = 1f;

        static readonly float[] Keys = { 0f, 0.84f, 0.86f, 0.90f, 0.94f, 1f };
        static readonly Vector2[] OffsetsPx =
        {
            Vector2.zero, Vector2.zero, new Vector2(-5f, 1f), new Vector2(3f, -1f), new Vector2(-2f, 0f), Vector2.zero
        };

        Vector3 basePos;

        void Awake()
        {
            basePos = transform.localPosition;
        }

        void Update()
        {
            float p = Mathf.Repeat(DesignSpace.Now, period) / period;
            Vector2 v = Vector2.zero;
            for (int i = 0; i < Keys.Length - 1; i++)
            {
                if (p >= Keys[i] && p <= Keys[i + 1])
                {
                    v = Vector2.Lerp(OffsetsPx[i], OffsetsPx[i + 1], (p - Keys[i]) / (Keys[i + 1] - Keys[i]));
                    break;
                }
            }
            v *= sign;
            transform.localPosition = basePos + new Vector3(v.x * DesignSpace.PxToUnit, -v.y * DesignSpace.PxToUnit, 0f);
        }
    }
}
