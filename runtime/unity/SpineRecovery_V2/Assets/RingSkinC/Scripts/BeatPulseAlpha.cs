using UnityEngine;

namespace RingSkinC
{
    /// <summary>On every beat the sprite alpha jumps to 'from' and eases out to 'to'.</summary>
    [RequireComponent(typeof(SpriteRenderer))]
    public class BeatPulseAlpha : MonoBehaviour
    {
        public float bpm = 120f;
        public float from = 1f;
        public float to = 0.15f;

        SpriteRenderer sr;

        void Awake()
        {
            sr = GetComponent<SpriteRenderer>();
        }

        void Update()
        {
            float beat = 60f / bpm;
            float p = Mathf.Repeat(DesignSpace.Now, beat) / beat;
            SpriteAlpha.Set(sr, Mathf.Lerp(from, to, Ease.OutQuad(p)));
        }
    }
}
