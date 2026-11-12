using UnityEngine;

namespace RingSkinD
{
    /// <summary>On every beat the scale jumps to 'peak' and eases back to 1 (combo / sync label).</summary>
    public class BeatBump : MonoBehaviour
    {
        public float bpm = 120f;
        public float peak = 1.18f;
        [Range(0.05f, 1f)] public float fractionOfBeat = 0.4f;

        void Update()
        {
            float beat = 60f / bpm;
            float p = Mathf.Repeat(DesignSpace.Now, beat) / (beat * fractionOfBeat);
            float s = Mathf.Lerp(peak, 1f, Ease.OutQuad(p));
            transform.localScale = new Vector3(s, s, 1f);
        }
    }
}
