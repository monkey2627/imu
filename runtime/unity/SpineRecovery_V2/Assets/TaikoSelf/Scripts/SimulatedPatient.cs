using UnityEngine;

namespace TaikoSelf
{
    /// <summary>
    /// Demo input for <see cref="SelfPacedDrum"/>: plays one ideal repetition on a loop, the same curve as the design
    /// board (lift 1.2 s, hold 2.8 s, slow lowering 5 s, rest). Replace with the real sensor: disable this component
    /// and write SelfPacedDrum.height from your IMU / pose tracking each frame.
    /// </summary>
    [RequireComponent(typeof(SelfPacedDrum))]
    public class SimulatedPatient : MonoBehaviour
    {
        public float restBefore = 0.4f;
        public float liftTime = 1.2f;
        public float holdTime = 2.8f;
        public float lowerTime = 5f;
        public float restAfter = 1.2f;

        SelfPacedDrum drum;

        void Awake()
        {
            drum = GetComponent<SelfPacedDrum>();
        }

        void Update()
        {
            float loop = restBefore + liftTime + holdTime + lowerTime + restAfter;
            float t = Mathf.Repeat(DesignSpace.Now, loop) - restBefore;
            float h = 0f;
            if (t >= 0f && t < liftTime) h = Ease.InOutSine(t / liftTime);
            else if ((t -= liftTime) >= 0f && t < holdTime) h = 1f;
            else if ((t -= holdTime) >= 0f && t < lowerTime) h = 1f - Ease.InOutSine(t / lowerTime);
            drum.height = h;
        }
    }
}
