using UnityEngine;

namespace RingSkinD
{
    /// <summary>Rotates clockwise at a constant speed (outer tick ring).</summary>
    public class SpinForever : MonoBehaviour
    {
        public float secondsPerTurn = 40f;

        void Update()
        {
            float turns = Mathf.Repeat(DesignSpace.Now / secondsPerTurn, 1f);
            transform.localRotation = Quaternion.Euler(0f, 0f, -360f * turns);
        }
    }
}
