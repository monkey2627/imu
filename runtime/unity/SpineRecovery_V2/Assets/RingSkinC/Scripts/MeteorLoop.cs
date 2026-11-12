using UnityEngine;

namespace RingSkinC
{
    /// <summary>
    /// Shooting star crossing once every 'period' seconds.
    /// Head anchor (240, 56) design px; travels from anchor + (120, -52) to anchor + (-90, 40) in 0.88 s.
    /// The sprite's pivot is its center, which sits (30, -12) design px from the meteor head.
    /// </summary>
    [RequireComponent(typeof(SpriteRenderer))]
    public class MeteorLoop : MonoBehaviour
    {
        public float period = 4f;
        public float delay = 2.4f;
        public float duration = 0.88f;
        public float fadeIn = 0.32f;
        public Vector2 headAnchorPx = new Vector2(240f, 56f);
        public Vector2 fromOffsetPx = new Vector2(120f, -52f);
        public Vector2 toOffsetPx = new Vector2(-90f, 40f);
        public Vector2 headToSpriteCenterPx = new Vector2(30f, -12f);

        SpriteRenderer sr;

        void Awake()
        {
            sr = GetComponent<SpriteRenderer>();
        }

        void Update()
        {
            float t = Mathf.Repeat(DesignSpace.Now - delay, period);
            if (t > duration)
            {
                SpriteAlpha.Set(sr, 0f);
                return;
            }
            float p = Ease.OutQuad(t / duration);
            Vector2 head = headAnchorPx + Vector2.Lerp(fromOffsetPx, toOffsetPx, p);
            Vector2 center = head + headToSpriteCenterPx;
            transform.localPosition = DesignSpace.ToWorld(center.x, center.y);
            float a = t < fadeIn ? t / fadeIn : 1f - (t - fadeIn) / (duration - fadeIn);
            SpriteAlpha.Set(sr, a);
        }
    }
}
