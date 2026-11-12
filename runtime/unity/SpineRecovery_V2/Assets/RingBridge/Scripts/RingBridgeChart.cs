using UnityEngine;

namespace RingBridge
{
    /// <summary>One ring section (抬起 / 保持 / 放下) and the renderers that animate with it.</summary>
    [System.Serializable]
    public class BridgeSection
    {
        public string name;
        [Tooltip("Active window inside the loop (s)")]
        public float start;
        public float end;
        [Tooltip("Outer progress arc from this SVG angle (deg, clockwise from +x), sweeping clockwise")]
        public float startAngle;
        public float sweep;

        public SpriteRenderer fill;
        public SpriteRenderer track;
        public SpriteRenderer progress;
        public Transform progressMask;
        public SpriteRenderer done;
        public Transform headPivot;
        public SpriteRenderer head;
        public SpriteRenderer ink;
    }

    /// <summary>
    /// Demo driver for 圆环 C · 臀桥三段 (7 s loop):
    ///   0.5-2.0 s 抬起 · 2.0-5.0 s 保持 · 5.0-6.5 s 放下 · then 1 s rest.
    /// While a section is active its outer arc fills clockwise (revealed by a rotating half-disc SpriteMask),
    /// a glowing head rides the arc, and the whole section brightens from dark to bright in step with the progress.
    /// When it completes: arc flash, small bump, then it dims in 0.4 s.
    /// The centre figure raises its hips during 抬起, holds, and lowers during 放下 (13 pose frames).
    ///
    /// Production: set autoPlay = false and call SetProgress(sectionIndex, 0..1) / Complete(sectionIndex)
    /// from the exercise tracking, or replace DesignSpace.Now with your own clock.
    /// </summary>
    public class RingBridgeChart : MonoBehaviour
    {
        public bool autoPlay = true;
        public float loopLength = 7f;
        public BridgeSection[] sections = new BridgeSection[3];

        [Header("Centre")]
        public Transform disc;
        public SpriteRenderer pose;
        public Sprite[] poseFrames;
        public SpriteRenderer rays;

        [Header("HUD")]
        public Transform combo;

        const float FillRest = 0.16f, FillPeak = 0.72f;
        const float TrackRest = 0.25f, TrackPeak = 0.8f;
        const float InkRest = 0.45f;
        const float Dim = 0.4f;
        MaterialPropertyBlock progressProperties;

        void Update()
        {
            if (!autoPlay) return;
            float now = DesignSpace.Now;
            if (combo != null)
            {
                float c = Mathf.Repeat(now, 0.5f) / 0.5f;
                float s = c < 0.4f ? Mathf.Lerp(1.18f, 1f, Ease.OutQuad(c / 0.4f)) : 1f;
                combo.localScale = new Vector3(s, s, 1f);
            }

            float t = Mathf.Repeat(now, loopLength);
            float beat = 1f;
            for (int i = 0; i < sections.Length; i++)
            {
                BridgeSection sec = sections[i];
                if (sec == null) continue;
                float len = sec.end - sec.start;
                float d = t - sec.start;
                if (d < 0f) d += loopLength;
                float u = -1f;          // progress 0..1 while active
                float after = -1f;      // seconds since completion
                if (d <= len) u = d / len;
                else after = d - len;
                Apply(sec, u, after);
                if (after >= 0f && after < 0.35f)
                    beat = Mathf.Max(beat, after < 0.06f ? Mathf.Lerp(1f, 1.06f, after / 0.06f) : Mathf.Lerp(1.06f, 1f, Ease.OutQuad((after - 0.06f) / 0.29f)));
            }
            if (disc != null) disc.localScale = new Vector3(beat, beat, 1f);
            UpdateCentre(t);
        }

        /// <summary>Manual control: show section i at progress u (0..1). Pass u &lt; 0 for idle.</summary>
        public void SetProgress(int i, float u)
        {
            if (i >= 0 && i < sections.Length) Apply(sections[i], u < 0 ? -1f : Mathf.Clamp01(u), -1f);
        }

        /// <summary>One deterministic frame, including centre animation and completion effects.</summary>
        public void RenderRecorded(HipThrustPhaseTimeline.Sample sample)
        {
            autoPlay = false;
            float bump = 1f;
            for (int i = 0; i < 3; i++)
            {
                float age = sample.completedSeconds[i];
                Apply(sections[i], sample.phase == i ? sample.progress : -1f, sample.phase == i ? -1f : age);
                if (age < .35f) bump = Mathf.Max(bump, age < .06f ? Mathf.Lerp(1f, 1.06f, age/.06f) : Mathf.Lerp(1.06f, 1f, Ease.OutQuad((age-.06f)/.29f)));
            }
            if (disc != null) disc.localScale = new Vector3(bump, bump, 1);
            if (pose != null && poseFrames != null && poseFrames.Length > 0)
                pose.sprite = poseFrames[Mathf.RoundToInt(Mathf.Clamp01(sample.poseHeight) * (poseFrames.Length-1))];
            float rayAlpha = sample.phase == 1 && sample.holdSeconds >= .1f ? .675f + .325f * Mathf.Cos((sample.holdSeconds-.1f) * 4f * Mathf.PI) : 0;
            SpriteAlpha.Set(rays, rayAlpha);
        }

        /// <summary>Manual control: play the completion state of section i, 'secondsSince' after it finished.</summary>
        public void Complete(int i, float secondsSince)
        {
            if (i >= 0 && i < sections.Length) Apply(sections[i], -1f, Mathf.Max(0f, secondsSince));
        }

        void Apply(BridgeSection sec, float u, float after)
        {
            float level;   // 0 = dark, 1 = fully lit
            if (u >= 0f) level = u;
            else if (after >= 0f && after < Dim) level = 1f - after / Dim;
            else level = 0f;

            SpriteAlpha.Set(sec.fill, Mathf.Lerp(FillRest, FillPeak, level));
            SpriteAlpha.Set(sec.track, Mathf.Lerp(TrackRest, TrackPeak, level));

            // Outer progress arc + riding head
            float p = u >= 0f ? u : 1f;
            if (sec.progress != null)
            {
                if (progressProperties == null) progressProperties = new MaterialPropertyBlock();
                sec.progress.GetPropertyBlock(progressProperties);
                progressProperties.SetTexture("_MainTex", sec.progress.sprite.texture);
                progressProperties.SetFloat("_UseArc", 1);
                progressProperties.SetFloat("_ArcStart", sec.startAngle);
                progressProperties.SetFloat("_ArcSweep", sec.sweep);
                progressProperties.SetFloat("_Progress", p);
                sec.progress.SetPropertyBlock(progressProperties);
            }
            float progAlpha = u >= 0f ? 1f : (after >= 0f && after < 0.45f ? 1f - after / 0.45f : 0f);
            SpriteAlpha.Set(sec.progress, progAlpha);
            float theta = sec.startAngle + p * sec.sweep;           // SVG clockwise angle of the leading edge
            if (sec.progressMask != null) sec.progressMask.localRotation = Quaternion.Euler(0f, 0f, -theta);
            if (sec.headPivot != null) sec.headPivot.localRotation = Quaternion.Euler(0f, 0f, -p * sec.sweep);
            float headAlpha = u >= 0f ? Mathf.Clamp01(u * (sec.end - sec.start) / 0.1f) : (after >= 0f && after < 0.3f ? 1f - after / 0.3f : 0f);
            SpriteAlpha.Set(sec.head, headAlpha);

            // Completion flash
            float flash = 0f;
            if (after >= 0f && after < 0.5f) flash = after < 0.04f ? after / 0.04f : 1f - (after - 0.04f) / 0.46f;
            SpriteAlpha.Set(sec.done, flash);

            // Figure + label: brighten with progress, bump on completion
            if (sec.ink != null)
            {
                SpriteAlpha.Set(sec.ink, Mathf.Lerp(InkRest, 1f, level));
                float s = 1f;
                if (after >= 0f && after < 0.4f)
                    s = after < 0.12f ? Mathf.Lerp(1f, 1.08f, after / 0.12f) : Mathf.Lerp(1.08f, 1f, (after - 0.12f) / 0.28f);
                sec.ink.transform.localScale = new Vector3(s, s, 1f);
            }
        }

        void UpdateCentre(float t)
        {
            if (sections.Length < 3 || sections[0] == null || sections[2] == null) return;
            BridgeSection lift = sections[0], hold = sections[1], lower = sections[2];
            float k;
            if (t < lift.start) k = 0f;
            else if (t < lift.end) k = Ease.InOutSine((t - lift.start) / (lift.end - lift.start));
            else if (t < lower.start) k = 1f;
            else if (t < lower.end) k = 1f - Ease.InOutSine((t - lower.start) / (lower.end - lower.start));
            else k = 0f;
            if (pose != null && poseFrames != null && poseFrames.Length > 0)
                pose.sprite = poseFrames[Mathf.Clamp(Mathf.RoundToInt(k * (poseFrames.Length - 1)), 0, poseFrames.Length - 1)];

            if (rays != null)
            {
                float a = 0f;
                if (hold != null && t >= hold.start + 0.1f && t < hold.end)
                {
                    float c = Mathf.Repeat(t - hold.start - 0.1f, 0.5f) / 0.5f;
                    a = 0.675f + 0.325f * Mathf.Cos(c * 2f * Mathf.PI);
                }
                SpriteAlpha.Set(rays, a);
            }
        }
    }
}
