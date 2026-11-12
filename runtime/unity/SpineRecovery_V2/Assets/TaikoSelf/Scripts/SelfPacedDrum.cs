using UnityEngine;

namespace TaikoSelf
{
    /// <summary>
    /// 敲鼓 · 正面鼓面 · 自主节奏 — patient-driven 抬起 / 保持 / 慢慢放下 with no preset timing.
    ///
    /// Feed <see cref="height"/> every frame with the normalised hip height (0 = flat, 1 = calibrated top),
    /// e.g. from the IMU. A small state machine detects the phases; every visual is mapped from the height and
    /// from how long the current phase has lasted:
    ///   抬起 (pink)  fill and progress arc follow the height
    ///   保持 (gold)  six stars light one every 0.4 s for as long as the hold lasts; the block brightens with them
    ///   慢慢放下 (lilac) a dot with a comet trail rides the arc with the height; the block brightens with the
    ///                same progress; notes light and star dust falls as the dot passes them
    /// The three blocks never overlap: the previous one is fully dark (gap) before the next starts to glow.
    /// A return that takes longer than the lift lights one more star in the constellation.
    /// </summary>
    public class SelfPacedDrum : MonoBehaviour
    {
        public enum Phase { Rest, Lift, Hold, Return }

        [Header("Input (0 = flat, 1 = top)")]
        [Range(0f, 1f)] public float height;

        [Header("Phase detection")]
        public float startThreshold = 0.05f;
        public float topThreshold = 0.92f;
        public float releaseDrop = 0.02f;
        public float restThreshold = 0.04f;
        [Tooltip("Previous block fades out completely before the next one starts (s)")]
        public float gap = 0.12f;

        [Header("抬起")]
        public SpriteRenderer liftFill;
        public SpriteRenderer liftProgress;
        public Transform liftMask;
        public SpriteRenderer captionLift;

        [Header("保持")]
        public SpriteRenderer holdFill;
        public SpriteRenderer[] stars;
        public SpriteRenderer coreGlow;
        public SpriteRenderer captionHold;

        [Header("慢慢放下")]
        public SpriteRenderer returnFill;
        public SpriteRenderer comet;
        public Transform dotPivot;
        public SpriteRenderer dot;
        public SpriteRenderer[] notes;
        public SpriteRenderer[] noteRings;
        public SpriteRenderer[] dust;
        public SpriteRenderer captionReturn;

        [Header("Centre / reward")]
        public Transform core;
        public SpriteRenderer constellation;

        public Phase CurrentPhase { get { return phase; } }

        Phase phase = Phase.Rest;
        float phaseStart = -10f;
        float liftStart, liftEnd, returnStart, returnEnd = -10f;
        float lastLift, lastHold, lastReturn, holdPeak, returnFrom = 1f;
        bool rewarded;
        float[] noteHit, dustHit;
        Vector3[] dustBase;
        float corePulse = -10f;

        void Start()
        {
            noteHit = new float[notes != null ? notes.Length : 0];
            dustHit = new float[dust != null ? dust.Length : 0];
            dustBase = new Vector3[dustHit.Length];
            for (int i = 0; i < dustHit.Length; i++) dustBase[i] = dust[i].transform.localPosition;
            ResetRep();
        }

        void ResetRep()
        {
            for (int i = 0; i < noteHit.Length; i++) noteHit[i] = -10f;
            for (int i = 0; i < dustHit.Length; i++) dustHit[i] = -10f;
        }

        void Update()
        {
            float now = DesignSpace.Now;
            float h = Mathf.Clamp01(height);
            Detect(now, h);
            float since = now - phaseStart;

            // ---------------- 抬起
            float liftLevel = 0f;
            if (phase == Phase.Lift) { liftLevel = h; lastLift = h; }
            else if (now - liftEnd < gap) liftLevel = lastLift * (1f - (now - liftEnd) / gap);
            SpriteAlpha.Set(liftFill, 0.95f * liftLevel);
            SpriteAlpha.Set(liftProgress, phase == Phase.Lift ? 1f : (lastLift > 0f ? liftLevel / lastLift : 0f));
            if (liftMask != null)
            {
                float p = phase == Phase.Lift ? h : lastLift;
                liftMask.localRotation = Quaternion.Euler(0f, 0f, -(SelfLayout.LiftStart + p * SelfLayout.LiftSweep));
            }

            // ---------------- 保持
            float holdLevel = 0f;
            if (phase == Phase.Hold)
            {
                holdLevel = since < gap ? 0f : 0.95f * Mathf.Clamp01((since - gap) / 2.55f);
                lastHold = holdLevel;
            }
            else if (phase == Phase.Return && since < gap) holdLevel = lastHold * (1f - since / gap);
            SpriteAlpha.Set(holdFill, holdLevel);
            float holdFade = phase == Phase.Hold ? 1f : (phase == Phase.Return && since < gap ? 1f - since / gap : 0f);
            if (stars != null)
            {
                float holdTime = phase == Phase.Hold ? since : (phase == Phase.Return ? returnStart - liftEnd : 0f);
                for (int k = 0; k < stars.Length; k++)
                {
                    float tk = 0.35f + 0.4f * k;
                    float d = holdTime - tk;
                    float a = d < 0f ? 0f : Mathf.Clamp01(d / 0.12f);
                    float s = d < 0f ? 0.4f : (d < 0.12f ? Mathf.Lerp(0.4f, 1.35f, d / 0.12f) : Mathf.Lerp(1.35f, 1f, Mathf.Clamp01((d - 0.12f) / 0.18f)));
                    SpriteAlpha.Set(stars[k], a * holdFade);
                    stars[k].transform.localScale = new Vector3(s, s, 1f);
                }
            }
            SpriteAlpha.Set(coreGlow, phase == Phase.Hold ? Mathf.Min(0.9f, 0.3f + since * 0.3f) : 0f);

            // ---------------- 慢慢放下
            float retK = 0f, progress = 0f;
            if (phase == Phase.Return)
            {
                retK = since < gap ? 0f : Mathf.Clamp01((since - gap) / 0.15f);
                progress = Mathf.Clamp01(1f - h / Mathf.Max(0.01f, returnFrom));
                lastReturn = progress;
            }
            else if (phase == Phase.Rest && now - returnEnd < gap)
            {
                retK = 1f - (now - returnEnd) / gap;
                progress = lastReturn;
            }
            SpriteAlpha.Set(returnFill, retK * (0.05f + 0.9f * progress));
            float sweep = progress * SelfLayout.ReturnSweep;
            if (comet != null)
            {
                comet.transform.localRotation = Quaternion.Euler(0f, 0f, -sweep);
                SpriteAlpha.Set(comet, retK);
            }
            if (dotPivot != null) dotPivot.localRotation = Quaternion.Euler(0f, 0f, -sweep);
            SpriteAlpha.Set(dot, retK);

            float dotAngle = SelfLayout.ReturnStart + sweep;
            if (phase == Phase.Return && retK > 0f)
            {
                for (int i = 0; i < noteHit.Length; i++)
                    if (noteHit[i] < 0f && dotAngle >= SelfLayout.NoteAngles[i]) noteHit[i] = now;
                for (int i = 0; i < dustHit.Length; i++)
                    if (dustHit[i] < 0f && dotAngle >= SelfLayout.DustAngles[i]) dustHit[i] = now;
            }
            UpdateNotes(now);
            UpdateDust(now);

            // ---------------- captions, centre, reward
            Caption(captionLift, Phase.Lift, since);
            Caption(captionHold, Phase.Hold, since);
            Caption(captionReturn, Phase.Return, since);

            if (core != null)
            {
                float d = now - corePulse;
                float s = d < 0f || d > 0.35f ? 1f : (d < 0.06f ? Mathf.Lerp(1f, 1.1f, d / 0.06f) : Mathf.Lerp(1.1f, 1f, (d - 0.06f) / 0.29f));
                core.localScale = new Vector3(s, s, 1f);
            }
            if (constellation != null)
            {
                float a = 0f;
                if (rewarded)
                {
                    float d = now - returnEnd;
                    a = Mathf.Clamp01((d - 0.2f) / 0.25f);
                }
                else if (phase == Phase.Lift && since < 0.3f) a = constellation.color.a * (1f - since / 0.3f);
                SpriteAlpha.Set(constellation, a);
            }
        }

        void Detect(float now, float h)
        {
            switch (phase)
            {
                case Phase.Rest:
                    if (h > startThreshold)
                    {
                        Enter(Phase.Lift, now);
                        liftStart = now;
                        rewarded = false;
                        ResetRep();
                    }
                    break;
                case Phase.Lift:
                    if (h >= topThreshold)
                    {
                        liftEnd = now;
                        holdPeak = h;
                        corePulse = now;
                        Enter(Phase.Hold, now);
                    }
                    else if (h < restThreshold)
                    {
                        liftEnd = now;
                        Enter(Phase.Rest, now);
                    }
                    break;
                case Phase.Hold:
                    holdPeak = Mathf.Max(holdPeak, h);
                    if (h < holdPeak - releaseDrop)
                    {
                        returnStart = now;
                        returnFrom = holdPeak;
                        Enter(Phase.Return, now);
                    }
                    break;
                case Phase.Return:
                    if (h <= restThreshold)
                    {
                        returnEnd = now;
                        // Self-referenced reward: lowering took longer than lifting
                        rewarded = (returnEnd - returnStart) > (liftEnd - liftStart);
                        corePulse = now + 0.5f;
                        Enter(Phase.Rest, now);
                    }
                    break;
            }
        }

        void Enter(Phase p, float now)
        {
            phase = p;
            phaseStart = now;
        }

        void Caption(SpriteRenderer r, Phase p, float since)
        {
            if (r == null) return;
            float target = phase == p && since >= gap ? 1f : 0f;
            float a = r.color.a;
            a = target > a ? Mathf.Min(target, a + Time.deltaTime / 0.12f) : Mathf.Max(target, a - Time.deltaTime / 0.1f);
            SpriteAlpha.Set(r, a);
        }

        void UpdateNotes(float now)
        {
            for (int i = 0; i < noteHit.Length; i++)
            {
                float a = 0.25f, s = 1f, ring = 0f, rs = 0.6f;
                if (noteHit[i] >= 0f)
                {
                    float d = now - noteHit[i];
                    if (d < 0.08f) { a = Mathf.Lerp(0.25f, 1f, d / 0.08f); s = Mathf.Lerp(1f, 1.6f, d / 0.08f); }
                    else if (d < 0.5f) { a = Mathf.Lerp(1f, 0.85f, (d - 0.08f) / 0.42f); s = Mathf.Lerp(1.6f, 1f, (d - 0.08f) / 0.42f); }
                    else a = 0.85f;
                    if (d < 0.5f) { ring = 0.9f * (1f - d / 0.5f); rs = Mathf.Lerp(0.6f, 2f, Ease.OutQuad(d / 0.5f)); }
                    if (phase == Phase.Rest)
                    {
                        float r = now - returnEnd;
                        if (r > 0.2f) a = Mathf.Lerp(0.85f, 0.25f, Mathf.Clamp01((r - 0.2f) / 0.4f));
                    }
                }
                if (notes != null && i < notes.Length)
                {
                    SpriteAlpha.Set(notes[i], a);
                    notes[i].transform.localScale = new Vector3(s, s, 1f);
                }
                if (noteRings != null && i < noteRings.Length)
                {
                    SpriteAlpha.Set(noteRings[i], ring);
                    noteRings[i].transform.localScale = new Vector3(rs, rs, 1f);
                }
            }
        }

        void UpdateDust(float now)
        {
            for (int i = 0; i < dustHit.Length; i++)
            {
                float d = dustHit[i] < 0f ? -1f : now - dustHit[i];
                if (d < 0f || d > 1.3f)
                {
                    SpriteAlpha.Set(dust[i], 0f);
                    continue;
                }
                float a = d < 0.05f ? d / 0.05f : 1f - (d - 0.05f) / 1.25f;
                float y = Mathf.Lerp(2f, 26f, Ease.OutQuad(d / 1.3f));
                SpriteAlpha.Set(dust[i], a);
                dust[i].transform.localPosition = dustBase[i] + new Vector3(0f, -y * DesignSpace.PxToUnit, 0f);
            }
        }
    }
}
