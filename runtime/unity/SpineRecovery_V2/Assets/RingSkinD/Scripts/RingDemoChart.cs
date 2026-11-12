using System;
using System.Collections.Generic;
using UnityEngine;

namespace RingSkinD
{
    /// <summary>
    /// Demo chart that loops every 4 s at 120 BPM:
    /// melody 0.5 s, harmony 1.0 s, bass 1.5 s, drums 2.0 s, center accent 3.0 s.
    /// Spawns notes, center cue, hit bursts, judgement text, and pulses the core.
    /// Replace the event table and DesignSpace.Now with real chart data / music time in production.
    /// </summary>
    public class RingDemoChart : MonoBehaviour
    {
        // Zone index order everywhere: 0 melody (UL), 1 harmony (UR), 2 bass (LL), 3 drums (LR)
        [Header("Timing")]
        public float loopLength = 4f;
        public float flightTime = 1.2f;
        public float fadeIn = 0.24f;

        [Header("Geometry (design px)")]
        public float spawnRadius = 330f;
        public float hitRadius = 146f;

        [Header("Sprites")]
        public Sprite[] noteBody = new Sprite[4];
        public Sprite[] noteGlow = new Sprite[4];
        public Sprite[] noteTrail = new Sprite[4];
        public Sprite[] hitBurst = new Sprite[4];
        public Sprite centerCue;
        public Sprite centerWave;
        public Sprite textPerfect;
        public Sprite textPerfectCenter;

        [Header("Scene refs")]
        public Transform core;

        [Header("Sorting orders")]
        public int orderCue = 50;
        public int orderTrail = 59;
        public int orderNoteGlow = 60;
        public int orderNoteBody = 61;
        public int orderHit = 70;
        public int orderText = 80;

        // Unity angles (deg, CCW from +x) of each zone's 45-degree bisector
        static readonly float[] ZoneAngle = { 135f, 45f, 225f, 315f };
        // Judgement text centers in design px
        static readonly Vector2[] TextPosPx =
        {
            new Vector2(252f, 79f), new Vector2(594f, 79f), new Vector2(252f, 351f), new Vector2(594f, 351f)
        };
        static readonly Vector2 CenterTextPosPx = new Vector2(422f, 121f);

        struct ChartEvent
        {
            public float hit;
            public int zone; // -1 = center
            public ChartEvent(float hit, int zone) { this.hit = hit; this.zone = zone; }
        }

        static readonly ChartEvent[] Events =
        {
            new ChartEvent(0.5f, 0), new ChartEvent(1.0f, 1), new ChartEvent(1.5f, 2),
            new ChartEvent(2.0f, 3), new ChartEvent(3.0f, -1)
        };

        class Fx
        {
            public GameObject go;
            public Func<float, bool> tick; // returns false when finished
        }

        readonly List<Fx> active = new List<Fx>();
        int loop;
        int idx;
        float corePulseStart = -100f;

        void Update()
        {
            float now = DesignSpace.Now;

            // Spawn every event whose flight has started
            int guard = 0;
            while (guard++ < 32)
            {
                ChartEvent e = Events[idx];
                float hitTime = loop * loopLength + e.hit;
                if (hitTime - flightTime > now) break;
                if (e.zone >= 0) SpawnNote(e.zone, hitTime);
                else SpawnCenterCue(hitTime);
                idx++;
                if (idx >= Events.Length) { idx = 0; loop++; }
            }

            for (int i = active.Count - 1; i >= 0; i--)
            {
                if (!active[i].tick(now))
                {
                    if (active[i].go != null) Destroy(active[i].go);
                    active.RemoveAt(i);
                }
            }

            UpdateCorePulse(now);
        }

        // ------------------------------------------------------------------ notes

        void SpawnNote(int zone, float hitTime)
        {
            GameObject root = new GameObject("Note_" + zone);
            root.transform.SetParent(transform, false);
            float ang = ZoneAngle[zone];
            Vector3 dir = new Vector3(Mathf.Cos(ang * Mathf.Deg2Rad), Mathf.Sin(ang * Mathf.Deg2Rad), 0f);

            SpriteRenderer trail = MakeSprite("Trail", noteTrail[zone], orderTrail, root.transform);
            trail.transform.localRotation = Quaternion.Euler(0f, 0f, ang);
            SpriteRenderer glow = MakeSprite("Glow", noteGlow[zone], orderNoteGlow, root.transform);
            SpriteRenderer body = MakeSprite("Body", noteBody[zone], orderNoteBody, root.transform);
            float spawnTime = hitTime - flightTime;

            Track(root, now =>
            {
                if (now >= hitTime)
                {
                    OnZoneHit(zone, hitTime);
                    return false;
                }
                float p = Mathf.Clamp01((now - spawnTime) / flightTime);
                float r = Mathf.Lerp(spawnRadius, hitRadius, p) * DesignSpace.PxToUnit;
                root.transform.localPosition = DesignSpace.RingCenter + dir * r;
                float a = Mathf.Clamp01((now - spawnTime) / fadeIn);
                SpriteAlpha.Set(trail, a);
                SpriteAlpha.Set(glow, a);
                SpriteAlpha.Set(body, a);
                return true;
            });
        }

        void OnZoneHit(int zone, float hitTime)
        {
            float ang = ZoneAngle[zone] * Mathf.Deg2Rad;
            Vector3 hitPos = DesignSpace.RingCenter + new Vector3(Mathf.Cos(ang), Mathf.Sin(ang), 0f) * hitRadius * DesignSpace.PxToUnit;
            SpawnBurst(hitBurst[zone], hitPos, hitTime);
            SpawnText(textPerfect, DesignSpace.ToWorld(TextPosPx[zone].x, TextPosPx[zone].y), hitTime);
        }

        // ------------------------------------------------------------------ center

        void SpawnCenterCue(float hitTime)
        {
            SpriteRenderer cue = MakeSprite("CenterCue", centerCue, orderCue, transform);
            cue.transform.localPosition = DesignSpace.RingCenter;
            float spawnTime = hitTime - flightTime;

            Track(cue.gameObject, now =>
            {
                if (now >= hitTime)
                {
                    SpawnBurst(centerWave, DesignSpace.RingCenter, hitTime);
                    SpawnText(textPerfectCenter, DesignSpace.ToWorld(CenterTextPosPx.x, CenterTextPosPx.y), hitTime);
                    corePulseStart = hitTime;
                    return false;
                }
                float e = now - spawnTime;
                float s = Mathf.Lerp(2.4f, 0.6f, Mathf.Clamp01(e / flightTime));
                cue.transform.localScale = new Vector3(s, s, 1f);
                float a = e < fadeIn ? Mathf.Lerp(0f, 0.9f, e / fadeIn) : Mathf.Lerp(0.9f, 1f, (e - fadeIn) / (flightTime - fadeIn));
                SpriteAlpha.Set(cue, a);
                return true;
            });
        }

        void UpdateCorePulse(float now)
        {
            if (core == null) return;
            float d = now - corePulseStart;
            float s = 1f;
            if (d >= 0f && d < 0.04f) s = Mathf.Lerp(1f, 1.25f, d / 0.04f);
            else if (d >= 0.04f && d < 0.6f) s = Mathf.Lerp(1.25f, 1f, Ease.OutQuad((d - 0.04f) / 0.56f));
            core.localScale = new Vector3(s, s, 1f);
        }

        // ------------------------------------------------------------------ effects

        /// <summary>Hit burst: scale 0.5 -> 1.9, rotate 0 -> 45 deg clockwise, alpha 1 -> 0 over 0.64 s (OutQuad).</summary>
        void SpawnBurst(Sprite sprite, Vector3 pos, float startTime)
        {
            SpriteRenderer r = MakeSprite("HitBurst", sprite, orderHit, transform);
            r.transform.localPosition = pos;
            Track(r.gameObject, now =>
            {
                float p = (now - startTime) / 0.64f;
                if (p >= 1f) return false;
                float k = Ease.OutQuad(p);
                float s = Mathf.Lerp(0.5f, 1.9f, k);
                r.transform.localScale = new Vector3(s, s, 1f);
                r.transform.localRotation = Quaternion.Euler(0f, 0f, -45f * k);
                SpriteAlpha.Set(r, 1f - k);
                return true;
            });
        }

        // Glitch-in judgement text keys: [t, alpha, xOffsetPx]
        static readonly float[,] TextKeys =
        {
            { 0f, 0f, -10f }, { 0.04f, 1f, 3f }, { 0.12f, 0.3f, -2f }, { 0.2f, 1f, 0f },
            { 0.8f, 1f, 0f }, { 0.88f, 0f, 0f }, { 0.92f, 1f, 0f }, { 1.04f, 0f, 0f }
        };

        void SpawnText(Sprite sprite, Vector3 pos, float startTime)
        {
            SpriteRenderer r = MakeSprite("JudgeText", sprite, orderText, transform);
            r.transform.localPosition = pos;
            Track(r.gameObject, now =>
            {
                float t = now - startTime;
                int n = TextKeys.GetLength(0);
                if (t >= TextKeys[n - 1, 0]) return false;
                float a = 0f, x = 0f;
                for (int i = 0; i < n - 1; i++)
                {
                    if (t >= TextKeys[i, 0] && t <= TextKeys[i + 1, 0])
                    {
                        float u = (t - TextKeys[i, 0]) / (TextKeys[i + 1, 0] - TextKeys[i, 0]);
                        a = Mathf.Lerp(TextKeys[i, 1], TextKeys[i + 1, 1], u);
                        x = Mathf.Lerp(TextKeys[i, 2], TextKeys[i + 1, 2], u);
                        break;
                    }
                }
                r.transform.localPosition = pos + new Vector3(x * DesignSpace.PxToUnit, 0f, 0f);
                SpriteAlpha.Set(r, a);
                return true;
            });
        }

        // ------------------------------------------------------------------ helpers

        void Track(GameObject go, Func<float, bool> tick)
        {
            // Tick once immediately so the first rendered frame is already in place
            if (!tick(DesignSpace.Now))
            {
                Destroy(go);
                return;
            }
            Fx fx = new Fx();
            fx.go = go;
            fx.tick = tick;
            active.Add(fx);
        }

        static SpriteRenderer MakeSprite(string name, Sprite sprite, int order, Transform parent)
        {
            GameObject g = new GameObject(name);
            g.transform.SetParent(parent, false);
            SpriteRenderer r = g.AddComponent<SpriteRenderer>();
            r.sprite = sprite;
            r.sortingOrder = order;
            return r;
        }
    }
}
