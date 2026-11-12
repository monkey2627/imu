using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

public sealed class RhythmModeController : MonoBehaviour
{
    [Serializable]
    sealed class RhythmDiagnosticRecord
    {
        public string timestampUtc,scene;
        public double songTime,noteHitTime;
        public RhythmJudgement judgement;
        public RhythmPoseDiagnostic pose;
    }
    [Serializable]
    public sealed class RecordedSegmentTarget
    {
        public string stateName;
        public RhythmBodyTarget first;
        public RhythmBodyTarget second;
    }

    [Tooltip("Optional limb mapping for recorded animations; empty preserves the existing mapping.")]
    [SerializeField] private RecordedSegmentTarget[] recordedSegmentTargets = new RecordedSegmentTarget[0];

    [Header("Session")]
    [SerializeField] private CoachActionController coachController;
    [SerializeField] private RhythmRingLayout layout;
    [SerializeField] private RhythmSongClock songClock;
    [SerializeField] private RhythmPoseJudge poseJudge;
    [SerializeField] private RhythmHitFeedback feedback;
    [SerializeField] private Transform noteContainer;
    [SerializeField] private GameObject visualEffectRoot;
    [Tooltip("Optional Ring Skin C presentation. Does not change note timing, paths or judgement.")]
    [SerializeField] private RhythmSkinCVisuals visualSkin;
    [SerializeField] private bool followCoachScoringSession = true;
    [SerializeField] private bool startAutomaticallyWithoutCoach;

    [Header("View Placement")]
    [Range(0.1f, 2f)] [SerializeField] private float worldVisualScale = 0.7f;

    [Header("Animation-driven Spawning")]
    [Min(0.2f)] [SerializeField] private float approachDuration = 2.2f;
    [Range(1, 16)] [SerializeField] private int maximumActiveNotes = 8;
    [Min(0.05f)] [SerializeField] private float sameLaneMinimumHitGap = 0.55f;
    [Tooltip("Lift clips that raise the left foot and right hand.")]
    [SerializeField] private string[] leftFootRightHandLiftStates =
    {
        "deadbug_1_up",
        "deadbug_3_up"
    };
    [Tooltip("Lift clips that raise the right foot and left hand.")]
    [SerializeField] private string[] rightFootLeftHandLiftStates =
    {
        "deadbug_2_up",
        "deadbug_4_up"
    };

    [Header("Target Appearance")]
    [SerializeField] private bool useRingArcTargets;
    [Tooltip("Explicit target visual. When assigned, the legacy musical-note pools are ignored.")]
    [SerializeField] private GameObject targetVisualPrefab;
    [SerializeField] private Vector3 targetVisualEulerRotation = Vector3.zero;
    [Tooltip("Sprite material used for the solid arc fragments emitted on HIT.")]
    [SerializeField] private Material targetFragmentMaterial;
    [Tooltip("Cloud material used only for the short HIT puff.")]
    [SerializeField] private Material targetSmokeMaterial;
    [ColorUsage(true, true)]
    [SerializeField] private Color targetArcColor = new Color(0f, 0.23921928f, 1.6457564f, 1f);
    [ColorUsage(true, true)]
    [SerializeField] private Color targetArcEdgeColor = new Color(0.30101353f, 1.1941785f, 5.038032f, 1f);
    [ColorUsage(true, true)]
    [SerializeField] private Color targetSmokeColor = new Color(0.10709113f, 0.42374614f, 3.4090674f, 1f);
    [SerializeField] private Vector3 noteBaseScale = new Vector3(0.24f, 0.24f, 0.12f);
    [SerializeField] private GameObject[] fallbackPrefabs = new GameObject[0];
    [SerializeField] private Material[] fallbackMaterials = new Material[0];
    [SerializeField] private bool normalizePrefabNotes = true;
    [SerializeField] private Vector3 prefabEulerRotation = new Vector3(0f, 90f, 0f);

    [Header("Arrival Indicator")]
    [SerializeField] private bool showArrivalIndicator = true;
    [SerializeField] private GameObject targetIndicatorPrefab;
    [Min(0.02f)] [SerializeField] private float targetIndicatorRadius = 0.13f;
    [Min(0.002f)] [SerializeField] private float targetIndicatorWidth = 0.018f;
    [Range(6, 32)] [SerializeField] private int targetIndicatorDashCount = 16;
    [Range(0.2f, 0.85f)] [SerializeField] private float targetIndicatorDashFill = 0.55f;
    [SerializeField] private bool targetIndicatorParticles = true;
    [Min(1f)] [SerializeField] private float targetIndicatorParticleRate = 18f;
    [Min(0.002f)] [SerializeField] private float targetIndicatorParticleSize = 0.014f;
    [Header("Judgement Timing")]
    [Tooltip("Time spent moving from the edge of the target ring to its center.")]
    [Min(0f)] [SerializeField] private float noteEntryDurationSeconds = 0.4f;
    [Tooltip("Time spent holding at the center after entering the target ring.")]
    [Min(0f)] [SerializeField] private float noteHoldDurationSeconds = 0.8f;
    [SerializeField] private RhythmNoteVariantPool[] targetVariantPools =
    {
        new RhythmNoteVariantPool { target = RhythmBodyTarget.LeftHand },
        new RhythmNoteVariantPool { target = RhythmBodyTarget.RightHand },
        new RhythmNoteVariantPool { target = RhythmBodyTarget.LeftFoot },
        new RhythmNoteVariantPool { target = RhythmBodyTarget.RightFoot },
        new RhythmNoteVariantPool { target = RhythmBodyTarget.Waist }
    };

    private readonly List<RhythmNoteView> activeNotes = new List<RhythmNoteView>();
    private readonly double[] lastLaneHitTimes =
    {
        double.NegativeInfinity,
        double.NegativeInfinity,
        double.NegativeInfinity,
        double.NegativeInfinity,
        double.NegativeInfinity
    };
    private bool running;
    private Material fallbackRuntimeMaterial;
    private RhythmJudgement? tutorialJudgementOverride;
    private RhythmMovementRange spawningReference;
    private string diagnosticFilePath;

    public event Action<RhythmNoteResult> NoteResolved;

    public bool IsRunning => running && songClock != null && songClock.IsRunning;
    public double SongTime => songClock != null ? songClock.SongTime : 0d;
    public Vector3 NoteBaseScale => noteBaseScale;
    public bool UsesRingArcTarget => useRingArcTargets && targetVisualPrefab != null;

    private void Awake()
    {
        EnsureHitHaptics();
        PrepareFixedVisualRoot();
        SkinnedMeshColorAnchorInstaller.EnsureOn(gameObject);
        ResolveReferences();
        if (layout != null)
        {
            layout.SetVisible(!followCoachScoringSession && startAutomaticallyWithoutCoach);
        }
        SetVisualEffectVisible(!followCoachScoringSession && startAutomaticallyWithoutCoach);
    }

    private void EnsureHitHaptics()
    {
        if (FindObjectOfType<RhythmHitHapticAdapter>(true) == null)
        {
            gameObject.AddComponent<RhythmHitHapticAdapter>();
        }
    }

    private void OnEnable()
    {
        PrepareFixedVisualRoot();
        ResolveReferences();
        Subscribe();
    }

    private void PrepareFixedVisualRoot()
    {
        if (GetComponentInParent<Canvas>() != null)
        {
            transform.SetParent(null, true);
        }
        transform.localScale = Vector3.one * worldVisualScale;

    }

    private void Start()
    {
        if ((!followCoachScoringSession || coachController == null) && startAutomaticallyWithoutCoach)
        {
            StartRhythmMode();
        }
    }

    private void OnDisable()
    {
        Unsubscribe();
        StopRhythmMode();
    }

    private void OnDestroy()
    {
        if (fallbackRuntimeMaterial != null)
        {
            Destroy(fallbackRuntimeMaterial);
        }
    }

    private void Update()
    {
        if (!IsRunning)
        {
            return;
        }

        RemoveDestroyedNotes();
    }

    public void StartRhythmMode()
    {
        if (layout == null || songClock == null || poseJudge == null || feedback == null)
        {
            Debug.LogError("RhythmModeController is missing an explicitly mounted component.", this);
            return;
        }

        ClearNotes();
        diagnosticFilePath=Path.Combine(Application.persistentDataPath,"rhythm-diagnostics",
            "rhythm-"+DateTime.Now.ToString("yyyyMMdd-HHmmss-fff",CultureInfo.InvariantCulture)+".jsonl");
        Debug.Log("Rhythm leg diagnostics: "+diagnosticFilePath,this);
        songClock.StartClock();
        running = true;
        layout.SetVisible(true);
        SetVisualEffectVisible(true);
    }

    public void ShowTutorialVisuals()
    {
        ResolveReferences();
        if (layout != null)
        {
            layout.SetVisible(true);
        }
        SetVisualEffectVisible(true);
    }

    public void StopRhythmMode()
    {
        running = false;
        tutorialJudgementOverride = null;
        if (songClock != null)
        {
            songClock.StopClock();
        }
        if (layout != null)
        {
            layout.SetVisible(false);
        }
        SetVisualEffectVisible(false);
        ClearNotes();
    }

    public RhythmNoteResult JudgeAtLine(RhythmNoteView note)
    {
        if (note == null)
        {
            return default(RhythmNoteResult);
        }

        if (tutorialJudgementOverride.HasValue)
        {
            RhythmJudgement judgement = tutorialJudgementOverride.Value;
            return new RhythmNoteResult(
                note.Target,
                judgement,
                judgement == RhythmJudgement.Miss ? 0f : 100f,
                0f,
                note.HitLocalPosition);
        }

        bool success = poseJudge.TryHit(note.Target,note.JudgementReference,out float poseScore);
        return new RhythmNoteResult(
            note.Target,
            success ? RhythmJudgement.Perfect : RhythmJudgement.Miss,
            poseScore,
            0f,
            note.HitLocalPosition);
    }

    public void SetTutorialJudgementOverride(RhythmJudgement? judgement)
    {
        tutorialJudgementOverride = judgement;
    }

    public float GetCurrentPoseScore(RhythmBodyTarget target)
    {
        return poseJudge != null ? poseJudge.GetScore(target) : 0f;
    }

    public void ResolveNote(RhythmNoteView note, RhythmNoteResult result)
    {
        activeNotes.Remove(note);
        if(result.Target==RhythmBodyTarget.LeftFoot||result.Target==RhythmBodyTarget.RightFoot)
            RecordLegDiagnostic(note,result);
        if (result.Judgement != RhythmJudgement.Miss)
        {
            lastLaneHitTimes[(int)result.Target] = SongTime;
            RhythmBreakableTarget breakableTarget = note != null
                ? note.GetComponent<RhythmBreakableTarget>()
                : null;
            if (breakableTarget != null)
            {
                breakableTarget.PlayBreakAndDissolve();
            }
        }
        feedback.Play(
            result,
            layout.GetTargetColor(result.Target),
            layout.GetRandomAccentColor());
        NoteResolved?.Invoke(result);

        if (note != null)
        {
            Destroy(note.gameObject);
        }
    }

    void RecordLegDiagnostic(RhythmNoteView note,RhythmNoteResult result)
    {
        if(tutorialJudgementOverride.HasValue||poseJudge==null||poseJudge.LastDiagnostic==null)return;
        try
        {
            string directory=Path.GetDirectoryName(diagnosticFilePath);
            Directory.CreateDirectory(directory);
            var record=new RhythmDiagnosticRecord{timestampUtc=DateTime.UtcNow.ToString("o",CultureInfo.InvariantCulture),
                scene=gameObject.scene.name,
                songTime=SongTime,noteHitTime=note!=null?note.HitTime:SongTime,
                judgement=result.Judgement,pose=poseJudge.LastDiagnostic};
            File.AppendAllText(diagnosticFilePath,JsonUtility.ToJson(record)+"\n");
        }
        catch(Exception exception)
        {
            Debug.LogWarning("Could not write rhythm diagnostic: "+exception.Message,this);
        }
    }

    private void HandleCoachSegmentStarted(CoachActionController.SegmentPlaybackInfo info)
    {
        if (!IsRunning || !TryGetSegmentTargets(
                info.stateName,
                out RhythmBodyTarget firstTarget,
                out RhythmBodyTarget secondTarget))
        {
            return;
        }

        float segmentDuration = GetSegmentDuration(info.stateName, info.playbackSpeed);
        spawningReference = poseJudge.CaptureTarget(info.stateName);
        double hitTime = SongTime + segmentDuration;
        if (activeNotes.Count < maximumActiveNotes && CanSpawnLane(firstTarget, hitTime))
        {
            SpawnNote(firstTarget, hitTime, segmentDuration);
        }
        if (activeNotes.Count < maximumActiveNotes && CanSpawnLane(secondTarget, hitTime))
        {
            SpawnNote(secondTarget, hitTime, segmentDuration);
        }
    }

    private void SpawnNote(
        RhythmBodyTarget target,
        double hitTime,
        float noteApproachDuration)
    {
        layout.GetRandomPath(target, out Vector3 spawn, out Vector3 hit);
        Vector3 resultPosition = hit;
        Vector3 targetDirection = hit.sqrMagnitude > 0.000001f
            ? hit.normalized
            : Vector3.up;
        if (UsesRingArcTarget)
        {
            float radialTravel = Mathf.Max(0f, spawn.magnitude - hit.magnitude);
            spawn = targetDirection * radialTravel;
            hit = Vector3.zero;
        }

        GameObject noteObject = CreateNoteObject(target);
        noteObject.name = $"Target_{target}_{hitTime:0.00}";
        noteObject.transform.SetParent(noteContainer != null ? noteContainer : transform, false);
        noteObject.transform.localScale = noteBaseScale;
        if (visualSkin == null)
        {
            if (UsesRingArcTarget) ApplyTargetMaterialColors(noteObject);
            else ApplyRandomMaterial(noteObject, target);
        }

        float noteRadius = Mathf.Max(noteBaseScale.x, noteBaseScale.y) * 0.5f;
        RhythmNoteTargetIndicator indicator = showArrivalIndicator
            ? RhythmNoteTargetIndicator.Create(
                noteContainer != null ? noteContainer : transform,
                hit,
                targetIndicatorRadius,
                targetIndicatorWidth,
                targetIndicatorDashCount,
                targetIndicatorDashFill,
                layout.GetTargetColor(target),
                targetIndicatorPrefab,
                targetIndicatorParticles,
                targetIndicatorParticleRate,
                targetIndicatorParticleSize)
            : null;

        RhythmNoteView view = noteObject.GetComponent<RhythmNoteView>();
        if (view == null)
        {
            view = noteObject.AddComponent<RhythmNoteView>();
        }
        view.Initialize(
            this,
            target,
            spawn,
            hit,
            hitTime,
            noteApproachDuration,
            noteEntryDurationSeconds,
            noteHoldDurationSeconds,
            targetIndicatorRadius,
            indicator,
            noteRadius,
            resultPosition);
        view.SetJudgementReference(spawningReference);
        if (UsesRingArcTarget)
        {
            float radialAngle = Mathf.Atan2(targetDirection.y, targetDirection.x) * Mathf.Rad2Deg;
            noteObject.transform.localRotation = Quaternion.Euler(0f, 0f, radialAngle + 180f);
        }
        activeNotes.Add(view);
    }

    private GameObject CreateNoteObject(RhythmBodyTarget target)
    {
        if (visualSkin != null)
        {
            return visualSkin.CreateNote(target, noteBaseScale);
        }
        if (UsesRingArcTarget)
        {
            GameObject targetObject = CreatePrefabNote(
                targetVisualPrefab,
                targetVisualEulerRotation,
                false);
            RhythmBreakableTarget breakableTarget =
                targetObject.AddComponent<RhythmBreakableTarget>();
            breakableTarget.Initialize(
                targetFragmentMaterial,
                targetSmokeMaterial,
                targetArcColor,
                targetSmokeColor);
            return targetObject;
        }

        RhythmNoteVariantPool pool = FindPool(target);
        GameObject[] candidates = pool != null && HasValues(pool.prefabs)
            ? pool.prefabs
            : fallbackPrefabs;
        GameObject prefab = ChooseRandomNonNull(candidates);
        if (prefab != null)
        {
            return CreatePrefabNote(prefab, prefabEulerRotation, normalizePrefabNotes);
        }

        GameObject generated = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        Collider collider = generated.GetComponent<Collider>();
        if (collider != null)
        {
            Destroy(collider);
        }
        return generated;
    }

    private GameObject CreatePrefabNote(
        GameObject prefab,
        Vector3 eulerRotation,
        bool normalizeVisual)
    {
        GameObject noteRoot = new GameObject(prefab.name + " Note");
        GameObject visual = Instantiate(prefab, noteRoot.transform, false);
        visual.name = "Visual";
        visual.transform.localPosition = Vector3.zero;
        visual.transform.localRotation = Quaternion.Euler(eulerRotation);
        visual.transform.localScale = Vector3.one;

        if (normalizeVisual)
        {
            NormalizePrefabVisual(noteRoot.transform, visual.transform);
        }

        return noteRoot;
    }

    private void ApplyTargetMaterialColors(GameObject targetObject)
    {
        foreach (MeshRenderer meshRenderer in targetObject.GetComponentsInChildren<MeshRenderer>(true))
        {
            MaterialPropertyBlock block = new MaterialPropertyBlock();
            meshRenderer.GetPropertyBlock(block);
            block.SetColor("_Color", targetArcColor);
            block.SetColor("_BaseColor", targetArcColor);
            block.SetColor("_EdgeColor", targetArcEdgeColor);
            block.SetFloat("_Dissolve", 0f);
            meshRenderer.SetPropertyBlock(block);
        }
    }

    private static void NormalizePrefabVisual(Transform noteRoot, Transform visual)
    {
        MeshRenderer[] renderers = visual.GetComponentsInChildren<MeshRenderer>(true);
        if (renderers.Length == 0)
        {
            return;
        }

        Bounds bounds = renderers[0].bounds;
        for (int index = 1; index < renderers.Length; index++)
        {
            bounds.Encapsulate(renderers[index].bounds);
        }

        Vector3 localCenter = noteRoot.InverseTransformPoint(bounds.center);
        float visibleSize = Mathf.Max(bounds.size.x, bounds.size.y);
        if (visibleSize <= 0.000001f)
        {
            return;
        }

        float normalizationScale = 1f / visibleSize;
        visual.localScale = Vector3.one * normalizationScale;
        visual.localPosition = -localCenter * normalizationScale;
    }

    private void ApplyRandomMaterial(GameObject noteObject, RhythmBodyTarget target)
    {
        RhythmNoteVariantPool pool = FindPool(target);
        Material[] candidates = pool != null && HasValues(pool.materials)
            ? pool.materials
            : fallbackMaterials;
        Material material = ChooseRandomNonNull(candidates);
        Color color = layout.GetTargetColor(target);

        bool hasAuthoredEffects = noteObject.GetComponentInChildren<ParticleSystem>(true) != null ||
            noteObject.GetComponentInChildren<TrailRenderer>(true) != null;
        if (hasAuthoredEffects && material == null)
        {
            return;
        }

        Renderer[] renderers = noteObject.GetComponentsInChildren<Renderer>(true);
        foreach (Renderer targetRenderer in renderers)
        {
            if (material != null)
            {
                targetRenderer.sharedMaterial = material;
            }
            else if (targetRenderer.sharedMaterial == null)
            {
                targetRenderer.sharedMaterial = GetFallbackMaterial();
            }

            Material effectiveMaterial = targetRenderer.sharedMaterial;
            MaterialPropertyBlock block = new MaterialPropertyBlock();
            targetRenderer.GetPropertyBlock(block);
            if (effectiveMaterial != null && effectiveMaterial.HasProperty("_BaseColor"))
                block.SetColor("_BaseColor", color);
            if (effectiveMaterial != null && effectiveMaterial.HasProperty("_Color"))
                block.SetColor("_Color", color);
            targetRenderer.SetPropertyBlock(block);
        }
    }

    private Material GetFallbackMaterial()
    {
        if (fallbackRuntimeMaterial != null)
        {
            return fallbackRuntimeMaterial;
        }

        Shader shader = Shader.Find("Universal Render Pipeline/Unlit") ??
            Shader.Find("Unlit/Color") ??
            Shader.Find("Standard");
        fallbackRuntimeMaterial = new Material(shader)
        {
            name = "Runtime Rhythm Note Material"
        };
        return fallbackRuntimeMaterial;
    }

    public bool TryGetSegmentTargets(
        string stateName,
        out RhythmBodyTarget first,
        out RhythmBodyTarget second)
    {
        // Recorded clips may label a different limb pair. Empty in the original scenes.
        if (recordedSegmentTargets != null)
        foreach (var target in recordedSegmentTargets)
        {
            if (target != null && string.Equals(target.stateName, stateName, StringComparison.OrdinalIgnoreCase))
            {
                first = target.first;
                second = target.second;
                return true;
            }
        }
        if (ContainsState(leftFootRightHandLiftStates, stateName))
        {
            first = RhythmBodyTarget.LeftFoot;
            second = RhythmBodyTarget.RightHand;
            return true;
        }
        if (ContainsState(rightFootLeftHandLiftStates, stateName))
        {
            first = RhythmBodyTarget.RightFoot;
            second = RhythmBodyTarget.LeftHand;
            return true;
        }

        first = default(RhythmBodyTarget);
        second = default(RhythmBodyTarget);
        return false;
    }

    private static bool ContainsState(string[] states, string stateName)
    {
        if (states == null || string.IsNullOrEmpty(stateName))
        {
            return false;
        }

        foreach (string candidate in states)
        {
            if (string.Equals(candidate, stateName, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return false;
    }

    private float GetSegmentDuration(string stateName, float playbackSpeed)
    {
        Animator animator = poseJudge != null ? poseJudge.CoachAnimator : null;
        RuntimeAnimatorController runtimeController = animator != null
            ? animator.runtimeAnimatorController
            : null;
        if (runtimeController != null)
        {
            foreach (AnimationClip clip in runtimeController.animationClips)
            {
                if (clip != null && string.Equals(clip.name, stateName, StringComparison.Ordinal))
                {
                    return Mathf.Max(0.1f, clip.length / Mathf.Max(0.05f, playbackSpeed));
                }
            }
        }

        return Mathf.Max(0.1f, approachDuration);
    }

    private bool CanSpawnLane(RhythmBodyTarget target, double hitTime)
    {
        if (hitTime - lastLaneHitTimes[(int)target] < sameLaneMinimumHitGap)
        {
            return false;
        }

        foreach (RhythmNoteView note in activeNotes)
        {
            if (note != null && note.Target == target &&
                Math.Abs(note.HitTime - hitTime) < sameLaneMinimumHitGap)
            {
                return false;
            }
        }
        return true;
    }

    private RhythmNoteVariantPool FindPool(RhythmBodyTarget target)
    {
        if (targetVariantPools == null)
        {
            return null;
        }

        foreach (RhythmNoteVariantPool pool in targetVariantPools)
        {
            if (pool != null && pool.target == target)
            {
                return pool;
            }
        }
        return null;
    }

    private static T ChooseRandomNonNull<T>(T[] values) where T : UnityEngine.Object
    {
        if (!HasValues(values))
        {
            return null;
        }

        int start = UnityEngine.Random.Range(0, values.Length);
        for (int offset = 0; offset < values.Length; offset++)
        {
            T value = values[(start + offset) % values.Length];
            if (value != null)
            {
                return value;
            }
        }
        return null;
    }

    private static bool HasValues<T>(T[] values)
    {
        return values != null && values.Length > 0;
    }

    private void HandleCoachSessionStarted()
    {
        StartRhythmMode();
    }

    private void HandleCoachSessionCompleted()
    {
        StopRhythmMode();
    }

    private void ResolveReferences()
    {
        if (coachController == null)
        {
            coachController = FindObjectOfType<CoachActionController>(true);
        }
        if (layout == null) layout = GetComponentInChildren<RhythmRingLayout>(true);
        if (songClock == null) songClock = GetComponentInChildren<RhythmSongClock>(true);
        if (poseJudge == null) poseJudge = GetComponentInChildren<RhythmPoseJudge>(true);
        if (feedback == null) feedback = GetComponentInChildren<RhythmHitFeedback>(true);
        if (noteContainer == null)
        {
            Transform found = transform.Find("ActiveNotes");
            noteContainer = found != null ? found : transform;
        }
        if (visualEffectRoot == null)
        {
            Transform found = transform.Find("RhythmVisualEffects");
            visualEffectRoot = found != null ? found.gameObject : null;
        }
    }

    private void SetVisualEffectVisible(bool visible)
    {
        if (visualEffectRoot != null && visualEffectRoot.activeSelf != visible)
        {
            visualEffectRoot.SetActive(visible);
        }
    }

    private void Subscribe()
    {
        if (coachController == null || !followCoachScoringSession)
        {
            return;
        }

        coachController.ScoringSessionStarted -= HandleCoachSessionStarted;
        coachController.ScoringSessionCompleted -= HandleCoachSessionCompleted;
        coachController.SegmentPlaybackStarted -= HandleCoachSegmentStarted;
        coachController.ScoringSessionStarted += HandleCoachSessionStarted;
        coachController.ScoringSessionCompleted += HandleCoachSessionCompleted;
        coachController.SegmentPlaybackStarted += HandleCoachSegmentStarted;
    }

    private void Unsubscribe()
    {
        if (coachController == null)
        {
            return;
        }

        coachController.ScoringSessionStarted -= HandleCoachSessionStarted;
        coachController.ScoringSessionCompleted -= HandleCoachSessionCompleted;
        coachController.SegmentPlaybackStarted -= HandleCoachSegmentStarted;
    }

    private void RemoveDestroyedNotes()
    {
        for (int index = activeNotes.Count - 1; index >= 0; index--)
        {
            if (activeNotes[index] == null)
            {
                activeNotes.RemoveAt(index);
            }
        }
    }

    private void ClearNotes()
    {
        foreach (RhythmNoteView note in activeNotes)
        {
            if (note != null)
            {
                Destroy(note.gameObject);
            }
        }
        activeNotes.Clear();
    }
}

internal sealed class RhythmBreakableTarget : MonoBehaviour
{
    private const float EffectLifetime = 0.65f;
    private ParticleSystem fragmentParticles;
    private ParticleSystem smokeParticles;
    private bool played;

    public void Initialize(
        Material fragmentMaterial,
        Material smokeMaterial,
        Color fragmentColor,
        Color smokeColor)
    {
        ParticleSystem[] particleSystems = GetComponentsInChildren<ParticleSystem>(true);
        foreach (ParticleSystem particles in particleSystems)
        {
            bool wasLooping = particles.main.loop;
            particles.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);

            ParticleSystem.MainModule main = particles.main;
            main.loop = false;
            main.playOnAwake = false;
            main.duration = 0.05f;
            main.simulationSpace = ParticleSystemSimulationSpace.Local;

            ParticleSystem.EmissionModule emission = particles.emission;
            emission.enabled = false;

            if (wasLooping && smokeParticles == null)
            {
                smokeParticles = particles;
                ConfigureSmoke(particles, smokeMaterial, smokeColor);
            }
            else if (fragmentParticles == null)
            {
                fragmentParticles = particles;
                ConfigureFragments(particles, fragmentMaterial, fragmentColor);
            }
        }
    }

    public void PlayBreakAndDissolve()
    {
        if (played)
        {
            return;
        }

        played = true;
        foreach (MeshRenderer meshRenderer in GetComponentsInChildren<MeshRenderer>(true))
        {
            meshRenderer.enabled = false;
        }

        EmitAndDetach(fragmentParticles, 12);
        EmitAndDetach(smokeParticles, 4);
    }

    private void EmitAndDetach(ParticleSystem particles, int count)
    {
        if (particles == null)
        {
            return;
        }

        Transform effectTransform = particles.transform;
        effectTransform.SetParent(transform.parent, true);
        particles.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
        particles.Emit(count);
        Destroy(effectTransform.gameObject, EffectLifetime);
    }

    private static void ConfigureFragments(
        ParticleSystem particles,
        Material material,
        Color color)
    {
        ParticleSystem.MainModule main = particles.main;
        main.startLifetime = new ParticleSystem.MinMaxCurve(0.28f, 0.48f);
        main.startSpeed = new ParticleSystem.MinMaxCurve(0.28f, 0.52f);
        main.startSize = new ParticleSystem.MinMaxCurve(0.09f, 0.16f);
        main.startRotation = new ParticleSystem.MinMaxCurve(-Mathf.PI, Mathf.PI);
        main.gravityModifier = 0.05f;
        main.maxParticles = 24;

        ParticleSystemRenderer renderer = particles.GetComponent<ParticleSystemRenderer>();
        if (renderer != null && material != null)
        {
            renderer.sharedMaterial = material;
            renderer.sortingOrder = 36;
            ApplyParticleColor(renderer, color);
        }
    }

    private static void ConfigureSmoke(
        ParticleSystem particles,
        Material material,
        Color color)
    {
        ParticleSystem.MainModule main = particles.main;
        main.startLifetime = new ParticleSystem.MinMaxCurve(0.18f, 0.34f);
        main.startSpeed = new ParticleSystem.MinMaxCurve(0.04f, 0.12f);
        main.startSize = new ParticleSystem.MinMaxCurve(0.07f, 0.14f);
        main.maxParticles = 8;

        ParticleSystemRenderer renderer = particles.GetComponent<ParticleSystemRenderer>();
        if (renderer != null && material != null)
        {
            renderer.sharedMaterial = material;
            renderer.sortingOrder = 35;
            ApplyParticleColor(renderer, color);
        }
    }

    private static void ApplyParticleColor(Renderer renderer, Color color)
    {
        MaterialPropertyBlock block = new MaterialPropertyBlock();
        renderer.GetPropertyBlock(block);
        block.SetColor("_Color", color);
        block.SetColor("_BaseColor", color);
        renderer.SetPropertyBlock(block);
    }
}
