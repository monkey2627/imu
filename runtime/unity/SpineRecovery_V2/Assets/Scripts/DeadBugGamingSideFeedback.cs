using UnityEngine;
using System.Collections.Generic;

[DefaultExecutionOrder(200)]
public class DeadBugGamingSideFeedback : MonoBehaviour
{
    private enum DeadBugTarget
    {
        LeftKnee,
        LeftHand,
        RightKnee,
        RightHand
    }

    private class StarTargetState
    {
        public bool hasTarget;
        public DeadBugTarget target;
        public bool arrivedAtTarget;
        public bool matched;
        public bool hasPredictedTarget;
        public Vector3 startPosition;
        public Vector3 targetPosition;
    }

    [Header("Avatars")]
    [SerializeField] private Animator coachAnimator;
    [SerializeField] private Animator userAnimator;
    [SerializeField] private bool hideUserRenderersDuringTraining = true;
    [SerializeField] private bool keepRootsOverlapped = true;
    [SerializeField] private bool moveCoachToUser = true;
    [SerializeField] private bool matchRootRotation = true;
    [SerializeField] private bool horizontalPositionOnly = true;
    [SerializeField] private bool yawRotationOnly = true;

    [Header("Grounding")]
    [SerializeField] private bool groundCoachToFloor = true;
    [SerializeField] private bool useUserFeetAsFloor = true;
    [SerializeField] private float floorY = 0f;
    [SerializeField] private float floorClearance = 0.02f;
    [SerializeField] private float groundSmoothSpeed = 18f;
    [SerializeField] private bool groundByTorsoOnly = true;
    [SerializeField] private bool groundOnlyWhenCoachLying = true;
    [SerializeField] private float lyingUpDotThreshold = 0.65f;

    [Header("Dead Bug Targets")]
    [SerializeField] private bool usePhaseTargets = true;
    [SerializeField] private float targetCyclesPerAnimationLoop = 2f;
    [SerializeField] private float phaseSplit = 0.5f;
    [SerializeField] private float arriveAtPhasePercent = 0.3f;
    [SerializeField] private float targetRotateSpeed = 180f;
    [SerializeField] private float kneeMatchDistance = 0.22f;
    [SerializeField] private float handMatchDistance = 0.18f;
    [SerializeField] private bool requireCoachOverlapForFlash = true;
    [SerializeField] private float coachOverlapDistance = 0.22f;
    [SerializeField] private bool requireCoachNearTargetForFlash = true;
    [SerializeField] private float coachTargetDistance = 0.28f;
    [SerializeField] private Transform leftKneePeakTarget;
    [SerializeField] private Transform leftHandPeakTarget;
    [SerializeField] private Transform rightKneePeakTarget;
    [SerializeField] private Transform rightHandPeakTarget;

    [Header("Stars")]
    [SerializeField] private Transform blueStar;
    [SerializeField] private Transform yellowStar;
    [SerializeField] private bool autoPlaceStars;
    [SerializeField] private float starSideOffset = 0.75f;
    [SerializeField] private float starVerticalOffset = 0.75f;
    [SerializeField] private float starForwardOffset = 0.05f;

    [Header("Match")]
    [SerializeField] private float matchDistance = 0.28f;
    [SerializeField] private float flashScore = 0.75f;
    [SerializeField] private float releaseScore = 0.55f;
    [SerializeField] private float smoothSpeed = 12f;

    [Header("Flash")]
    [SerializeField] private float idleScale = 1f;
    [SerializeField] private float flashScale = 1.35f;
    [SerializeField] private float flashPulseSpeed = 9f;

    private readonly HumanBodyBones[] leftBones =
    {
        HumanBodyBones.LeftShoulder,
        HumanBodyBones.LeftUpperArm,
        HumanBodyBones.LeftLowerArm,
        HumanBodyBones.LeftHand,
        HumanBodyBones.LeftUpperLeg,
        HumanBodyBones.LeftLowerLeg,
        HumanBodyBones.LeftFoot
    };

    private readonly HumanBodyBones[] rightBones =
    {
        HumanBodyBones.RightShoulder,
        HumanBodyBones.RightUpperArm,
        HumanBodyBones.RightLowerArm,
        HumanBodyBones.RightHand,
        HumanBodyBones.RightUpperLeg,
        HumanBodyBones.RightLowerLeg,
        HumanBodyBones.RightFoot
    };

    private ParticleSystem[] blueParticles;
    private ParticleSystem[] yellowParticles;
    private Light[] blueLights;
    private Light[] yellowLights;
    private Vector3 blueBaseScale = Vector3.one;
    private Vector3 yellowBaseScale = Vector3.one;
    private float blueScore;
    private float yellowScore;
    private bool blueMatched;
    private bool yellowMatched;
    private bool feedbackActive;
    private bool hasSavedCoachRoot;
    private Vector3 savedCoachRootPosition;
    private Quaternion savedCoachRootRotation;
    private readonly StarTargetState blueTargetState = new StarTargetState();
    private readonly StarTargetState yellowTargetState = new StarTargetState();
    private readonly List<RendererState> hiddenUserRendererStates = new List<RendererState>();

    private void Awake()
    {
        ResolveAvatarReferences();
        AutoFindPeakTargets();

        if (blueStar == null)
        {
            GameObject foundBlue = GameObject.Find("EnergyNovaBlue");
            if (foundBlue != null) blueStar = foundBlue.transform;
        }

        if (yellowStar == null)
        {
            GameObject foundYellow = GameObject.Find("EnergyNovaYellow");
            if (foundYellow != null) yellowStar = foundYellow.transform;
        }

        CacheStar(blueStar, ref blueBaseScale, ref blueParticles, ref blueLights);
        CacheStar(yellowStar, ref yellowBaseScale, ref yellowParticles, ref yellowLights);
    }

    private void LateUpdate()
    {
        ResolveAvatarReferences();

        if (!feedbackActive || coachAnimator == null || userAnimator == null)
        {
            return;
        }

        if (keepRootsOverlapped)
        {
            OverlapAvatarRoots();
        }

        if (ShouldGroundCoach())
        {
            GroundAvatarToFloor(coachAnimator);
        }

        if (usePhaseTargets)
        {
            UpdatePhaseTargets();
        }
        else
        {
            if (autoPlaceStars)
            {
                PlaceStars();
            }

            blueScore = Mathf.Lerp(blueScore, CalculateSideScore(leftBones), GetFrameBlend());
            yellowScore = Mathf.Lerp(yellowScore, CalculateSideScore(rightBones), GetFrameBlend());

            blueMatched = UpdateMatchedState(blueMatched, blueScore);
            yellowMatched = UpdateMatchedState(yellowMatched, yellowScore);

            ApplyStarFeedback(blueStar, blueBaseScale, blueParticles, blueLights, blueMatched);
            ApplyStarFeedback(yellowStar, yellowBaseScale, yellowParticles, yellowLights, yellowMatched);
        }
    }

    public void BeginFeedback()
    {
        ResolveAvatarReferences();
        HideUserModelForTraining();
        feedbackActive = true;
        ResetMatchState();
        SaveCoachRootPose();

        if (coachAnimator == null || userAnimator == null)
        {
            return;
        }

        if (keepRootsOverlapped)
        {
            OverlapAvatarRoots();
        }

        if (ShouldGroundCoach())
        {
            GroundAvatarToFloor(coachAnimator);
        }
    }

    public void EndFeedback()
    {
        feedbackActive = false;
        ResetMatchState();
        ApplyStarFeedback(blueStar, blueBaseScale, blueParticles, blueLights, false);
        ApplyStarFeedback(yellowStar, yellowBaseScale, yellowParticles, yellowLights, false);
        RestoreCoachRootPose();
    }

    private void OnDisable()
    {
        RestoreUserModelVisibility();
    }

    public void HideUserModelForTraining()
    {
        ResolveAvatarReferences();

        if (!hideUserRenderersDuringTraining || userAnimator == null)
        {
            return;
        }

        Transform userRoot = userAnimator.transform;

        if (userRoot == null)
        {
            return;
        }

        Renderer[] renderers = userRoot.GetComponentsInChildren<Renderer>(true);
        foreach (Renderer renderer in renderers)
        {
            if (!ShouldHideUserRenderer(renderer) || ContainsHiddenUserRenderer(renderer))
            {
                continue;
            }

            hiddenUserRendererStates.Add(new RendererState(renderer, renderer.enabled));
            renderer.enabled = false;
        }
    }

    public void RestoreUserModelVisibility()
    {
        foreach (RendererState state in hiddenUserRendererStates)
        {
            if (state.Renderer != null)
            {
                state.Renderer.enabled = state.WasEnabled;
            }
        }

        hiddenUserRendererStates.Clear();
    }

    private bool ContainsHiddenUserRenderer(Renderer renderer)
    {
        foreach (RendererState state in hiddenUserRendererStates)
        {
            if (state.Renderer == renderer)
            {
                return true;
            }
        }

        return false;
    }

    private bool ShouldHideUserRenderer(Renderer renderer)
    {
        if (renderer == null)
        {
            return false;
        }

        if (coachAnimator != null &&
            coachAnimator.transform != null &&
            renderer.transform.IsChildOf(coachAnimator.transform))
        {
            return false;
        }

        Animator ownerAnimator = renderer.GetComponentInParent<Animator>();
        return ownerAnimator == null || ownerAnimator == userAnimator;
    }

    private void ResetMatchState()
    {
        blueScore = 0f;
        yellowScore = 0f;
        blueMatched = false;
        yellowMatched = false;
        ResetTargetState(blueTargetState);
        ResetTargetState(yellowTargetState);
    }

    private void UpdatePhaseTargets()
    {
        float phase = GetCoachPhase();
        float safePhaseSplit = Mathf.Clamp(phaseSplit, 0.05f, 0.95f);
        bool firstHalf = phase < safePhaseSplit;
        float phaseInSegment = firstHalf
            ? phase / safePhaseSplit
            : (phase - safePhaseSplit) / (1f - safePhaseSplit);

        phaseInSegment = Mathf.Clamp01(phaseInSegment);

        DeadBugTarget blueTarget = firstHalf ? DeadBugTarget.LeftKnee : DeadBugTarget.LeftHand;
        DeadBugTarget yellowTarget = firstHalf ? DeadBugTarget.RightHand : DeadBugTarget.RightKnee;

        UpdateStarTarget(
            blueStar,
            blueBaseScale,
            blueParticles,
            blueLights,
            blueTargetState,
            blueTarget,
            phaseInSegment,
            1f);

        UpdateStarTarget(
            yellowStar,
            yellowBaseScale,
            yellowParticles,
            yellowLights,
            yellowTargetState,
            yellowTarget,
            phaseInSegment,
            -1f);
    }

    private float GetCoachPhase()
    {
        AnimatorStateInfo stateInfo = coachAnimator.GetCurrentAnimatorStateInfo(0);
        float cycleMultiplier = Mathf.Max(0.01f, targetCyclesPerAnimationLoop);
        return Mathf.Repeat(stateInfo.normalizedTime * cycleMultiplier, 1f);
    }

    private void UpdateStarTarget(
        Transform star,
        Vector3 baseScale,
        ParticleSystem[] particles,
        Light[] lights,
        StarTargetState state,
        DeadBugTarget target,
        float phaseInSegment,
        float rotationDirection)
    {
        if (star == null)
        {
            return;
        }

        if (!state.hasTarget || state.target != target)
        {
            ResetTargetState(state);
            state.hasTarget = true;
            state.target = target;
            state.startPosition = star.position;
            state.hasPredictedTarget = TryGetPredictedTargetPosition(target, out state.targetPosition);
            if (!state.hasPredictedTarget)
            {
                state.targetPosition = state.startPosition;
            }
        }

        float arrivalPercent = Mathf.Clamp(arriveAtPhasePercent, 0.01f, 1f);
        float flyT = Mathf.Clamp01(phaseInSegment / arrivalPercent);
        if (!state.arrivedAtTarget && state.hasPredictedTarget)
        {
            star.position = Vector3.Lerp(state.startPosition, state.targetPosition, SmoothStep(flyT));
            state.arrivedAtTarget = flyT >= 1f;
        }
        else if (!state.arrivedAtTarget)
        {
            star.position = state.startPosition;
        }

        if (state.arrivedAtTarget)
        {
            star.position = state.targetPosition;
            star.Rotate(Vector3.up, targetRotateSpeed * rotationDirection * Time.deltaTime, Space.World);
            state.matched = IsUserAtTarget(target, state.targetPosition);
        }
        else
        {
            state.matched = false;
        }

        ApplyStarFeedback(star, baseScale, particles, lights, state.matched);
    }

    private void AutoFindPeakTargets()
    {
        if (leftKneePeakTarget == null) leftKneePeakTarget = FindTransformByTrimmedName("LeftKneePeakTarget");
        if (leftHandPeakTarget == null) leftHandPeakTarget = FindTransformByTrimmedName("LeftHandPeakTarget");
        if (rightKneePeakTarget == null) rightKneePeakTarget = FindTransformByTrimmedName("RightKneePeakTarget");
        if (rightHandPeakTarget == null) rightHandPeakTarget = FindTransformByTrimmedName("RightHandPeakTarget");
    }

    private Transform FindTransformByTrimmedName(string targetName)
    {
        Transform[] transforms = FindObjectsOfType<Transform>(true);
        foreach (Transform transform in transforms)
        {
            if (transform.name.Trim() == targetName)
            {
                return transform;
            }
        }

        return null;
    }

    private bool TryGetPredictedTargetPosition(DeadBugTarget target, out Vector3 position)
    {
        Transform manualTarget = GetManualPeakTarget(target);
        if (manualTarget != null)
        {
            position = manualTarget.position;
            return true;
        }

        position = Vector3.zero;
        return false;
    }

    private Transform GetManualPeakTarget(DeadBugTarget target)
    {
        switch (target)
        {
            case DeadBugTarget.LeftKnee:
                return leftKneePeakTarget;
            case DeadBugTarget.LeftHand:
                return leftHandPeakTarget;
            case DeadBugTarget.RightKnee:
                return rightKneePeakTarget;
            case DeadBugTarget.RightHand:
                return rightHandPeakTarget;
            default:
                return null;
        }
    }

    private float SmoothStep(float value)
    {
        return value * value * (3f - 2f * value);
    }

    private bool IsUserAtTarget(DeadBugTarget target, Vector3 targetPosition)
    {
        Transform userBone = GetUserTargetBone(target);
        if (userBone == null)
        {
            return false;
        }

        float matchDistanceForTarget = IsKneeTarget(target) ? kneeMatchDistance : handMatchDistance;
        if (Vector3.Distance(userBone.position, targetPosition) > matchDistanceForTarget)
        {
            return false;
        }

        Transform coachBone = GetCoachTargetBone(target);
        if (requireCoachOverlapForFlash)
        {
            if (coachBone == null || Vector3.Distance(userBone.position, coachBone.position) > coachOverlapDistance)
            {
                return false;
            }
        }

        if (requireCoachNearTargetForFlash)
        {
            if (coachBone == null || Vector3.Distance(coachBone.position, targetPosition) > coachTargetDistance)
            {
                return false;
            }
        }

        return true;
    }

    private Transform GetCoachTargetBone(DeadBugTarget target)
    {
        return coachAnimator.GetBoneTransform(GetBoneForTarget(target));
    }

    private Transform GetUserTargetBone(DeadBugTarget target)
    {
        return userAnimator.GetBoneTransform(GetBoneForTarget(target));
    }

    private HumanBodyBones GetBoneForTarget(DeadBugTarget target)
    {
        switch (target)
        {
            case DeadBugTarget.LeftKnee:
                return HumanBodyBones.LeftLowerLeg;
            case DeadBugTarget.LeftHand:
                return HumanBodyBones.LeftHand;
            case DeadBugTarget.RightKnee:
                return HumanBodyBones.RightLowerLeg;
            case DeadBugTarget.RightHand:
                return HumanBodyBones.RightHand;
            default:
                return HumanBodyBones.Hips;
        }
    }

    private bool IsKneeTarget(DeadBugTarget target)
    {
        return target == DeadBugTarget.LeftKnee || target == DeadBugTarget.RightKnee;
    }

    private void ResetTargetState(StarTargetState state)
    {
        state.hasTarget = false;
        state.arrivedAtTarget = false;
        state.matched = false;
        state.hasPredictedTarget = false;
        state.startPosition = Vector3.zero;
        state.targetPosition = Vector3.zero;
    }

    private void OverlapAvatarRoots()
    {
        Transform coachRoot = coachAnimator.avatarRoot;
        Transform userRoot = userAnimator.avatarRoot;
        if (coachRoot == null || userRoot == null)
        {
            return;
        }

        Transform follower = moveCoachToUser ? coachRoot : userRoot;
        Transform target = moveCoachToUser ? userRoot : coachRoot;

        Vector3 targetPosition = target.position;
        // Each avatar now has its own GroundRoot. Keep the follower's vertical
        // position so the two grounding components can solve height independently.
        if (horizontalPositionOnly)
        {
            targetPosition.y = follower.position.y;
        }

        follower.position = targetPosition;
        if (matchRootRotation)
        {
            follower.rotation = yawRotationOnly
                ? Quaternion.Euler(0f, target.eulerAngles.y, 0f)
                : target.rotation;
        }
    }

    private void ResolveAvatarReferences()
    {
        if (coachAnimator != null && userAnimator != null)
        {
            return;
        }

        Animator[] animators = FindObjectsOfType<Animator>(true);
        foreach (Animator animator in animators)
        {
            if (animator == null)
            {
                continue;
            }

            string avatarName = animator.gameObject.name;
            if (coachAnimator == null && string.Equals(avatarName, "coach", System.StringComparison.OrdinalIgnoreCase))
            {
                coachAnimator = animator;
            }
            else if (userAnimator == null && string.Equals(avatarName, "user", System.StringComparison.OrdinalIgnoreCase))
            {
                userAnimator = animator;
            }

            if (coachAnimator != null && userAnimator != null)
            {
                return;
            }
        }
    }

    private void SaveCoachRootPose()
    {
        if (coachAnimator == null || coachAnimator.avatarRoot == null)
        {
            return;
        }

        Transform coachRoot = coachAnimator.avatarRoot;
        savedCoachRootPosition = coachRoot.position;
        savedCoachRootRotation = coachRoot.rotation;
        hasSavedCoachRoot = true;
    }

    private void RestoreCoachRootPose()
    {
        if (!hasSavedCoachRoot || coachAnimator == null || coachAnimator.avatarRoot == null)
        {
            return;
        }

        Transform coachRoot = coachAnimator.avatarRoot;
        coachRoot.SetPositionAndRotation(savedCoachRootPosition, savedCoachRootRotation);
        hasSavedCoachRoot = false;
    }

    private bool ShouldGroundCoach()
    {
        if (!groundCoachToFloor)
        {
            return false;
        }

        return !groundOnlyWhenCoachLying || IsCoachLying();
    }

    private bool IsCoachLying()
    {
        if (coachAnimator == null)
        {
            return false;
        }

        Transform hips = coachAnimator.GetBoneTransform(HumanBodyBones.Hips);
        Transform chest = coachAnimator.GetBoneTransform(HumanBodyBones.Chest);
        if (chest == null)
        {
            chest = coachAnimator.GetBoneTransform(HumanBodyBones.UpperChest);
        }

        if (hips == null || chest == null)
        {
            return false;
        }

        Vector3 torsoUp = (chest.position - hips.position).normalized;
        return Mathf.Abs(Vector3.Dot(torsoUp, Vector3.up)) < lyingUpDotThreshold;
    }

    private void GroundAvatarToFloor(Animator animator)
    {
        Transform root = animator.avatarRoot;
        if (root == null)
        {
            return;
        }

        if (!TryGetGroundReferenceY(animator, out float lowestY))
        {
            return;
        }

        float targetLowestY = GetFloorY() + floorClearance;
        float yOffset = targetLowestY - lowestY;
        if (Mathf.Abs(yOffset) < 0.001f)
        {
            return;
        }

        Vector3 position = root.position;
        position.y = Mathf.Lerp(position.y, position.y + yOffset, GetGroundBlend());
        root.position = position;
    }

    private float GetFloorY()
    {
        if (useUserFeetAsFloor && TryGetUserFeetY(out float userFeetY))
        {
            return userFeetY;
        }

        return floorY;
    }

    private bool TryGetUserFeetY(out float feetY)
    {
        feetY = float.PositiveInfinity;
        bool found = false;

        HumanBodyBones[] footBones =
        {
            HumanBodyBones.LeftFoot,
            HumanBodyBones.RightFoot,
            HumanBodyBones.LeftToes,
            HumanBodyBones.RightToes
        };

        foreach (HumanBodyBones bone in footBones)
        {
            Transform boneTransform = userAnimator.GetBoneTransform(bone);
            if (boneTransform == null)
            {
                continue;
            }

            feetY = Mathf.Min(feetY, boneTransform.position.y);
            found = true;
        }

        return found;
    }

    private bool TryGetGroundReferenceY(Animator animator, out float lowestY)
    {
        return groundByTorsoOnly
            ? TryGetLowestTorsoY(animator, out lowestY)
            : TryGetLowestBodyY(animator, out lowestY);
    }

    private bool TryGetLowestTorsoY(Animator animator, out float lowestY)
    {
        lowestY = float.PositiveInfinity;
        bool found = false;

        HumanBodyBones[] torsoBones =
        {
            HumanBodyBones.Hips,
            HumanBodyBones.Spine,
            HumanBodyBones.Chest,
            HumanBodyBones.UpperChest,
            HumanBodyBones.Neck,
            HumanBodyBones.Head,
            HumanBodyBones.LeftShoulder,
            HumanBodyBones.RightShoulder,
            HumanBodyBones.LeftUpperArm,
            HumanBodyBones.RightUpperArm,
            HumanBodyBones.LeftUpperLeg,
            HumanBodyBones.RightUpperLeg
        };

        foreach (HumanBodyBones bone in torsoBones)
        {
            Transform boneTransform = animator.GetBoneTransform(bone);
            if (boneTransform == null)
            {
                continue;
            }

            lowestY = Mathf.Min(lowestY, boneTransform.position.y);
            found = true;
        }

        return found;
    }

    private bool TryGetLowestBodyY(Animator animator, out float lowestY)
    {
        lowestY = float.PositiveInfinity;
        bool found = false;

        for (HumanBodyBones bone = HumanBodyBones.Hips; bone < HumanBodyBones.LastBone; bone++)
        {
            Transform boneTransform = animator.GetBoneTransform(bone);
            if (boneTransform == null)
            {
                continue;
            }

            lowestY = Mathf.Min(lowestY, boneTransform.position.y);
            found = true;
        }

        return found;
    }

    private void PlaceStars()
    {
        Transform anchor = GetStarAnchor();
        if (anchor == null)
        {
            return;
        }

        Vector3 center = anchor.position + anchor.up * starVerticalOffset + anchor.forward * starForwardOffset;
        if (blueStar != null)
        {
            blueStar.position = center - anchor.right * starSideOffset;
            blueStar.rotation = Quaternion.LookRotation(anchor.forward, anchor.up);
        }

        if (yellowStar != null)
        {
            yellowStar.position = center + anchor.right * starSideOffset;
            yellowStar.rotation = Quaternion.LookRotation(anchor.forward, anchor.up);
        }
    }

    private Transform GetStarAnchor()
    {
        Transform chest = coachAnimator.GetBoneTransform(HumanBodyBones.Chest);
        if (chest != null) return chest;

        Transform hips = coachAnimator.GetBoneTransform(HumanBodyBones.Hips);
        if (hips != null) return hips;

        return coachAnimator.avatarRoot;
    }

    private float CalculateSideScore(HumanBodyBones[] bones)
    {
        float distanceSum = 0f;
        int count = 0;

        foreach (HumanBodyBones bone in bones)
        {
            Transform coachBone = coachAnimator.GetBoneTransform(bone);
            Transform userBone = userAnimator.GetBoneTransform(bone);
            if (coachBone == null || userBone == null)
            {
                continue;
            }

            distanceSum += Vector3.Distance(coachBone.position, userBone.position);
            count++;
        }

        if (count == 0)
        {
            return 0f;
        }

        float averageDistance = distanceSum / count;
        return Mathf.Clamp01(1f - averageDistance / matchDistance);
    }

    private bool UpdateMatchedState(bool current, float score)
    {
        if (current)
        {
            return score > releaseScore;
        }

        return score >= flashScore;
    }

    private void CacheStar(Transform star, ref Vector3 baseScale, ref ParticleSystem[] particles, ref Light[] lights)
    {
        if (star == null)
        {
            particles = new ParticleSystem[0];
            lights = new Light[0];
            return;
        }

        baseScale = star.localScale;
        particles = star.GetComponentsInChildren<ParticleSystem>(true);
        lights = star.GetComponentsInChildren<Light>(true);
        ApplyStarFeedback(star, baseScale, particles, lights, false);
    }

    private void ApplyStarFeedback(
        Transform star,
        Vector3 baseScale,
        ParticleSystem[] particles,
        Light[] lights,
        bool matched)
    {
        if (star == null)
        {
            return;
        }

        float scale = matched
            ? Mathf.Lerp(idleScale, flashScale, 0.5f + Mathf.Sin(Time.time * flashPulseSpeed) * 0.5f)
            : idleScale;

        star.localScale = baseScale * scale;

        foreach (ParticleSystem particle in particles)
        {
            if (particle == null)
            {
                continue;
            }

            if (matched)
            {
                if (!particle.isPlaying) particle.Play(true);
            }
            else if (particle.isPlaying)
            {
                particle.Stop(true, ParticleSystemStopBehavior.StopEmitting);
            }
        }

        foreach (Light starLight in lights)
        {
            if (starLight != null)
            {
                starLight.enabled = matched;
            }
        }
    }

    private float GetFrameBlend()
    {
        return 1f - Mathf.Exp(-smoothSpeed * Time.deltaTime);
    }

    private float GetGroundBlend()
    {
        return 1f - Mathf.Exp(-groundSmoothSpeed * Time.deltaTime);
    }

    private readonly struct RendererState
    {
        public RendererState(Renderer renderer, bool wasEnabled)
        {
            Renderer = renderer;
            WasEnabled = wasEnabled;
        }

        public Renderer Renderer { get; }
        public bool WasEnabled { get; }
    }
}
