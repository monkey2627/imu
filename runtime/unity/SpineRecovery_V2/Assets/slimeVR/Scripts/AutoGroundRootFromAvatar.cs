using UnityEngine;

[DefaultExecutionOrder(10002)]
public sealed class AutoGroundRootFromAvatar : MonoBehaviour
{
    [SerializeField] private Transform targetRoot;
    [SerializeField] private Transform avatarRoot;
    [SerializeField] private Animator avatarAnimator;
    [SerializeField] private float floorY = 0f;
    [SerializeField] private float groundOffset = 0.01f;
    [SerializeField] private float smoothing = 18f;
    [SerializeField] private float maxCorrectionPerSecond = 3f;
    [SerializeField] private bool useHumanoidBones = true;
    [SerializeField] private bool useRendererBounds = false;
    [SerializeField] private bool includeInactiveRenderers = false;
    [SerializeField] private bool lockHorizontalPosition = true;
    [SerializeField] private bool lockRootRotation = true;

    private Renderer[] renderers;
    private Transform[] groundBones;
    private Vector3 initialRootPosition;
    private Quaternion initialRootRotation;
    private ReplayPlacementController replayPlacement;

    private void Awake()
    {
        if (targetRoot == null)
        {
            targetRoot = transform;
        }

        if (avatarRoot == null)
        {
            avatarRoot = targetRoot;
        }

        if (avatarAnimator == null)
        {
            avatarAnimator = avatarRoot.GetComponentInChildren<Animator>();
        }

        CacheHumanoidBones();
        RefreshRenderers();

        initialRootPosition = targetRoot.position;
        initialRootRotation = targetRoot.rotation;
        replayPlacement = GetComponentInParent<ReplayPlacementController>();
    }

    private void LateUpdate()
    {
        if (targetRoot == null || avatarRoot == null)
        {
            return;
        }

        if (useHumanoidBones && (groundBones == null || groundBones.Length == 0))
        {
            CacheHumanoidBones();
        }

        if (useRendererBounds && (renderers == null || renderers.Length == 0))
        {
            RefreshRenderers();
        }

        if (!TryGetLowestY(out float lowestY))
        {
            return;
        }

        float desiredDelta = floorY + groundOffset - lowestY;
        float deltaTime = Mathf.Max(Time.deltaTime, 0.0001f);
        float smoothedDelta = Mathf.Lerp(0f, desiredDelta, 1f - Mathf.Exp(-smoothing * deltaTime));
        float limitedDelta = Mathf.Clamp(
            smoothedDelta,
            -maxCorrectionPerSecond * deltaTime,
            maxCorrectionPerSecond * deltaTime);

        targetRoot.position += Vector3.up * limitedDelta;

        // Replay placement owns horizontal position and root rotation. This
        // component may still make a bounded vertical grounding correction,
        // but it must never restore the pre-placement X/Z/yaw values.
        bool replayOwnsSpatialRoot = replayPlacement != null;
        if (lockHorizontalPosition && !replayOwnsSpatialRoot)
        {
            Vector3 position = targetRoot.position;
            position.x = initialRootPosition.x;
            position.z = initialRootPosition.z;
            targetRoot.position = position;
        }

        if (lockRootRotation && !replayOwnsSpatialRoot)
        {
            targetRoot.rotation = initialRootRotation;
        }
    }

    public void RebaseAfterPlacement(float newFloorY)
    {
        if (targetRoot == null)
        {
            targetRoot = transform;
        }

        floorY = newFloorY;
        initialRootPosition = targetRoot.position;
        initialRootRotation = targetRoot.rotation;
        replayPlacement = GetComponentInParent<ReplayPlacementController>();
    }

    [ContextMenu("Refresh Renderers")]
    private void RefreshRenderers()
    {
        if (avatarRoot == null)
        {
            renderers = new Renderer[0];
            return;
        }

        renderers = avatarRoot.GetComponentsInChildren<Renderer>(includeInactiveRenderers);
    }

    private bool TryGetLowestY(out float lowestY)
    {
        lowestY = float.PositiveInfinity;

        bool found = false;

        if (useHumanoidBones && groundBones != null)
        {
            for (int i = 0; i < groundBones.Length; i++)
            {
                Transform bone = groundBones[i];
                if (bone == null)
                {
                    continue;
                }

                lowestY = Mathf.Min(lowestY, bone.position.y);
                found = true;
            }
        }

        if (!useRendererBounds)
        {
            return found;
        }

        for (int i = 0; i < renderers.Length; i++)
        {
            Renderer renderer = renderers[i];
            if (renderer == null || !renderer.enabled)
            {
                continue;
            }

            lowestY = Mathf.Min(lowestY, renderer.bounds.min.y);
            found = true;
        }

        return found;
    }

    private void CacheHumanoidBones()
    {
        if (avatarAnimator == null || !avatarAnimator.isHuman)
        {
            groundBones = new Transform[0];
            return;
        }

        groundBones = new[]
        {
            avatarAnimator.GetBoneTransform(HumanBodyBones.LeftFoot),
            avatarAnimator.GetBoneTransform(HumanBodyBones.RightFoot),
            avatarAnimator.GetBoneTransform(HumanBodyBones.LeftToes),
            avatarAnimator.GetBoneTransform(HumanBodyBones.RightToes),
            avatarAnimator.GetBoneTransform(HumanBodyBones.LeftHand),
            avatarAnimator.GetBoneTransform(HumanBodyBones.RightHand),
            avatarAnimator.GetBoneTransform(HumanBodyBones.Head),
            avatarAnimator.GetBoneTransform(HumanBodyBones.Hips),
            avatarAnimator.GetBoneTransform(HumanBodyBones.Spine),
            avatarAnimator.GetBoneTransform(HumanBodyBones.Chest),
            avatarAnimator.GetBoneTransform(HumanBodyBones.LeftLowerLeg),
            avatarAnimator.GetBoneTransform(HumanBodyBones.RightLowerLeg),
            avatarAnimator.GetBoneTransform(HumanBodyBones.LeftUpperLeg),
            avatarAnimator.GetBoneTransform(HumanBodyBones.RightUpperLeg),
            avatarAnimator.GetBoneTransform(HumanBodyBones.LeftLowerArm),
            avatarAnimator.GetBoneTransform(HumanBodyBones.RightLowerArm),
            avatarAnimator.GetBoneTransform(HumanBodyBones.LeftUpperArm),
            avatarAnimator.GetBoneTransform(HumanBodyBones.RightUpperArm),
        };
    }
}
