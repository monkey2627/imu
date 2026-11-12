using System;
using System.Collections.Generic;
using UnityEngine;

[DefaultExecutionOrder(-40)]
public sealed class ReplayPlacementController : MonoBehaviour
{
    // Replay content stays at its authored scene pose in mobile display mode.
    [SerializeField] private Transform replayRoot;
    [SerializeField] private ReplayPlayer player;
    private readonly List<ContentPose> contentPoses = new List<ContentPose>();

    public bool IsPlaced { get; private set; }
    public bool UsedFallback { get; private set; }
    public Transform ReplayRoot => replayRoot;

    public void Configure(
        Transform spatialRoot,
        ReplayPlayer replayPlayer,
        Camera camera)
    {
        replayRoot = spatialRoot;
        player = replayPlayer;
        IsPlaced = false;
        UsedFallback = false;
    }

    public void UseConfiguredPoseWithoutSpatialRelocation()
    {
        IsPlaced = false;
    }

    public void ConfigureContent(params Transform[] contentRoots)
    {
        contentPoses.Clear();
        if (replayRoot == null || contentRoots == null)
        {
            return;
        }

        var seen = new HashSet<Transform>();
        foreach (Transform contentRoot in contentRoots)
        {
            if (contentRoot == null || contentRoot == replayRoot ||
                !seen.Add(contentRoot))
            {
                continue;
            }

            contentPoses.Add(ContentPose.Capture(replayRoot, contentRoot));
        }
    }

    public void ReplaceContentTransform(Transform previous, Transform replacement)
    {
        if (replayRoot == null || replacement == null)
        {
            return;
        }

        for (int i = 0; i < contentPoses.Count; i++)
        {
            if (contentPoses[i].Target == previous)
            {
                contentPoses[i] = ContentPose.Capture(replayRoot, replacement);
                return;
            }
        }

        contentPoses.Add(ContentPose.Capture(replayRoot, replacement));
    }

    private void OnEnable()
    {
    }

    private void Update()
    {
        if (IsPlaced || replayRoot == null || player == null || !player.HasReplay)
        {
            return;
        }

        Place(replayRoot.position, replayRoot.rotation, false);
    }

    public void Place(Vector3 position, Quaternion rotation, bool usedFallback)
    {
        if (replayRoot == null)
        {
            throw new InvalidOperationException("ReplayRoot is not configured.");
        }

        replayRoot.SetPositionAndRotation(
            position,
            ReplayDataValidator.NormalizeFinite(rotation));
        ApplyContentPoses();
        UsedFallback = usedFallback;
        IsPlaced = true;

        foreach (AutoGroundRootFromAvatar grounding in
                 replayRoot.GetComponentsInChildren<AutoGroundRootFromAvatar>(true))
        {
            grounding.RebaseAfterPlacement(position.y);
        }

        player?.MarkPlacementReady();
    }

    public void ResetPlacement()
    {
        IsPlaced = false;
        UsedFallback = false;
    }

    private void ApplyContentPoses()
    {
        foreach (ContentPose content in contentPoses)
        {
            if (content.Target == null)
            {
                continue;
            }

            content.Target.SetPositionAndRotation(
                replayRoot.TransformPoint(content.LocalPosition),
                replayRoot.rotation * content.LocalRotation);
            content.Target.localScale = content.LocalScale;
        }
    }

    private readonly struct ContentPose
    {
        public readonly Transform Target;
        public readonly Vector3 LocalPosition;
        public readonly Quaternion LocalRotation;
        public readonly Vector3 LocalScale;

        private ContentPose(
            Transform target,
            Vector3 localPosition,
            Quaternion localRotation,
            Vector3 localScale)
        {
            Target = target;
            LocalPosition = localPosition;
            LocalRotation = localRotation;
            LocalScale = localScale;
        }

        public static ContentPose Capture(Transform anchor, Transform target)
        {
            return new ContentPose(
                target,
                anchor.InverseTransformPoint(target.position),
                Quaternion.Inverse(anchor.rotation) * target.rotation,
                target.localScale);
        }
    }
}
