using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public sealed class LegacyReplayV2Migrator
{
    public string LegacyPath { get; }

    public LegacyReplayV2Migrator(string legacyPath = null)
    {
        LegacyPath = string.IsNullOrWhiteSpace(legacyPath)
            ? Path.Combine(Application.persistentDataPath, "latest_user_motion.json")
            : legacyPath;
    }

    public bool TryMigrate(
        ReplayManifest manifestTemplate,
        Animator userAnimator,
        Animator coachAnimator,
        Transform userRoot,
        Transform coachRoot,
        Transform sessionOrigin,
        ReplayRepository repository,
        out ReplayManifest migratedManifest,
        out string error)
    {
        migratedManifest = null;
        if (!File.Exists(LegacyPath))
        {
            error = "No legacy V2 replay exists.";
            return false;
        }

        if (manifestTemplate == null ||
            string.IsNullOrWhiteSpace(manifestTemplate.actionId) ||
            manifestTemplate.actionId == "unknown" ||
            string.IsNullOrWhiteSpace(manifestTemplate.avatarId))
        {
            error =
                "Legacy V2 replay has no reliable action/avatar identity and cannot be migrated automatically.";
            return false;
        }

        if (!TryValidateHumanoid(userAnimator, "user", out error) ||
            !TryValidateHumanoid(coachAnimator, "coach", out error))
        {
            return false;
        }

        RecordedMotionClip legacy;
        try
        {
            legacy = JsonUtility.FromJson<RecordedMotionClip>(File.ReadAllText(LegacyPath));
        }
        catch (Exception exception)
        {
            error = $"Legacy replay JSON is invalid: {exception.Message}";
            return false;
        }

        if (legacy == null || legacy.formatVersion != 2 || !legacy.completed ||
            legacy.frames == null || legacy.frames.Count == 0)
        {
            error = "Legacy replay is incomplete, empty, or not V2.";
            return false;
        }

        var userHandler = new HumanPoseHandler(userAnimator.avatar, userAnimator.transform);
        var coachHandler = new HumanPoseHandler(coachAnimator.avatar, coachAnimator.transform);
        try
        {
            var frames = new List<ReplayFrame>(legacy.frames.Count);
            float firstTime = legacy.frames[0].time;
            double previous = -1d;
            for (int i = 0; i < legacy.frames.Count; i++)
            {
                RecordedMotionFrame legacyFrame = legacy.frames[i];
                double timestamp = Math.Max(0d, legacyFrame.time - firstTime);
                if (i > 0 && timestamp <= previous)
                {
                    error = "Legacy replay timestamps are not strictly monotonic.";
                    return false;
                }

                ApplyLegacyPose(userAnimator, legacyFrame.bones);
                ApplyLegacyPose(coachAnimator, legacyFrame.coachBones);
                userAnimator.Update(0f);
                coachAnimator.Update(0f);
                frames.Add(new ReplayFrame
                {
                    timestamp = timestamp,
                    user = CaptureActor(userHandler, userAnimator, userRoot, sessionOrigin),
                    coach = CaptureActor(coachHandler, coachAnimator, coachRoot, sessionOrigin)
                });
                previous = timestamp;
            }

            ReplayManifest manifest = manifestTemplate.Clone();
            manifest.legacy = true;
            manifest.compatibilityNotice =
                "Migrated from V2 bone data; presentation roots were not present in the source.";
            manifest.sampleRate = Mathf.Max(1f, legacy.fps);
            migratedManifest = (repository ?? new ReplayRepository()).CommitCompleted(
                manifest,
                frames);
            error = null;
            return true;
        }
        catch (Exception exception)
        {
            error = $"Legacy replay migration failed: {exception.Message}";
            return false;
        }
        finally
        {
            userHandler.Dispose();
            coachHandler.Dispose();
        }
    }

    private static void ApplyLegacyPose(
        Animator animator,
        List<RecordedBonePose> poses)
    {
        if (poses == null)
        {
            return;
        }

        for (int i = 0; i < poses.Count; i++)
        {
            RecordedBonePose pose = poses[i];
            if (!Enum.TryParse(pose.bone, out HumanBodyBones bone))
            {
                continue;
            }

            Transform boneTransform = animator.GetBoneTransform(bone);
            if (boneTransform != null)
            {
                boneTransform.localPosition = pose.localPosition;
                boneTransform.localRotation =
                    ReplayDataValidator.NormalizeFinite(pose.localRotation);
            }
        }
    }

    private static ReplayActorPose CaptureActor(
        HumanPoseHandler handler,
        Animator animator,
        Transform presentationRoot,
        Transform sessionOrigin)
    {
        var pose = new HumanPose();
        handler.GetHumanPose(ref pose);
        float humanScale = animator == null || animator.humanScale <= 0.0001f
            ? 1.0f
            : animator.humanScale;
        return new ReplayActorPose
        {
            bodyPosition = presentationRoot.InverseTransformPoint(
                    pose.bodyPosition * humanScale)
                / humanScale,
            bodyRotation = ReplayDataValidator.NormalizeFinite(
                Quaternion.Inverse(presentationRoot.rotation) * pose.bodyRotation),
            muscles = (float[])pose.muscles.Clone(),
            presentationRootPosition = sessionOrigin != null
                ? sessionOrigin.InverseTransformPoint(presentationRoot.position)
                : presentationRoot.position,
            presentationRootRotation = sessionOrigin != null
                ? Quaternion.Inverse(sessionOrigin.rotation) * presentationRoot.rotation
                : presentationRoot.rotation,
            trackingValid = true,
            trackingConfidence = 1f
        };
    }

    private static bool TryValidateHumanoid(
        Animator animator,
        string actor,
        out string error)
    {
        if (animator == null || animator.avatar == null ||
            !animator.avatar.isValid || !animator.isHuman || !animator.avatar.isHuman)
        {
            error = $"Legacy migration needs a valid Humanoid {actor} Avatar.";
            return false;
        }

        error = null;
        return true;
    }
}
