using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class MotionPlaybackController : MonoBehaviour
{
    [SerializeField] private Animator userAnimator;
    [SerializeField] private Animator coachAnimator;
    [SerializeField] private string coachActionStateName = "action1";
    [SerializeField] private string coachClipName = "animation_pose";
    [SerializeField] private bool playOnStart;

    private RecordedMotionClip clip;
    private bool isPlaying;
    private float playbackTime;
    private float duration;
    private float coachClipLength = 1f;
    private int currentFrameIndex;

    private readonly Dictionary<string, Transform> userBones = new Dictionary<string, Transform>();
    private readonly Dictionary<string, Transform> coachBones = new Dictionary<string, Transform>();
    private readonly Dictionary<string, Vector3> userInitialPositions = new Dictionary<string, Vector3>();
    private readonly Dictionary<string, Quaternion> userInitialRotations =
        new Dictionary<string, Quaternion>();

    private readonly HumanBodyBones[] trackedBones =
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
        HumanBodyBones.LeftLowerArm,
        HumanBodyBones.RightLowerArm,
        HumanBodyBones.LeftHand,
        HumanBodyBones.RightHand,

        HumanBodyBones.LeftUpperLeg,
        HumanBodyBones.RightUpperLeg,
        HumanBodyBones.LeftLowerLeg,
        HumanBodyBones.RightLowerLeg,
        HumanBodyBones.LeftFoot,
        HumanBodyBones.RightFoot
    };

    public Animator UserAnimator => userAnimator;
    public Animator CoachAnimator => coachAnimator;

    private string SavePath => new LegacyReplayV2Migrator().LegacyPath;

    private void Start()
    {
        CacheUserBones();
        CacheCoachBones();
        CacheCoachClipLength();

        if (!LoadRecordedMotion())
        {
            return;
        }

        // Playback writes the retargeted user pose directly to the bone transforms.
        // Prevent Animator evaluation from restoring a different bind pose afterward.
        if (userAnimator != null)
        {
            userAnimator.enabled = false;
        }

        // New recordings contain the exact coach pose on the same timeline.
        // Disable Animator evaluation so it cannot overwrite those bone poses.
        if (HasRecordedCoachPose && coachAnimator != null)
        {
            coachAnimator.enabled = false;
        }

        RestartPlayback();

        if (!playOnStart)
        {
            PausePlayback();
        }
    }

    private void Update()
    {
        if (!isPlaying || clip == null || clip.frames == null || clip.frames.Count == 0)
        {
            return;
        }

        playbackTime += Time.deltaTime;

        if (playbackTime >= duration)
        {
            playbackTime = duration;
            isPlaying = false;
        }

        ApplyUserPose(playbackTime);
        ApplyCoachPose(playbackTime);
    }

    public void TogglePause()
    {
        if (!enabled)
        {
            return;
        }

        if (clip == null || clip.frames == null || clip.frames.Count == 0)
        {
            return;
        }

        if (!isPlaying && playbackTime >= duration)
        {
            RestartPlayback();
            return;
        }

        isPlaying = !isPlaying;
    }

    public void PausePlayback()
    {
        if (!enabled)
        {
            return;
        }

        isPlaying = false;
    }

    public void ResumePlayback()
    {
        if (!enabled)
        {
            return;
        }

        if (clip == null || clip.frames == null || clip.frames.Count == 0)
        {
            return;
        }

        if (playbackTime >= duration)
        {
            RestartPlayback();
            return;
        }

        isPlaying = true;
    }

    public void RestartPlayback()
    {
        if (!enabled)
        {
            return;
        }

        if (clip == null || clip.frames == null || clip.frames.Count == 0)
        {
            return;
        }

        playbackTime = 0f;
        currentFrameIndex = 0;
        duration = Mathf.Max(clip.duration, 0.01f);
        isPlaying = true;

        ApplyUserPose(0f);
        ApplyCoachPose(0f);
    }

    public float Duration => duration;

    public float PlaybackTime => playbackTime;

    public bool IsPlaying => isPlaying;

    public bool HasClip => clip != null && clip.frames != null && clip.frames.Count > 0;

    private bool HasRecordedCoachPose => HasClip &&
        clip.frames[0].coachBones != null && clip.frames[0].coachBones.Count > 0;

    public float NormalizedProgress
    {
        get
        {
            if (!HasClip || duration <= 0f)
            {
                return 0f;
            }

            return Mathf.Clamp01(playbackTime / duration);
        }
    }

    public void SeekNormalized(float normalizedProgress)
    {
        if (!enabled)
        {
            return;
        }

        if (!HasClip)
        {
            return;
        }

        duration = Mathf.Max(duration, 0.01f);
        playbackTime = Mathf.Clamp01(normalizedProgress) * duration;
        currentFrameIndex = FindFrameIntervalIndex(playbackTime);

        ApplyUserPose(playbackTime);
        ApplyCoachPose(playbackTime);
    }

    private bool LoadRecordedMotion()
    {
        if (!File.Exists(SavePath))
        {
            Debug.LogWarning($"No recorded motion file found: {SavePath}");
            return false;
        }

        string json = File.ReadAllText(SavePath);
        clip = JsonUtility.FromJson<RecordedMotionClip>(json);

        if (clip == null || !clip.completed ||
            clip.frames == null || clip.frames.Count == 0)
        {
            Debug.LogWarning("Legacy recorded motion file is incomplete, empty or invalid.");
            return false;
        }

        duration = clip.duration;
        currentFrameIndex = 0;

        if (duration <= 0f)
        {
            duration = clip.frames[clip.frames.Count - 1].time;
        }

        Debug.Log($"Loaded recorded motion: {clip.frames.Count} frames, {duration:F2}s");
        return true;
    }

    private void CacheUserBones()
    {
        userBones.Clear();
        userInitialPositions.Clear();
        userInitialRotations.Clear();

        if (userAnimator == null)
        {
            Debug.LogError("MotionPlaybackController needs a user Animator.");
            return;
        }

        foreach (HumanBodyBones bone in trackedBones)
        {
            Transform boneTransform = userAnimator.GetBoneTransform(bone);
            if (boneTransform != null)
            {
                string boneName = bone.ToString();
                userBones[boneName] = boneTransform;
                userInitialPositions[boneName] = boneTransform.localPosition;
                userInitialRotations[boneName] = boneTransform.localRotation;
            }
        }
    }

    private void CacheCoachClipLength()
    {
        if (coachAnimator == null || coachAnimator.runtimeAnimatorController == null)
        {
            return;
        }

        foreach (AnimationClip animationClip in coachAnimator.runtimeAnimatorController.animationClips)
        {
            if (animationClip.name == coachClipName)
            {
                coachClipLength = animationClip.length;
                return;
            }
        }

        Debug.LogWarning($"Coach clip not found by name: {coachClipName}");
    }

    private void CacheCoachBones()
    {
        coachBones.Clear();

        if (coachAnimator == null)
        {
            Debug.LogError("MotionPlaybackController needs a coach Animator.");
            return;
        }

        foreach (HumanBodyBones bone in trackedBones)
        {
            Transform boneTransform = coachAnimator.GetBoneTransform(bone);
            if (boneTransform != null)
            {
                coachBones[bone.ToString()] = boneTransform;
            }
        }
    }

    private void ApplyCoachPose(float time)
    {
        if (coachAnimator == null)
        {
            return;
        }

        if (HasRecordedCoachPose)
        {
            ApplyRecordedPose(time, coachBones, true, false);
            return;
        }

        float normalizedTime = Mathf.Clamp01(time / coachClipLength);
        coachAnimator.Play(coachActionStateName, 0, normalizedTime);
        coachAnimator.Update(0f);
    }

    private void ApplyUserPose(float time)
    {
        // The recording may come from either the female or male practice rig,
        // while the Playback scene owns its own avatar. Transfer rotation
        // deltas from the recorded reference pose instead of copying absolute
        // local rotations across potentially different skeleton bind poses.
        ApplyRecordedPose(time, userBones, false, true);
    }

    private void ApplyRecordedPose(
        float time,
        Dictionary<string, Transform> targetBones,
        bool useCoachBones,
        bool preserveTargetBonePositions)
    {
        if (clip.frames.Count == 1)
        {
            ApplyFrame(clip.frames[0], targetBones, useCoachBones, preserveTargetBonePositions);
            return;
        }

        currentFrameIndex = FindFrameIntervalIndex(time);
        RecordedMotionFrame a = clip.frames[currentFrameIndex];
        RecordedMotionFrame b = clip.frames[currentFrameIndex + 1];

        float frameDuration = Mathf.Max(b.time - a.time, 0.0001f);
        float t = Mathf.Clamp01((time - a.time) / frameDuration);

        List<RecordedBonePose> posesA = useCoachBones ? a.coachBones : a.bones;
        List<RecordedBonePose> posesB = useCoachBones ? b.coachBones : b.bones;
        if (posesA == null || posesB == null)
        {
            return;
        }

        for (int i = 0; i < posesA.Count; i++)
        {
            RecordedBonePose poseA = posesA[i];
            RecordedBonePose poseB = FindBonePose(posesB, poseA.bone);

            if (poseB == null || !targetBones.TryGetValue(poseA.bone, out Transform boneTransform))
            {
                continue;
            }

            Vector3 position = Vector3.Lerp(poseA.localPosition, poseB.localPosition, t);
            Quaternion rotation = Quaternion.Slerp(
                NormalizeRotation(poseA.localRotation),
                NormalizeRotation(poseB.localRotation),
                t);
            ApplyBonePose(
                poseA.bone,
                position,
                rotation,
                boneTransform,
                preserveTargetBonePositions);
        }
    }

    private void ApplyFrame(
        RecordedMotionFrame frame,
        Dictionary<string, Transform> targetBones,
        bool useCoachBones,
        bool preserveTargetBonePositions)
    {
        List<RecordedBonePose> poses = useCoachBones ? frame.coachBones : frame.bones;
        if (poses == null)
        {
            return;
        }

        foreach (RecordedBonePose pose in poses)
        {
            if (targetBones.TryGetValue(pose.bone, out Transform boneTransform))
            {
                ApplyBonePose(
                    pose.bone,
                    pose.localPosition,
                    NormalizeRotation(pose.localRotation),
                    boneTransform,
                    preserveTargetBonePositions);
            }
        }
    }

    private void ApplyBonePose(
        string boneName,
        Vector3 recordedPosition,
        Quaternion recordedRotation,
        Transform boneTransform,
        bool preserveTargetBonePositions)
    {
        if (!preserveTargetBonePositions)
        {
            boneTransform.localPosition = recordedPosition;
            boneTransform.localRotation = recordedRotation;
            return;
        }

        RecordedBonePose sourceReference = FindBonePose(clip.frames[0].bones, boneName);
        if (sourceReference == null ||
            !userInitialRotations.TryGetValue(
                boneName,
                out Quaternion targetInitialRotation))
        {
            return;
        }

        boneTransform.localRotation = RetargetLocalRotation(
            NormalizeRotation(sourceReference.localRotation),
            recordedRotation,
            targetInitialRotation);

        // Bone lengths and offsets belong to the playback rig. Copying
        // absolute positions from another avatar deforms the mesh, so only
        // transfer the hips displacement relative to the first recorded frame.
        if (boneName == HumanBodyBones.Hips.ToString() &&
            userInitialPositions.TryGetValue(
                boneName,
                out Vector3 targetInitialPosition))
        {
            boneTransform.localPosition = targetInitialPosition +
                (recordedPosition - sourceReference.localPosition);
        }
    }

    public static Quaternion RetargetLocalRotation(
        Quaternion sourceReference,
        Quaternion recordedRotation,
        Quaternion targetReference)
    {
        Quaternion sourceDelta =
            Quaternion.Inverse(NormalizeRotation(sourceReference)) *
            NormalizeRotation(recordedRotation);
        return NormalizeRotation(
            NormalizeRotation(targetReference) * sourceDelta);
    }

    private int FindFrameIntervalIndex(float time)
    {
        int frameCount = clip?.frames?.Count ?? 0;
        if (frameCount <= 1)
        {
            return 0;
        }

        int lastInterval = frameCount - 2;
        if (time <= clip.frames[0].time)
        {
            return 0;
        }

        if (time >= clip.frames[frameCount - 1].time)
        {
            return lastInterval;
        }

        int index = Mathf.Clamp(currentFrameIndex, 0, lastInterval);
        if (clip.frames[index].time <= time &&
            clip.frames[index + 1].time >= time)
        {
            return index;
        }

        // Normal playback advances monotonically, so this path is O(1) for
        // almost every frame and only walks over samples skipped by a hitch.
        if (time > clip.frames[index + 1].time)
        {
            while (index < lastInterval &&
                time > clip.frames[index + 1].time)
            {
                index++;
            }

            return index;
        }

        // Seeking backwards uses binary search instead of scanning the entire
        // recording from its first frame.
        int low = 0;
        int high = lastInterval;
        while (low <= high)
        {
            int middle = low + (high - low) / 2;
            if (clip.frames[middle].time <= time &&
                clip.frames[middle + 1].time >= time)
            {
                return middle;
            }

            if (clip.frames[middle].time > time)
            {
                high = middle - 1;
            }
            else
            {
                low = middle + 1;
            }
        }

        return Mathf.Clamp(low, 0, lastInterval);
    }

    private static Quaternion NormalizeRotation(Quaternion rotation)
    {
        float magnitudeSquared = rotation.x * rotation.x +
            rotation.y * rotation.y +
            rotation.z * rotation.z +
            rotation.w * rotation.w;

        if (magnitudeSquared < 0.000001f)
        {
            return Quaternion.identity;
        }

        float inverseMagnitude = 1f / Mathf.Sqrt(magnitudeSquared);
        return new Quaternion(
            rotation.x * inverseMagnitude,
            rotation.y * inverseMagnitude,
            rotation.z * inverseMagnitude,
            rotation.w * inverseMagnitude);
    }

    private RecordedBonePose FindBonePose(List<RecordedBonePose> poses, string boneName)
    {
        if (poses == null)
        {
            return null;
        }

        foreach (RecordedBonePose pose in poses)
        {
            if (pose.bone == boneName)
            {
                return pose;
            }
        }

        return null;
    }
}
