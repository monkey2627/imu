using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;

// Opt-in for the new game scenes; legacy scenes retain their original calibration.
public sealed class SupineGameCalibration : MonoBehaviour
{
    [Tooltip("Temporarily skip calibration to preview the game flow. Disable to restore lying calibration.")]
    public bool skipCalibrationForPreview;
    public AnimationClip referenceClip;
    public float referenceSeconds = 6.8f;
    public const string Instruction = "请仰卧屈膝、双脚着地，双臂放身体两侧";
    public bool leftSideBridge;
    public string PoseInstruction => leftSideBridge ? "请按教练左侧支撑，左肘与左脚支撑，肩髋叠放" : Instruction;
    public bool IsPoseReady(Animator avatar) => leftSideBridge
        ? SideBridgePoseGeometry.TryMeasure(avatar,out var pose)&&pose.IsLeftSupport
        : IsLying(avatar);
    private PlayableGraph graph;

    public bool Prepare(Animator coach)
    {
        Release();
        if (referenceClip == null || coach == null || !coach.isHuman) return false;
        coach.enabled = true;
        coach.cullingMode = AnimatorCullingMode.AlwaysAnimate;
        graph = PlayableGraph.Create("Supine calibration reference");
        graph.SetTimeUpdateMode(DirectorUpdateMode.Manual);
        var clip = AnimationClipPlayable.Create(graph, referenceClip);
        clip.SetTime(referenceSeconds);
        clip.SetSpeed(0);
        AnimationPlayableOutput.Create(graph, "Reference", coach).SetSourcePlayable(clip);
        graph.Play();
        graph.Evaluate(0);
        return true;
    }

    public void SampleReference() { if (graph.IsValid()) graph.Evaluate(0); }

    public static bool IsLying(Animator avatar)
    {
        if (avatar == null || !avatar.isHuman) return false;
        Transform hips = avatar.GetBoneTransform(HumanBodyBones.Hips);
        Transform chest = avatar.GetBoneTransform(HumanBodyBones.Chest) ?? avatar.GetBoneTransform(HumanBodyBones.Spine);
        if (hips == null || chest == null) return false;
        Vector3 torso = chest.position - hips.position;
        Transform leftKnee = avatar.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
        Transform rightKnee = avatar.GetBoneTransform(HumanBodyBones.RightLowerLeg);
        Transform thigh = avatar.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
        if (leftKnee == null || rightKnee == null || thigh == null) return false;
        float legLength = Vector3.Distance(thigh.position, leftKnee.position);
        float kneeHeight = (leftKnee.position.y + rightKnee.position.y) * 0.5f - hips.position.y;
        // Bent knees above the hips distinguish the requested supine pose from four-point kneeling.
        return torso.sqrMagnitude > 0.001f && Mathf.Abs(Vector3.Dot(torso.normalized, Vector3.up)) < 0.45f
            && legLength > 0.001f && kneeHeight > legLength * 0.15f;
    }

    public void Release() { if (graph.IsValid()) graph.Destroy(); }
    private void OnDisable() { Release(); }
}
