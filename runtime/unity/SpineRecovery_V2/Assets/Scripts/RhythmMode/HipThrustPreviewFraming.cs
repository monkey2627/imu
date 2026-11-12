using UnityEngine;

// Fit the transparent header preview to the current pose, including standing preparation.
[DefaultExecutionOrder(200)]
public sealed class HipThrustPreviewFraming : MonoBehaviour
{
    public Animator avatar;
    public Camera previewCamera;
    [Min(0f)] public float framingPadding = 0.22f;
    static readonly HumanBodyBones[] Bones = {
        HumanBodyBones.Head, HumanBodyBones.Hips, HumanBodyBones.Chest,
        HumanBodyBones.LeftHand, HumanBodyBones.RightHand,
        HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot,
        HumanBodyBones.LeftLowerLeg, HumanBodyBones.RightLowerLeg,
        HumanBodyBones.LeftLowerArm, HumanBodyBones.RightLowerArm
    };
    void LateUpdate()
    {
        if (avatar == null || !avatar.isHuman || previewCamera == null) return;
        Quaternion rotation = previewCamera.transform.rotation;
        Quaternion inverse = Quaternion.Inverse(rotation);
        Vector3 min = Vector3.one * float.PositiveInfinity, max = Vector3.one * float.NegativeInfinity;
        foreach (var bone in Bones)
        {
            Transform target = avatar.GetBoneTransform(bone);
            if (target == null) continue;
            Vector3 p = inverse * target.position;
            min = Vector3.Min(min, p); max = Vector3.Max(max, p);
        }
        if (float.IsInfinity(min.x)) return;
        Vector3 center = (min + max) * 0.5f;
        center.z = min.z - 3f;
        previewCamera.transform.position = rotation * center;
        previewCamera.orthographic = true;
        previewCamera.orthographicSize = Mathf.Max((max.y - min.y + framingPadding) / 2f,
            (max.x - min.x + framingPadding) / (2f * previewCamera.aspect));
    }
}
