using System;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class RecordedCoachOrientationVerification
{
    [MenuItem("Tools/Recorded Coach/Verify Recorded Bone Orientation")]
    public static void Run()
    {
        Directory.CreateDirectory(RecordedCoachSceneTools.Output);
        var report = new StringBuilder("Compare Animator output against original HumanPose on the same avatar, at every used source frame.\n");
        foreach (string action in RecordedCoachSceneTools.Actions)
        {
            string path = action == "HipThrust" && !File.Exists(RecordedCoachSceneTools.Recorded(action))
                ? RingBridgeGameBuilder.ScenePath : RecordedCoachSceneTools.Recorded(action);
            var scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Additive);
            try
            {
                var all = scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<Transform>(true)).ToArray();
                var source = all.Single(t => t.name == "coach");
                var sample = UnityEngine.Object.Instantiate(source.gameObject);
                sample.hideFlags = HideFlags.HideAndDontSave;
                sample.transform.SetParent(null);
                sample.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
                sample.transform.localScale = Vector3.one;
                foreach (var script in sample.GetComponentsInChildren<MonoBehaviour>(true)) script.enabled = false;
                var animator = sample.GetComponent<Animator>();
                animator.enabled = true; animator.applyRootMotion = false;
                sample.SetActive(true); animator.Rebind(); animator.Update(0);
                var map = JsonUtility.FromJson<RecordedCoachSceneTools.PhaseMap>(File.ReadAllText(RecordedCoachSceneTools.Root + action + ".phases.json"));
                var frames = JsonUtility.FromJson<CoachMotionPackage>(File.ReadAllText(RecordedCoachSceneTools.Root + action + ".coachrecording")).segments[0].frames;
                var bones = Enum.GetValues(typeof(HumanBodyBones)).Cast<HumanBodyBones>().Where(b => b < HumanBodyBones.LastBone)
                    .Select(b => animator.GetBoneTransform(b)).Where(t => t != null).ToArray();
                var tips = new[] { HumanBodyBones.LeftHand, HumanBodyBones.RightHand, HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot };
                float worstRotation = 0, worstPosition = 0;
                var worstTips = new float[4];
                var trajectory = new StringBuilder("frame,hipRelativeKnee,leftHandY,rightHandY,leftFootY,rightFootY\n");
                try
                {
                    using (var handler = new HumanPoseHandler(animator.avatar, animator.transform))
                    for (int f = map.routineStart; f <= map.routineEnd; f++)
                    {
                        animator.Play(action.ToLowerInvariant() + "_routine", 0, (f - map.routineStart) / (float)(map.routineEnd - map.routineStart));
                        animator.Update(0);
                        Vector3 Point(HumanBodyBones bone) => animator.GetBoneTransform(bone).position;
                        trajectory.AppendLine(FormattableString.Invariant($"{f},{Point(HumanBodyBones.Hips).y-(Point(HumanBodyBones.LeftLowerLeg).y+Point(HumanBodyBones.RightLowerLeg).y)*.5f},{Point(HumanBodyBones.LeftHand).y},{Point(HumanBodyBones.RightHand).y},{Point(HumanBodyBones.LeftFoot).y},{Point(HumanBodyBones.RightFoot).y}"));
                        var rotations = bones.Select(t => t.rotation).ToArray();
                        var positions = bones.Select(t => t.position).ToArray();
                        var tipRotations = tips.Select(b => animator.GetBoneTransform(b).rotation).ToArray();
                        var expected = new HumanPose(); frames[f].ApplyTo(ref expected);
                        if (map.mirrorArmChannels)
                        {
                            var muscles = (float[])expected.muscles.Clone();
                            for (int m=0; m<HumanTrait.MuscleCount; m++)
                                expected.muscles[m] = muscles[RecordedCoachMotionImporter.SourceMuscleIndex(m,true)];
                        }
                        Quaternion heading = Quaternion.Euler(0, map.bodyYawDegrees, 0);
                        expected.bodyPosition = heading * expected.bodyPosition;
                        expected.bodyRotation = heading * expected.bodyRotation;
                        handler.SetHumanPose(ref expected);
                        for (int b = 0; b < bones.Length; b++)
                        {
                            worstRotation = Mathf.Max(worstRotation, Quaternion.Angle(rotations[b], bones[b].rotation));
                            worstPosition = Mathf.Max(worstPosition, Vector3.Distance(positions[b], bones[b].position));
                        }
                        for (int b = 0; b < tips.Length; b++) worstTips[b] = Mathf.Max(worstTips[b], Quaternion.Angle(tipRotations[b], animator.GetBoneTransform(tips[b]).rotation));
                    }
                }
                finally { UnityEngine.Object.DestroyImmediate(sample); }
                File.WriteAllText(RecordedCoachSceneTools.Output + action + "-verified-trajectory.csv", trajectory.ToString());
                report.AppendLine(FormattableString.Invariant($"{action}: {map.routineEnd-map.routineStart+1} frames; maximum rotation {worstRotation:F5} deg, position {worstPosition:F6} m; LH/RH/LF/RF {string.Join(", ", worstTips.Select(v => v.ToString("F5", System.Globalization.CultureInfo.InvariantCulture)))} deg."));
                var preview = all.Single(t => string.Equals(t.name, action + "Preview", StringComparison.OrdinalIgnoreCase)).GetComponentInChildren<Animator>(true);
                var camera = all.Single(t => string.Equals(t.name, action + "PreviewCamera", StringComparison.OrdinalIgnoreCase)).GetComponent<Camera>();
                foreach (int frame in new[] { map.boundaries[0], map.peaks[0], map.peaks[1], map.boundaries[4] }) Capture(preview, camera, action, map, frame);
                if (worstTips.Max() > 1 || worstPosition > .01f) throw new Exception(action + " differs from original recording: " + report);
                report.AppendLine("PASS " + action + (map.mirrorArmChannels
                    ? ": anatomically mirrored arm channels, original ankle rotations and leg timeline preserved."
                    : ": original wrist/ankle rotations preserved."));
            }
            finally { EditorSceneManager.CloseScene(scene, true); }
        }
        File.WriteAllText(RecordedCoachSceneTools.Output + "orientation-verification.txt", report.ToString());
        Debug.Log(report.ToString());
    }

    static void Capture(Animator avatar, Camera camera, string action, RecordedCoachSceneTools.PhaseMap map, int frame)
    {
        var model = UnityEngine.Object.Instantiate(avatar.gameObject);
        model.hideFlags = HideFlags.HideAndDontSave;
        model.transform.SetParent(null);
        model.transform.position = avatar.transform.position + Vector3.one * 10000;
        model.transform.rotation = avatar.transform.rotation;
        model.transform.localScale = avatar.transform.lossyScale;
        model.SetActive(true);
        foreach (var script in model.GetComponentsInChildren<MonoBehaviour>(true)) script.enabled = false;
        foreach (var renderer in model.GetComponentsInChildren<SkinnedMeshRenderer>(true)) renderer.updateWhenOffscreen = true;
        avatar = model.GetComponent<Animator>();
        avatar.cullingMode = AnimatorCullingMode.AlwaysAnimate;
        var cameraObject = UnityEngine.Object.Instantiate(camera.gameObject);
        cameraObject.hideFlags = HideFlags.HideAndDontSave;
        camera = cameraObject.GetComponent<Camera>();
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = Color.clear;
        avatar.enabled = true; avatar.Rebind(); avatar.Update(0);
        avatar.Play(action.ToLowerInvariant() + "_routine", 0, (frame-map.routineStart)/(float)(map.routineEnd-map.routineStart)); avatar.Update(0);
        var original = camera.targetTexture;
        var rt = RenderTexture.GetTemporary(1024, 576, 24, RenderTextureFormat.ARGB32);
        var previous = RenderTexture.active;
        try
        {
            camera.targetTexture = rt;
            var framing = camera.GetComponent<HipThrustPreviewFraming>() ?? camera.gameObject.AddComponent<HipThrustPreviewFraming>();
            framing.avatar = avatar; framing.previewCamera = camera; framing.framingPadding = .5f;
            typeof(HipThrustPreviewFraming).GetMethod("LateUpdate", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic).Invoke(framing, null);
            camera.Render(); RenderTexture.active = rt;
            var texture = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false);
            texture.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0); texture.Apply();
            File.WriteAllBytes(RecordedCoachSceneTools.Output + action + "-pose-" + frame + ".png", texture.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(texture);
        }
        finally
        {
            camera.targetTexture = original; RenderTexture.active = previous; RenderTexture.ReleaseTemporary(rt);
            UnityEngine.Object.DestroyImmediate(model); UnityEngine.Object.DestroyImmediate(cameraObject);
        }
    }
}
