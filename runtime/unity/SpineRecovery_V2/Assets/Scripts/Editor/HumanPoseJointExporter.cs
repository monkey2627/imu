using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using UnityEditor;
using UnityEngine;

public static class HumanPoseJointExporter
{
    private static readonly string[] BoneNames =
    {
        "pelvis", "femur_l", "femur_r", "tibia_l", "tibia_r", "talus_l", "talus_r",
        "toes_l", "toes_r", "lumbar_body", "thorax", "head", "humerus_l", "humerus_r",
        "ulna_l", "ulna_r", "hand_l", "hand_r"
    };

    private static readonly HumanBodyBones[] HumanBones =
    {
        HumanBodyBones.Hips, HumanBodyBones.LeftUpperLeg, HumanBodyBones.RightUpperLeg,
        HumanBodyBones.LeftLowerLeg, HumanBodyBones.RightLowerLeg, HumanBodyBones.LeftFoot,
        HumanBodyBones.RightFoot, HumanBodyBones.LeftToes, HumanBodyBones.RightToes,
        HumanBodyBones.Spine, HumanBodyBones.Chest, HumanBodyBones.Head,
        HumanBodyBones.LeftUpperArm, HumanBodyBones.RightUpperArm, HumanBodyBones.LeftLowerArm,
        HumanBodyBones.RightLowerArm, HumanBodyBones.LeftHand, HumanBodyBones.RightHand
    };

    [Serializable] private sealed class MotionPackage
    {
        public float fps;
        public CoachMotionSegment[] segments;
    }

    [Serializable] private sealed class OutputPackage
    {
        public string coordinateSystem = "unity_world_xyz";
        public string sourceMotionJson;
        public string sourceMotionSha256;
        public string avatarAssetPath;
        public float fps;
        public int frameCount;
        public string[] bones;
        public OutputPose restPose;
        public OutputFrame[] frames;
    }

    [Serializable] private sealed class OutputPose
    {
        public bool[] present;
        public Vector3[] positions;
        public Quaternion[] rotations;
    }

    [Serializable] private sealed class OutputFrame
    {
        public float time;
        public bool[] present;
        public Vector3[] positions;
        public Quaternion[] rotations;
    }

    [MenuItem("Tools/Motion Recovery/Export Deadbug HumanPose Joint Reference")]
    private static void ExportDeadbugFromProjectMenu()
    {
        string projectRoot = Directory.GetParent(Application.dataPath).FullName;
        string workspaceRoot = FindWorkspaceRoot(projectRoot);
        string dataRoot = Directory.GetParent(Directory.GetParent(workspaceRoot).FullName).FullName;
        string outputPath = Path.Combine(workspaceRoot, "runtime", "exports", "unity_humanpose",
            "deadbug_humanpose_joints.json");
        string actionRoot = Path.Combine(dataRoot, "data", "raw", "ours");
        string actionDirectory = Directory.GetDirectories(actionRoot, "7_*").FirstOrDefault();
        if (actionDirectory == null)
            throw new DirectoryNotFoundException("Cannot find action folder 7_* under " + actionRoot);
        string motionPath = Path.Combine(actionDirectory, "1", "7_v1.motion.json");
        try
        {
            if (!File.Exists(motionPath))
                throw new FileNotFoundException("Deadbug motion package was not found.", motionPath);
            Export(motionPath, outputPath);
            EditorUtility.DisplayDialog("HumanPose export complete",
                $"Exported {outputPath}", "OK");
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorUtility.DisplayDialog("HumanPose export failed", exception.Message, "OK");
        }
    }

    [MenuItem("Tools/Motion Recovery/Export All Ours HumanPose Joint References")]
    private static void ExportAllOursFromProjectMenu()
    {
        string projectRoot = Directory.GetParent(Application.dataPath).FullName;
        string workspaceRoot = FindWorkspaceRoot(projectRoot);
        string dataRoot = Directory.GetParent(Directory.GetParent(workspaceRoot).FullName).FullName;
        string actionRoot = Path.Combine(dataRoot, "data", "raw", "ours");
        string outputRoot = Path.Combine(workspaceRoot, "runtime", "exports", "unity_humanpose", "humanpose_ours");
        try
        {
            if (!Directory.Exists(actionRoot))
                throw new DirectoryNotFoundException("Cannot find Ours action root: " + actionRoot);

            int exported = 0;
            foreach (string actionDirectory in Directory.GetDirectories(actionRoot))
            {
                string actionId = Path.GetFileName(actionDirectory).Split('_')[0];
                if (!int.TryParse(actionId, out _))
                    continue;
                string motionPath = Directory.GetFiles(actionDirectory, "*_v*.motion.json", SearchOption.AllDirectories).FirstOrDefault();
                if (string.IsNullOrEmpty(motionPath))
                    continue;
                string outputPath = Path.Combine(outputRoot, actionId + ".humanpose_joints.json");
                Export(motionPath, outputPath);
                exported++;
            }

            AssetDatabase.Refresh();
            EditorUtility.DisplayDialog("HumanPose export complete",
                $"Exported {exported} Ours action references to {outputRoot}", "OK");
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorUtility.DisplayDialog("HumanPose export failed", exception.Message, "OK");
        }
    }

    public static void ExportFromCommandLine()
    {
        string motionPath = Argument("-motionJson");
        string outputPath = Argument("-outJson");
        if (string.IsNullOrWhiteSpace(motionPath) || string.IsNullOrWhiteSpace(outputPath))
            throw new ArgumentException("Pass -motionJson <path> and -outJson <path>.");

        Export(motionPath, outputPath);
    }

    private static void Export(string motionPath, string outputPath)
    {

        MotionPackage package = JsonUtility.FromJson<MotionPackage>(File.ReadAllText(motionPath));
        CoachMotionFrame[] frames = package?.segments?.FirstOrDefault()?.frames;
        if (frames == null || frames.Length == 0)
            throw new InvalidDataException("Motion package contains no frames.");
        if (frames.Any(f => f.muscles == null || f.muscles.Length != HumanTrait.MuscleCount))
            throw new InvalidDataException("A frame has an invalid HumanPose muscle count.");

        const string avatarPath = "Assets/Models/mesh_edit1.vrm";
        GameObject avatarAsset = AssetDatabase.LoadAssetAtPath<GameObject>(avatarPath);
        if (avatarAsset == null) throw new FileNotFoundException("Cannot load Humanoid avatar", avatarPath);
        GameObject instance = PrefabUtility.InstantiatePrefab(avatarAsset) as GameObject;
        if (instance == null) instance = UnityEngine.Object.Instantiate(avatarAsset);

        try
        {
            Animator animator = instance.GetComponentInChildren<Animator>(true);
            if (animator == null || animator.avatar == null || !animator.avatar.isHuman)
                throw new InvalidDataException("mesh_edit1.vrm has no valid Humanoid Animator.");
            Transform[] transforms = HumanBones.Select(animator.GetBoneTransform).ToArray();
            if (transforms.Any(t => t == null))
                throw new InvalidDataException("The avatar is missing required HumanBodyBones.");

            var output = new OutputPackage
            {
                sourceMotionJson = Path.GetFullPath(motionPath),
                sourceMotionSha256 = Sha256(motionPath),
                avatarAssetPath = avatarPath,
                fps = package.fps,
                frameCount = frames.Length,
                bones = BoneNames,
                restPose = CapturePose(transforms),
                frames = new OutputFrame[frames.Length]
            };
            using (var handler = new HumanPoseHandler(animator.avatar, animator.transform))
            {
                var pose = new HumanPose { muscles = new float[HumanTrait.MuscleCount] };
                for (int i = 0; i < frames.Length; i++)
                {
                    frames[i].ApplyTo(ref pose);
                    handler.SetHumanPose(ref pose);
                    output.frames[i] = new OutputFrame
                    {
                        time = frames[i].time,
                        present = Enumerable.Repeat(true, transforms.Length).ToArray(),
                        positions = transforms.Select(t => t.position).ToArray(),
                        rotations = transforms.Select(t => t.rotation).ToArray()
                    };
                }
            }

            string directory = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
            File.WriteAllText(outputPath, JsonUtility.ToJson(output));
            Debug.Log($"Exported {frames.Length} HumanPose frames to {outputPath}");
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(instance);
        }
    }

    private static OutputPose CapturePose(Transform[] transforms)
    {
        return new OutputPose
        {
            present = Enumerable.Repeat(true, transforms.Length).ToArray(),
            positions = transforms.Select(t => t.position).ToArray(),
            rotations = transforms.Select(t => t.rotation).ToArray()
        };
    }

    private static string Sha256(string path)
    {
        using (var stream = File.OpenRead(path))
        using (var sha = SHA256.Create())
        {
            return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }
    }

    private static string FindWorkspaceRoot(string projectRoot)
    {
        DirectoryInfo directory = new DirectoryInfo(projectRoot);
        while (directory != null)
        {
            if (Directory.Exists(Path.Combine(directory.FullName, "runtime")) &&
                Directory.Exists(Path.Combine(directory.FullName, "local_models")))
                return directory.FullName;
            directory = directory.Parent;
        }
        throw new DirectoryNotFoundException("Cannot locate MotionRecoverWorkspace above " + projectRoot);
    }

    private static string Argument(string name)
    {
        string[] args = Environment.GetCommandLineArgs();
        int index = Array.IndexOf(args, name);
        return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
    }
}
