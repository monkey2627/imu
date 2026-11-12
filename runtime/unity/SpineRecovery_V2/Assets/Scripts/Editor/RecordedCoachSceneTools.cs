using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Globalization;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class RecordedCoachSceneTools
{
    public const string Root = "Assets/Movements/Recorded20261018/";
    public const string Output = "output/recorded-coach/";
    public static readonly string[] Actions = { "BirdDog", "DeadBug", "HipThrust" };

    [Serializable] public sealed class PhaseMap
    {
        public string action;
        public string source;
        public float fps = 30;
        public float rotationOffset;
        public float bodyYawDegrees;
        public bool alignBodyOrientation;
        public bool mirrorArmChannels;
        public int routineStart, routineEnd;
        public int[] boundaries;
        public int[] peaks;
        public int[] firstTargets, secondTargets;
        public HipThrustTaikoController.HipBeat[] beats;
        public int repeatStart, repeatEnd;
    }

    public static string Original(string action) => "Assets/Scenes/" + (action == "HipThrust" ? action : action + "_RingSkinC") + ".unity";
    public static string Recorded(string action) => "Assets/Scenes/" + action + "_RingSkinC_Recorded.unity";

    public static System.Collections.Generic.Dictionary<string, AnimationClip> LoadClips(string action) =>
        AssetDatabase.LoadAllAssetsAtPath(Root + action + ".coachrecording").OfType<AnimationClip>().ToDictionary(c => c.name);

    [MenuItem("Tools/Recorded Coach/Connect Original Pose Clips")]
    public static void ConnectOriginalPoseClips()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) throw new InvalidOperationException("Exit Play mode first.");
        foreach (string action in Actions)
        {
            AssetDatabase.ImportAsset(Root + action + ".coachrecording", ImportAssetOptions.ForceSynchronousImport);
            var clips = LoadClips(action);
            if (clips.Count != 11 || clips.Values.Any(c => !c.humanMotion)) throw new Exception("Invalid original pose clips");
            var controller = AssetDatabase.LoadAssetAtPath<AnimatorController>(Root + action + "_Recorded.controller");
            string prefix = action.ToLowerInvariant();
            foreach (var child in controller.layers[0].stateMachine.states)
                child.state.motion = clips.TryGetValue(child.state.name, out var clip) ? clip : clips[prefix + (child.state.name == "Idle" ? "_start" : "_routine")];
            EditorUtility.SetDirty(controller);
            if (action == "HipThrust")
            {
                var scene = EditorSceneManager.OpenScene(Recorded(action), OpenSceneMode.Additive);
                try
                {
                    var taiko = scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<HipThrustTaikoController>(true)).Single();
                    var data = new SerializedObject(taiko);
                    data.FindProperty("sourceClip").objectReferenceValue = clips[prefix + "_routine"];
                    data.ApplyModifiedPropertiesWithoutUndo();
                    EditorSceneManager.SaveScene(scene);
                }
                finally { EditorSceneManager.CloseScene(scene, true); }
            }
        }
        AssetDatabase.SaveAssets();
    }

    [MenuItem("Tools/Recorded Coach/Validate Imported Phases")]
    public static void ValidatePhases()
    {
        Directory.CreateDirectory(Output);
        var report = new StringBuilder();
        var previous = SceneManager.GetActiveScene();
        foreach (string action in Actions)
        {
            var scene = EditorSceneManager.OpenScene(Recorded(action), OpenSceneMode.Additive);
            try
            {
                var all = scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<Transform>(true)).ToArray();
                if (all.Any(t => GameObjectUtility.GetMonoBehavioursWithMissingScriptCount(t.gameObject) > 0)) throw new Exception("Missing script in " + action);
                var source = all.Single(t => t.name == "coach").gameObject;
                var sample = UnityEngine.Object.Instantiate(source);
                sample.hideFlags = HideFlags.HideAndDontSave;
                foreach (var script in sample.GetComponentsInChildren<MonoBehaviour>(true)) script.enabled = false;
                var animator = sample.GetComponent<Animator>(); animator.enabled = true;
                sample.SetActive(true); animator.Rebind(); animator.Update(0);
                sample.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
                sample.transform.localScale = Vector3.one;
                var clips = LoadClips(action);
                if (clips.Count != 11 || clips.Values.Any(c => !c.humanMotion || c.length <= 0)) throw new Exception("Invalid pose clips: " + action);
                try
                {
                    string prefix = action.ToLowerInvariant();
                    string[] ordered = new[] { prefix + "_start" }.Concat(Enumerable.Range(1,4).SelectMany(i => new[] { prefix + "_" + i + "_up", prefix + "_" + i + "_down" })).Concat(new[] { prefix + "_end" }).ToArray();
                    var bones = Enum.GetValues(typeof(HumanBodyBones)).Cast<HumanBodyBones>().Where(b => b < HumanBodyBones.LastBone).Select(animator.GetBoneTransform).Where(t => t != null).ToArray();
                    float worst=0;
                    for(int i=0;i<ordered.Length-1;i++)
                    {
                        if (!animator.HasState(0, Animator.StringToHash(ordered[i])) || !clips.ContainsKey(ordered[i+1])) throw new Exception("Missing phase state");
                        animator.Play(ordered[i],0,1); animator.Update(0); var end=bones.Select(t=>t.position).ToArray();
                        animator.Play(ordered[i+1],0,0); animator.Update(0);
                        for(int n=0;n<bones.Length;n++) worst=Mathf.Max(worst,Vector3.Distance(end[n],bones[n].position));
                    }
                    if (worst > .01f) throw new Exception(action + " phase seam exceeds 1 cm: " + worst);
                    var map=JsonUtility.FromJson<PhaseMap>(File.ReadAllText(Root + action + ".phases.json"));
                    if (action == "HipThrust")
                    {
                        for(int pass=0;pass<3;pass++) foreach(var beat in map.beats)
                        {
                            if (!(beat.upFrame < beat.holdEndFrame && beat.holdEndFrame < beat.downFrame && beat.downFrame < map.repeatEnd)) throw new Exception("Invalid hip beat order");
                            int absolute=beat.upFrame+pass*(map.repeatEnd-map.repeatStart);
                            int mapped=absolute<map.repeatStart?absolute:map.repeatStart+(absolute-map.repeatStart)%(map.repeatEnd-map.repeatStart);
                            if(mapped!=beat.upFrame) throw new Exception("Hip repeat mapping failed");
                        }
                    }
                    report.AppendLine(FormattableString.Invariant($"PASS {action}: 11 humanoid clips, 10 ordered phases, maximum adjacent seam {worst:F6} m; source FPS {map.fps}."));
                }
                finally { UnityEngine.Object.DestroyImmediate(sample); }
            }
            finally { if(previous.IsValid() && previous.isLoaded) SceneManager.SetActiveScene(previous); EditorSceneManager.CloseScene(scene,true); }
        }
        File.WriteAllText(Output + "phase-verification.txt",report.ToString());
        Debug.Log(report.ToString());
    }

    [MenuItem("Tools/Recorded Coach/Create Three Recorded Scenes")]
    public static void Build()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) throw new InvalidOperationException("Exit Play mode first.");
        if (Actions.Any(a => File.Exists(Recorded(a)))) throw new IOException("Recorded scenes already exist; refusing to replace them.");
        var previous = SceneManager.GetActiveScene();
        foreach (string action in Actions)
        {
            var map = JsonUtility.FromJson<PhaseMap>(File.ReadAllText(Root + action + ".phases.json"));
            string prefix = action.ToLowerInvariant();
            var importer = (ModelImporter)AssetImporter.GetAtPath(Root + action + ".fbx");
            string take = importer.defaultClipAnimations[0].takeName;
            ModelImporterClipAnimation Slice(string name, int first, int last)
            {
                if (first < 0 || last <= first || last > importer.defaultClipAnimations[0].lastFrame) throw new Exception("Invalid phase: " + name);
                return new ModelImporterClipAnimation {
                    name = name, takeName = take, firstFrame = first, lastFrame = last,
                    lockRootRotation = !map.alignBodyOrientation, lockRootHeightY = true, lockRootPositionXZ = true,
                    keepOriginalOrientation = !map.alignBodyOrientation, keepOriginalPositionY = true, keepOriginalPositionXZ = true,
                    rotationOffset = map.rotationOffset,
                    loopTime = false
                };
            }
            var slices = new System.Collections.Generic.List<ModelImporterClipAnimation> {
                Slice(prefix + "_routine", map.routineStart, map.routineEnd),
                Slice(prefix + "_start", map.routineStart, map.boundaries[0]),
                Slice(prefix + "_end", map.boundaries[4], map.routineEnd)
            };
            for (int i=0; i<4; i++)
            {
                slices.Add(Slice(prefix + "_" + (i+1) + "_up", map.boundaries[i], map.peaks[i]));
                slices.Add(Slice(prefix + "_" + (i+1) + "_down", map.peaks[i], map.boundaries[i+1]));
            }
            importer.clipAnimations = slices.ToArray();
            importer.SaveAndReimport();
            var clips = LoadClips(action);
            if (clips.Count != 11 || clips.Values.Any(c => !c.humanMotion || c.length <= 0)) throw new Exception("Invalid imported clips: " + action);
            string controllerPath = Root + action + "_Recorded.controller";
            var controller = AnimatorController.CreateAnimatorControllerAtPath(controllerPath);
            var machine = controller.layers[0].stateMachine;
            int stateIndex = 0;
            foreach (var clip in clips.Values)
            {
                var state = machine.AddState(clip.name, new Vector3((stateIndex%3)*260, (stateIndex/3)*80, 0));
                state.motion = clip; stateIndex++;
                if (clip.name == prefix + "_routine") machine.defaultState = state;
            }
            foreach (string alias in new[] { "animation_pose", "Deadbug1", "Idle" })
                machine.AddState(alias).motion = clips[prefix + (alias == "Idle" ? "_start" : "_routine")];
            EditorUtility.SetDirty(controller);
            AssetDatabase.SaveAssets();
            if (!AssetDatabase.CopyAsset(Original(action), Recorded(action))) throw new IOException("Cannot copy scene: " + action);
            var scene = EditorSceneManager.OpenScene(Recorded(action), OpenSceneMode.Additive);
            SceneManager.SetActiveScene(scene);
            try
            {
                T Find<T>() where T : Component => scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<T>(true)).Single();
                var coach = Find<CoachActionController>();
                var card = Find<ActionPreviewCard>();
                var coachAnimator = (Animator)new SerializedObject(coach).FindProperty("coachAnimator").objectReferenceValue;
                var previewAnimator = (Animator)new SerializedObject(card).FindProperty("previewAnimator").objectReferenceValue;
                var transforms = scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<Transform>(true)).ToArray();
                if (coachAnimator == null) coachAnimator = transforms.Single(t => t.name == "coach").GetComponent<Animator>();
                if (previewAnimator == null) previewAnimator = transforms.Single(t => string.Equals(t.name, action + "Preview", StringComparison.OrdinalIgnoreCase)).GetComponentInChildren<Animator>(true);
                var coachProperties = new SerializedObject(coach);
                coachProperties.FindProperty("coachAnimator").objectReferenceValue = coachAnimator;
                coachProperties.ApplyModifiedPropertiesWithoutUndo();
                coachAnimator.runtimeAnimatorController = controller;
                previewAnimator.runtimeAnimatorController = controller;
                PrefabUtility.RecordPrefabInstancePropertyModifications(coachAnimator);
                PrefabUtility.RecordPrefabInstancePropertyModifications(previewAnimator);
                var cardProperties = new SerializedObject(card);
                cardProperties.FindProperty("previewAnimator").objectReferenceValue = previewAnimator;
                cardProperties.FindProperty("targetSceneName").stringValue = Path.GetFileNameWithoutExtension(Recorded(action));
                cardProperties.ApplyModifiedPropertiesWithoutUndo();
                Find<GameWorkoutScene>().useRecordedCoachAnimations = true;
                if (action == "HipThrust")
                {
                    Find<HipThrustPreviewFraming>().framingPadding = .5f;
                    var properties = new SerializedObject(Find<HipThrustTaikoController>());
                    properties.FindProperty("sourceClip").objectReferenceValue = clips[prefix + "_routine"];
                    properties.FindProperty("hipThrustAnimatorController").objectReferenceValue = controller;
                    properties.FindProperty("repeatStartFrame").intValue = map.repeatStart;
                    properties.FindProperty("repeatEndFrame").intValue = map.repeatEnd;
                    var beats = properties.FindProperty("sourceBeats"); beats.arraySize = 4;
                    for (int i=0; i<4; i++)
                    {
                        var beat = beats.GetArrayElementAtIndex(i);
                        beat.FindPropertyRelative("upFrame").intValue = map.beats[i].upFrame;
                        beat.FindPropertyRelative("holdEndFrame").intValue = map.beats[i].holdEndFrame;
                        beat.FindPropertyRelative("downFrame").intValue = map.beats[i].downFrame;
                    }
                    properties.ApplyModifiedPropertiesWithoutUndo();
                }
                else
                {
                    var properties = new SerializedObject(Find<RhythmModeController>());
                    var targets = properties.FindProperty("recordedSegmentTargets"); targets.arraySize = 4;
                    for (int i=0; i<4; i++)
                    {
                        var target = targets.GetArrayElementAtIndex(i);
                        target.FindPropertyRelative("stateName").stringValue = prefix + "_" + (i+1) + "_up";
                        target.FindPropertyRelative("first").enumValueIndex = map.firstTargets[i];
                        target.FindPropertyRelative("second").enumValueIndex = map.secondTargets[i];
                    }
                    properties.ApplyModifiedPropertiesWithoutUndo();
                }
                EditorSceneManager.MarkSceneDirty(scene);
                EditorSceneManager.SaveScene(scene);
            }
            finally
            {
                if (previous.IsValid() && previous.isLoaded) SceneManager.SetActiveScene(previous);
                EditorSceneManager.CloseScene(scene, true);
            }
        }
        EditorBuildSettings.scenes = Actions.Select(a => new EditorBuildSettingsScene(Original(a), true))
            .Concat(Actions.Select(a => new EditorBuildSettingsScene(Recorded(a), true))).ToArray();
        AssetDatabase.SaveAssets();
        File.WriteAllText(Output + "build-result.txt", "Created three recorded-coach scenes. Original scene bytes preserved.\n");
    }

    [MenuItem("Tools/Recorded Coach/Analyze Source Animations")]
    public static void Analyze()
    {
        if (Actions.Any(a => File.Exists(Recorded(a)))) throw new InvalidOperationException("Analyze before creating recorded scenes; their phase imports must be preserved.");
        Directory.CreateDirectory(Output);
        var report = new StringBuilder();
        foreach (string action in Actions)
        {
            string path = Root + action + ".fbx";
            var importer = (ModelImporter)AssetImporter.GetAtPath(path);
            importer.animationType = ModelImporterAnimationType.Human;
            importer.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel;
            importer.animationCompression = ModelImporterAnimationCompression.Off;
            importer.importAnimation = true;
            importer.SaveAndReimport();
            // These exports use the OpenSim skeleton. Unity's guess selects radius_l
            // as the left hand and omits thorax; use the actual anatomical joints.
            var description = importer.humanDescription;
            string[] human = { "Hips", "Spine", "Chest", "Head", "LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm", "LeftLowerArm", "RightLowerArm", "LeftHand", "RightHand", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg", "LeftFoot", "RightFoot", "LeftToes", "RightToes" };
            string[] bones = { "pelvis", "lumbar_body", "thorax", "head", "scapula_l", "scapula_r", "humerus_l", "humerus_r", "ulna_l", "ulna_r", "hand_l", "hand_r", "femur_l", "femur_r", "tibia_l", "tibia_r", "calcn_l", "calcn_r", "toes_l", "toes_r" };
            description.human = human.Select((name, i) => new HumanBone { humanName = name, boneName = bones[i], limit = new HumanLimit { useDefaultValues = true } }).ToArray();
            importer.humanDescription = description;
            var clips = importer.defaultClipAnimations;
            report.AppendLine(action + " takes: " + string.Join(", ", clips.Select(c => c.takeName + " " + c.firstFrame + "-" + c.lastFrame)));
            foreach (var c in clips)
            {
                c.name = action.ToLowerInvariant() + "_routine";
                c.lockRootRotation = true;
                c.lockRootHeightY = true;
                c.lockRootPositionXZ = true;
                c.keepOriginalOrientation = true;
                c.keepOriginalPositionY = true;
                c.keepOriginalPositionXZ = true;
                c.loopTime = false;
            }
            importer.clipAnimations = clips;
            importer.SaveAndReimport();
            var model = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            var instance = UnityEngine.Object.Instantiate(model);
            instance.hideFlags = HideFlags.HideAndDontSave;
            try
            {
                var animator = instance.GetComponent<Animator>();
                report.AppendLine("Avatar valid=" + animator.avatar.isValid + " human=" + animator.avatar.isHuman);
                if (!animator.avatar.isValid || !animator.avatar.isHuman) throw new Exception("Invalid humanoid avatar: " + action);
                var clip = AssetDatabase.LoadAllAssetsAtPath(path).OfType<AnimationClip>().First(c => !c.name.StartsWith("__"));
                report.AppendLine("Clip " + clip.name + " length=" + clip.length + " fps=" + clip.frameRate);
                var csv = new StringBuilder("frame,time,hipY,hipRelativeKnee,leftHandY,rightHandY,leftFootY,rightFootY,leftHandZ,rightHandZ,leftFootZ,rightFootZ\n");
                for (int f = 0; f <= Mathf.RoundToInt(clip.length * clip.frameRate); f++)
                {
                    clip.SampleAnimation(instance, f / clip.frameRate);
                    Vector3 Bone(HumanBodyBones b) => animator.GetBoneTransform(b).position;
                    var h = Bone(HumanBodyBones.Hips);
                    var lh = Bone(HumanBodyBones.LeftHand); var rh = Bone(HumanBodyBones.RightHand);
                    var lf = Bone(HumanBodyBones.LeftFoot); var rf = Bone(HumanBodyBones.RightFoot);
                    float knee = (Bone(HumanBodyBones.LeftLowerLeg).y + Bone(HumanBodyBones.RightLowerLeg).y) * .5f;
                    float[] row = { f, f / clip.frameRate, h.y, h.y-knee, lh.y, rh.y, lf.y, rf.y, lh.z, rh.z, lf.z, rf.z };
                    csv.AppendLine(string.Join(",", row.Select(v => v.ToString("R", CultureInfo.InvariantCulture))));
                }
                File.WriteAllText(Output + action + ".csv", csv.ToString());
            }
            finally { UnityEngine.Object.DestroyImmediate(instance); }
        }
        File.WriteAllText(Output + "source-analysis.txt", report.ToString());
        Debug.Log("RECORDED_COACH_ANALYZED\n" + report);
    }
}
