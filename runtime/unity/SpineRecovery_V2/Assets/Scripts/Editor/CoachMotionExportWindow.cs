using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

/// <summary>
/// 导出窗口：选 FBX + 选场景里的 SegmentedCoachController（提供分段参数），
/// 导出成可下发的 CoachMotionPackage JSON。
/// 菜单：Spine Recovery / Export Coach Motion
/// </summary>
public class CoachMotionExportWindow : EditorWindow
{
    private GameObject sourceFbx;
    private SegmentedCoachController segmentSource;
    private string actionId = "dead_bug";
    private string displayName = "死虫式";
    private float fps = 30f;
    private int setCount = 4;
    private Vector2 scroll;
    private string lastResult;

    [MenuItem("Spine Recovery/Export Coach Motion")]
    public static void Open()
    {
        GetWindow<CoachMotionExportWindow>("Export Coach Motion");
    }

    private void OnGUI()
    {
        scroll = EditorGUILayout.BeginScrollView(scroll);

        EditorGUILayout.LabelField("源数据", EditorStyles.boldLabel);
        sourceFbx = (GameObject)EditorGUILayout.ObjectField(
            new GUIContent("FBX", "Assets/Movements/deadbug.fbx 这类模型资产"),
            sourceFbx, typeof(GameObject), false);

        segmentSource = (SegmentedCoachController)EditorGUILayout.ObjectField(
            new GUIContent("分段来源", "场景里已配置好 segments 的 SegmentedCoachController"),
            segmentSource, typeof(SegmentedCoachController), true);

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("输出参数", EditorStyles.boldLabel);
        actionId = EditorGUILayout.TextField(
            new GUIContent("actionId", "与手机端 payload 一致，如 dead_bug"), actionId);
        displayName = EditorGUILayout.TextField("显示名", displayName);
        fps = EditorGUILayout.Slider("采样帧率", fps, 15f, 60f);
        setCount = EditorGUILayout.IntField("默认组数", setCount);

        EditorGUILayout.Space();
        using (new EditorGUI.DisabledScope(sourceFbx == null || segmentSource == null))
        {
            if (GUILayout.Button("导出 JSON", GUILayout.Height(30)))
            {
                Export();
            }
        }

        if (!string.IsNullOrEmpty(lastResult))
        {
            EditorGUILayout.Space();
            EditorGUILayout.HelpBox(lastResult, MessageType.Info);
        }

        EditorGUILayout.EndScrollView();
    }

    private void Export()
    {
        lastResult = null;

        List<ActionSegment> segments = ReadSegments(segmentSource);
        if (segments == null || segments.Count == 0)
        {
            lastResult = "分段来源里没有 segments。";
            return;
        }

        Dictionary<string, AnimationClip> clips = LoadClips(sourceFbx);
        if (clips.Count == 0)
        {
            lastResult = "该 FBX 里没有找到 AnimationClip。";
            return;
        }

        string savePath = EditorUtility.SaveFilePanel(
            "导出教练动作包", Application.dataPath, $"{actionId}.motion", "json");
        if (string.IsNullOrEmpty(savePath))
        {
            return;
        }

        GameObject sampleTarget = Object.Instantiate(sourceFbx);
        sampleTarget.name = "__CoachMotionSampleTarget";
        var sampleAnimator = sampleTarget.GetComponent<Animator>();

        try
        {
            if (sampleAnimator == null || sampleAnimator.avatar == null ||
                !sampleAnimator.avatar.isHuman)
            {
                lastResult = "该 FBX 不是 Humanoid（需要 Animation Type = Humanoid）。";
                return;
            }

            var package = BuildPackage(segments, clips, sampleTarget, sampleAnimator);
            if (package == null)
            {
                return;
            }

            File.WriteAllText(savePath, JsonUtility.ToJson(package, true));

            long sizeKb = new FileInfo(savePath).Length / 1024;
            int frameTotal = 0;
            foreach (CoachMotionSegment s in package.segments)
            {
                frameTotal += s.frames.Length;
            }

            lastResult =
                $"已导出 {package.segments.Length} 段 / {frameTotal} 帧 / {sizeKb} KB\n{savePath}";
            Debug.Log($"[CoachMotionExporter] {lastResult}");
        }
        finally
        {
            DestroyImmediate(sampleTarget);
        }
    }

    private CoachMotionPackage BuildPackage(
        List<ActionSegment> segments,
        Dictionary<string, AnimationClip> clips,
        GameObject sampleTarget,
        Animator sampleAnimator)
    {
        var exported = new List<CoachMotionSegment>();
        var missing = new List<string>();

        for (int i = 0; i < segments.Count; i++)
        {
            ActionSegment source = segments[i];
            string stateName = source.clipStateName;

            if (string.IsNullOrEmpty(stateName) ||
                !clips.TryGetValue(stateName, out AnimationClip clip))
            {
                missing.Add(string.IsNullOrEmpty(stateName) ? $"[段{i}无状态名]" : stateName);
                continue;
            }

            EditorUtility.DisplayProgressBar(
                "导出教练动作包",
                $"采样 {stateName} ({i + 1}/{segments.Count})",
                (float)i / segments.Count);

            exported.Add(new CoachMotionSegment
            {
                label = source.label,
                sourceStateName = stateName,
                repeatCount = Mathf.Max(1, source.repeatCount),
                scoreLeniency = source.scoreLeniency,
                scoringDuration = source.scoringDuration,
                applyFormalActionScoreMapping = source.applyFormalActionScoreMapping,
                poseGuidanceRule = source.poseGuidanceRule,
                voiceClipName = source.voiceClip != null ? source.voiceClip.name : null,
                frames = CoachMotionExporter.SampleClip(clip, sampleTarget, sampleAnimator, fps)
            });
        }

        EditorUtility.ClearProgressBar();

        if (missing.Count > 0)
        {
            lastResult = $"FBX 里缺少这些片段，已跳过：{string.Join(", ", missing)}";
            Debug.LogWarning($"[CoachMotionExporter] {lastResult}");
        }

        if (exported.Count == 0)
        {
            lastResult = "没有任何分段导出成功。检查 clipStateName 是否与 FBX 片段名一致。";
            return null;
        }

        return new CoachMotionPackage
        {
            actionId = actionId,
            displayName = displayName,
            fps = fps,
            sourceClipName = sourceFbx.name,
            setCount = Mathf.Max(1, setCount),
            segments = exported.ToArray()
        };
    }

    /// <summary>segments 是私有字段，用 SerializedObject 读</summary>
    private static List<ActionSegment> ReadSegments(SegmentedCoachController controller)
    {
        var result = new List<ActionSegment>();
        var so = new SerializedObject(controller);
        SerializedProperty list = so.FindProperty("segments");
        if (list == null || !list.isArray)
        {
            return result;
        }

        for (int i = 0; i < list.arraySize; i++)
        {
            SerializedProperty element = list.GetArrayElementAtIndex(i);
            result.Add(new ActionSegment
            {
                label = element.FindPropertyRelative("label").stringValue,
                clipStateName = element.FindPropertyRelative("clipStateName").stringValue,
                repeatCount = element.FindPropertyRelative("repeatCount").intValue,
                scoreLeniency = element.FindPropertyRelative("scoreLeniency").floatValue,
                scoringDuration = element.FindPropertyRelative("scoringDuration").floatValue,
                applyFormalActionScoreMapping =
                    element.FindPropertyRelative("applyFormalActionScoreMapping").boolValue,
                poseGuidanceRule = ReadGuidanceRule(element),
                voiceClip =
                    element.FindPropertyRelative("voiceClip").objectReferenceValue as AudioClip
            });
        }

        return result;
    }

    private static SegmentPoseGuidanceRule ReadGuidanceRule(
        SerializedProperty segment)
    {
        SerializedProperty rule =
            segment.FindPropertyRelative("poseGuidanceRule");
        if (rule == null)
        {
            return SegmentPoseGuidanceRule.Auto;
        }

        return new SegmentPoseGuidanceRule
        {
            mode = (PoseGuidanceProfileMode)rule
                .FindPropertyRelative("mode").enumValueIndex,
            activeRegions = (PoseGuidanceRegionMask)rule
                .FindPropertyRelative("activeRegions").intValue,
            stabilizerRegions = (PoseGuidanceRegionMask)rule
                .FindPropertyRelative("stabilizerRegions").intValue
        };
    }

    private static Dictionary<string, AnimationClip> LoadClips(GameObject fbx)
    {
        var clips = new Dictionary<string, AnimationClip>();
        string path = AssetDatabase.GetAssetPath(fbx);
        if (string.IsNullOrEmpty(path))
        {
            return clips;
        }

        foreach (Object asset in AssetDatabase.LoadAllAssetsAtPath(path))
        {
            if (asset is AnimationClip clip && !clip.name.StartsWith("__preview__"))
            {
                clips[clip.name] = clip;
            }
        }

        return clips;
    }
}
