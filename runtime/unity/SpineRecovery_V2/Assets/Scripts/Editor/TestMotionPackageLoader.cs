using UnityEditor;
using UnityEngine;

/// <summary>
/// 测试工具：在编辑器里加载导出的 motion package JSON 文件，验证格式正确性。
/// 菜单：Spine Recovery → Test Motion Package Loader
/// </summary>
public class TestMotionPackageLoader : EditorWindow
{
    private string jsonFilePath = "";
    private CoachMotionPackage loadedPackage;
    private Vector2 scrollPosition;

    [MenuItem("Spine Recovery/Test Motion Package Loader")]
    private static void ShowWindow()
    {
        var window = GetWindow<TestMotionPackageLoader>("Test Motion Package Loader");
        window.Show();
    }

    private void OnGUI()
    {
        GUILayout.Label("测试加载导出的 Motion Package JSON", EditorStyles.boldLabel);
        GUILayout.Space(10);

        // 文件选择
        GUILayout.BeginHorizontal();
        GUILayout.Label("JSON 文件路径:", GUILayout.Width(120));
        jsonFilePath = GUILayout.TextField(jsonFilePath);
        if (GUILayout.Button("浏览...", GUILayout.Width(80)))
        {
            string path = EditorUtility.OpenFilePanel("选择 Motion Package JSON", Application.dataPath, "json");
            if (!string.IsNullOrEmpty(path))
            {
                jsonFilePath = path;
            }
        }
        GUILayout.EndHorizontal();

        GUILayout.Space(10);

        // 加载按钮
        if (GUILayout.Button("加载并验证", GUILayout.Height(30)))
        {
            LoadAndValidate();
        }

        GUILayout.Space(10);

        // 显示结果
        if (loadedPackage != null)
        {
            scrollPosition = GUILayout.BeginScrollView(scrollPosition);

            GUILayout.Label("✓ 加载成功", EditorStyles.boldLabel);
            GUILayout.Space(5);

            GUILayout.Label($"动作 ID: {loadedPackage.actionId}");
            GUILayout.Label($"显示名: {loadedPackage.displayName}");
            GUILayout.Label($"帧率: {loadedPackage.fps}");
            GUILayout.Label($"默认组数: {loadedPackage.setCount}");
            GUILayout.Label($"分段数: {loadedPackage.segments.Length}");

            GUILayout.Space(10);
            GUILayout.Label("分段详情:", EditorStyles.boldLabel);

            int totalFrames = 0;
            foreach (var segment in loadedPackage.segments)
            {
                GUILayout.BeginVertical(EditorStyles.helpBox);
                GUILayout.Label($"• {segment.label} ({segment.sourceStateName})");
                GUILayout.Label($"  帧数: {segment.frames.Length}, 重复: {segment.repeatCount}, 评分时长: {segment.scoringDuration}s");
                totalFrames += segment.frames.Length;
                GUILayout.EndVertical();
            }

            GUILayout.Space(10);
            GUILayout.Label($"总帧数: {totalFrames}");
            float estimatedDuration = totalFrames / loadedPackage.fps;
            GUILayout.Label($"预估时长: {estimatedDuration:F2} 秒");

            float estimatedSizeKB = (totalFrames * 95 * 4 + totalFrames * 7 * 4) / 1024f;
            GUILayout.Label($"预估大小: {estimatedSizeKB:F1} KB (未压缩)");

            GUILayout.EndScrollView();
        }
    }

    private void LoadAndValidate()
    {
        loadedPackage = null;

        if (string.IsNullOrEmpty(jsonFilePath))
        {
            EditorUtility.DisplayDialog("错误", "请先选择 JSON 文件", "确定");
            return;
        }

        if (!System.IO.File.Exists(jsonFilePath))
        {
            EditorUtility.DisplayDialog("错误", $"文件不存在:\n{jsonFilePath}", "确定");
            return;
        }

        try
        {
            string jsonText = System.IO.File.ReadAllText(jsonFilePath);
            loadedPackage = JsonUtility.FromJson<CoachMotionPackage>(jsonText);

            if (loadedPackage == null)
            {
                EditorUtility.DisplayDialog("错误", "JSON 反序列化失败，可能格式不正确", "确定");
                return;
            }

            if (!loadedPackage.IsValid(out string error))
            {
                EditorUtility.DisplayDialog("验证失败", $"动作包数据不完整:\n{error}", "确定");
                loadedPackage = null;
                return;
            }

            EditorUtility.DisplayDialog("成功", "动作包加载并验证成功！", "确定");
        }
        catch (System.Exception ex)
        {
            EditorUtility.DisplayDialog("错误", $"加载失败:\n{ex.Message}", "确定");
            loadedPackage = null;
        }
    }
}
