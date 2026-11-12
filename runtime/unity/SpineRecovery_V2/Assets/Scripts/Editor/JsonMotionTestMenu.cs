using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Unity 编辑器菜单，用于测试 JSON 驱动的练习和游戏场景
/// </summary>
public static class JsonMotionTestMenu
{
    private const string DeadBugJsonPath = @"E:\ARp\dead_bug.motion.json";
    private const string BirdDogJsonPath = @"E:\ARp\bird_dog.motion.json";

    [MenuItem("Spine Recovery/Test JSON/Browse and Load JSON for testPractice...")]
    public static void BrowseAndLoadJsonForPractice()
    {
        string jsonPath = EditorUtility.OpenFilePanel(
            "选择 Motion Package JSON 文件",
            "E:\\ARp",
            "json");

        if (string.IsNullOrEmpty(jsonPath))
        {
            Debug.Log("[JsonMotionTestMenu] 用户取消了文件选择");
            return;
        }

        if (!System.IO.File.Exists(jsonPath))
        {
            EditorUtility.DisplayDialog(
                "文件不存在",
                $"找不到文件: {jsonPath}",
                "确定");
            return;
        }

        // 验证 JSON 文件
        try
        {
            string jsonText = System.IO.File.ReadAllText(jsonPath);
            CoachMotionPackage package = JsonUtility.FromJson<CoachMotionPackage>(jsonText);
            if (package == null)
            {
                EditorUtility.DisplayDialog(
                    "JSON 格式错误",
                    $"无法解析 JSON 文件: {jsonPath}\n\n请确保文件是有效的 CoachMotionPackage 格式。",
                    "确定");
                return;
            }

            if (!package.IsValid(out string error))
            {
                EditorUtility.DisplayDialog(
                    "Motion Package 无效",
                    $"JSON 文件格式正确，但内容无效:\n\n{error}",
                    "确定");
                return;
            }

            Debug.Log($"[JsonMotionTestMenu] JSON 验证成功: actionId='{package.actionId}', segments={package.segments.Length}");
        }
        catch (System.Exception ex)
        {
            EditorUtility.DisplayDialog(
                "JSON 读取失败",
                $"读取或解析 JSON 文件时出错:\n\n{ex.Message}",
                "确定");
            return;
        }

        // 设置 URI（现在会自动保存到 EditorPrefs）
        PracticeSceneInitializer.PendingMotionPackageUri = jsonPath;
        Debug.Log($"[JsonMotionTestMenu] 已设置 PendingMotionPackageUri = {jsonPath}");

        // 加载场景
        EditorSceneManager.OpenScene("Assets/Scenes/testPractice.unity");
        Debug.Log($"[JsonMotionTestMenu] testPractice 场景已加载，将使用 JSON: {jsonPath}");
    }

    [MenuItem("Spine Recovery/Test JSON/Load testPractice (DeadBug JSON)")]
    public static void TestPracticeWithDeadBugJson()
    {
        if (!System.IO.File.Exists(DeadBugJsonPath))
        {
            EditorUtility.DisplayDialog(
                "JSON 文件不存在",
                $"找不到文件: {DeadBugJsonPath}\n\n请使用 'Browse and Load JSON' 选择其他文件。",
                "确定");
            return;
        }

        PracticeSceneInitializer.PendingMotionPackageUri = DeadBugJsonPath;
        EditorSceneManager.OpenScene("Assets/Scenes/testPractice.unity");
        Debug.Log($"[JsonMotionTestMenu] 已设置 PendingMotionPackageUri = {DeadBugJsonPath}，即将加载 testPractice 场景");
    }

    [MenuItem("Spine Recovery/Test JSON/Load testPractice (BirdDog JSON)")]
    public static void TestPracticeWithBirdDogJson()
    {
        if (!System.IO.File.Exists(BirdDogJsonPath))
        {
            EditorUtility.DisplayDialog(
                "JSON 文件不存在",
                $"找不到文件: {BirdDogJsonPath}\n\n请使用 'Browse and Load JSON' 选择其他文件。",
                "确定");
            return;
        }

        PracticeSceneInitializer.PendingMotionPackageUri = BirdDogJsonPath;
        EditorSceneManager.OpenScene("Assets/Scenes/testPractice.unity");
        Debug.Log($"[JsonMotionTestMenu] 已设置 PendingMotionPackageUri = {BirdDogJsonPath}，即将加载 testPractice 场景");
    }

    [MenuItem("Spine Recovery/Test JSON/Load testGaming (需要先运行 testPractice)")]
    public static void TestGaming()
    {
        if (SpineFlowTrainingSession.CachedMotionPackage == null)
        {
            bool proceed = EditorUtility.DisplayDialog(
                "无缓存的 Motion Package",
                "testGaming 需要先在 testPractice 中缓存 Motion Package。\n\n" +
                "你可以：\n" +
                "1. 先运行 'Load testPractice (DeadBug JSON)' 或 'Load testPractice (BirdDog JSON)'\n" +
                "2. 或者点击「仍然加载」在编辑器中测试（跳过校准）",
                "仍然加载",
                "取消");

            if (!proceed)
            {
                return;
            }
        }

        EditorSceneManager.OpenScene("Assets/Scenes/testGaming.unity");
        Debug.Log("[JsonMotionTestMenu] 已加载 testGaming 场景" +
            (SpineFlowTrainingSession.CachedMotionPackage != null
                ? "，使用缓存的 Motion Package"
                : "，将使用传统模式"));
    }

    [MenuItem("Spine Recovery/Test JSON/Clear Cached Package")]
    public static void ClearCachedPackage()
    {
        SpineFlowTrainingSession.CacheMotionPackage(null);
        Debug.Log("[JsonMotionTestMenu] 已清除缓存的 Motion Package");
    }

    [MenuItem("Spine Recovery/Test JSON/Load testPractice (传统模式 - 无 JSON)")]
    public static void TestPracticeLegacyMode()
    {
        PracticeSceneInitializer.PendingMotionPackageUri = null;
        EditorSceneManager.OpenScene("Assets/Scenes/testPractice.unity");
        Debug.Log("[JsonMotionTestMenu] 已清除 PendingMotionPackageUri，testPractice 将使用传统模式");
    }

    [MenuItem("Spine Recovery/Test JSON/Load testGaming (传统模式 - 无 JSON)")]
    public static void TestGamingLegacyMode()
    {
        SpineFlowTrainingSession.CacheMotionPackage(null);
        EditorSceneManager.OpenScene("Assets/Scenes/testGaming.unity");
        Debug.Log("[JsonMotionTestMenu] 已清除缓存的 Package，testGaming 将使用传统模式");
    }

    [MenuItem("Spine Recovery/Test JSON/Check JSON Files")]
    public static void CheckJsonFiles()
    {
        string message = "JSON 文件检查结果:\n\n";

        message += CheckFile(DeadBugJsonPath, "DeadBug");
        message += CheckFile(BirdDogJsonPath, "BirdDog");

        EditorUtility.DisplayDialog("JSON 文件状态", message, "确定");
    }

    private static string CheckFile(string path, string name)
    {
        if (System.IO.File.Exists(path))
        {
            long fileSize = new System.IO.FileInfo(path).Length;
            return $"✓ {name}: 存在 ({fileSize} bytes)\n   路径: {path}\n\n";
        }
        else
        {
            return $"✗ {name}: 不存在\n   路径: {path}\n\n";
        }
    }
}
