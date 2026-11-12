using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// 自动创建难度选择UI的辅助工具
/// 使用方法: 在Unity Editor的菜单栏选择 Tools > Create Difficulty Selection UI
/// </summary>
public class DifficultyUICreator : MonoBehaviour
{
#if UNITY_EDITOR
    [UnityEditor.MenuItem("Tools/Spine Recovery/Create Difficulty Selection UI")]
    public static void CreateDifficultySelectionUI()
    {
        try
        {
            // 创建Canvas
            GameObject canvasObj = new GameObject("DifficultySelectionCanvas");
            Canvas canvas = canvasObj.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.worldCamera = null;

            CanvasScaler scaler = canvasObj.AddComponent<CanvasScaler>();
            scaler.dynamicPixelsPerUnit = 10;

            canvasObj.AddComponent<GraphicRaycaster>();

            RectTransform canvasRect = canvasObj.GetComponent<RectTransform>();
            canvasRect.sizeDelta = new Vector2(1000, 800);
            canvasRect.localScale = Vector3.one;
            canvasRect.anchoredPosition3D = Vector3.zero;

            // 创建主面板
            GameObject panelObj = CreatePanel(canvasRect, "DifficultySelectionPanel", new Vector2(900, 700));
            RectTransform panelRect = panelObj.GetComponent<RectTransform>();

            // 创建标题
            CreateText(panelRect, "TitleText", "选择练习难度", new Vector2(0, 280), new Vector2(800, 80), 48, TextAnchor.MiddleCenter);

            Debug.Log("创建标题完成");

            // 创建稳定性区域
            Debug.Log("开始创建稳定性区域");
            CreateDifficultySection(panelRect, "StabilitySection", "稳定性要求", new Vector2(0, 100),
                out Button stabRelaxed, out Button stabMedium, out Button stabStrict,
                out Text stabLevelText, out Text stabDescText);
            Debug.Log("稳定性区域创建完成");

            // 创建评分区域
            Debug.Log("开始创建评分区域");
            CreateDifficultySection(panelRect, "ScoringSection", "评分严格度", new Vector2(0, -150),
                out Button scoreRelaxed, out Button scoreMedium, out Button scoreStrict,
                out Text scoreLevelText, out Text scoreDescText);
            Debug.Log("评分区域创建完成");

            // 创建确认按钮
            Debug.Log("开始创建确认按钮");
            Button confirmButton = CreateButton(panelRect, "ConfirmButton", "确认开始",
                new Vector2(0, -300), new Vector2(300, 80), 36);
            Debug.Log("确认按钮创建完成");

            // 保留视觉占位对象；交互完全由手机触控按钮处理。
            Debug.Log("开始创建进度环");
            GameObject ringObj = CreateProgressRing(canvasRect, "SelectionProgressRing");
            Debug.Log("进度环创建完成");

            // 添加并配置 DifficultySelectionUI 组件
            Debug.Log("开始添加 DifficultySelectionUI 组件");
            DifficultySelectionUI uiController = canvasObj.AddComponent<DifficultySelectionUI>();

            UnityEditor.SerializedObject so = new UnityEditor.SerializedObject(uiController);
            so.FindProperty("selectionPanel").objectReferenceValue = panelObj;
            so.FindProperty("stabilityRelaxedButton").objectReferenceValue = stabRelaxed;
            so.FindProperty("stabilityMediumButton").objectReferenceValue = stabMedium;
            so.FindProperty("stabilityStrictButton").objectReferenceValue = stabStrict;
            so.FindProperty("scoringRelaxedButton").objectReferenceValue = scoreRelaxed;
            so.FindProperty("scoringMediumButton").objectReferenceValue = scoreMedium;
            so.FindProperty("scoringStrictButton").objectReferenceValue = scoreStrict;
            so.FindProperty("confirmButton").objectReferenceValue = confirmButton;
            so.FindProperty("stabilityLevelText").objectReferenceValue = stabLevelText;
            so.FindProperty("scoringLevelText").objectReferenceValue = scoreLevelText;
            so.FindProperty("stabilityDescriptionText").objectReferenceValue = stabDescText;
            so.FindProperty("scoringDescriptionText").objectReferenceValue = scoreDescText;
            so.ApplyModifiedProperties();
            Debug.Log("DifficultySelectionUI 组件配置完成");

            // 手机触控通过 GraphicRaycaster + EventSystem 处理，不添加凝视点击器。

            // 选中创建的对象
            UnityEditor.Selection.activeGameObject = canvasObj;

            Debug.Log("=== 难度选择UI创建完成！请配置 DifficultySettings 资源并调整UI位置。===");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"创建UI时发生错误: {e.Message}\n{e.StackTrace}");
        }
    }

    private static void CreateDifficultySection(RectTransform parent, string name, string label, Vector2 position,
        out Button relaxedBtn, out Button mediumBtn, out Button strictBtn,
        out Text levelText, out Text descText)
    {
        GameObject section = new GameObject(name);
        section.transform.SetParent(parent, false);
        RectTransform sectionRect = section.AddComponent<RectTransform>();
        sectionRect.anchoredPosition = position;
        sectionRect.sizeDelta = new Vector2(800, 180);

        // 标签
        CreateText(sectionRect, "Label", label, new Vector2(0, 70), new Vector2(600, 40), 32, TextAnchor.MiddleCenter);

        // 当前选择显示
        levelText = CreateText(sectionRect, "LevelText", "当前: 中等", new Vector2(0, 35), new Vector2(600, 30), 24, TextAnchor.MiddleCenter);
        descText = CreateText(sectionRect, "DescriptionText", "推荐选择\n标准的姿势要求", new Vector2(0, -10), new Vector2(600, 50), 20, TextAnchor.MiddleCenter);

        // 按钮组 - 用颜色区分，不用文字
        // 白色 = 宽松/简单
        relaxedBtn = CreateColorButton(sectionRect, "RelaxedButton", new Vector2(-220, -50), new Vector2(150, 150), Color.white);
        // 绿色 = 中等
        mediumBtn = CreateColorButton(sectionRect, "MediumButton", new Vector2(0, -50), new Vector2(150, 150), new Color(0.2f, 0.8f, 0.2f));
        // 红色 = 严格/困难
        strictBtn = CreateColorButton(sectionRect, "StrictButton", new Vector2(220, -50), new Vector2(150, 150), new Color(0.9f, 0.2f, 0.2f));
    }

    private static GameObject CreatePanel(RectTransform parent, string name, Vector2 size)
    {
        GameObject panelObj = new GameObject(name);
        panelObj.transform.SetParent(parent, false);

        RectTransform rect = panelObj.AddComponent<RectTransform>();
        rect.sizeDelta = size;
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.pivot = new Vector2(0.5f, 0.5f);

        Image image = panelObj.AddComponent<Image>();
        image.color = new Color(0.1f, 0.1f, 0.15f, 0.95f);

        return panelObj;
    }

    private static Text CreateText(RectTransform parent, string name, string content, Vector2 position, Vector2 size, int fontSize, TextAnchor alignment)
    {
        GameObject textObj = new GameObject(name);
        textObj.transform.SetParent(parent, false);

        RectTransform rect = textObj.AddComponent<RectTransform>();
        rect.anchoredPosition = position;
        rect.sizeDelta = size;

        Text text = textObj.AddComponent<Text>();
        text.text = content;

        // 尝试获取内置字体，如果失败则使用默认字体
        try
        {
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        }
        catch
        {
            // 如果获取内置字体失败，使用Resources中的字体或留空让Unity使用默认字体
            text.font = Resources.Load<Font>("Arial");
            if (text.font == null)
            {
                Debug.LogWarning("未找到字体，将使用Unity默认字体");
            }
        }

        text.fontSize = fontSize;
        text.alignment = alignment;
        text.color = Color.white;

        return text;
    }

    private static Button CreateButton(RectTransform parent, string name, string label, Vector2 position, Vector2 size, int fontSize)
    {
        GameObject buttonObj = new GameObject(name);
        buttonObj.transform.SetParent(parent, false);

        RectTransform rect = buttonObj.AddComponent<RectTransform>();
        rect.anchoredPosition = position;
        rect.sizeDelta = size;

        Image image = buttonObj.AddComponent<Image>();
        image.color = Color.white;

        Button button = buttonObj.AddComponent<Button>();
        button.targetGraphic = image;

        // 按钮文字
        GameObject textObj = new GameObject("Text");
        textObj.transform.SetParent(buttonObj.transform, false);

        RectTransform textRect = textObj.AddComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.sizeDelta = Vector2.zero;

        Text text = textObj.AddComponent<Text>();
        text.text = label;

        // 尝试获取内置字体，如果失败则使用默认字体
        try
        {
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        }
        catch
        {
            text.font = Resources.Load<Font>("Arial");
            if (text.font == null)
            {
                Debug.LogWarning("未找到字体，将使用Unity默认字体");
            }
        }

        text.fontSize = fontSize;
        text.alignment = TextAnchor.MiddleCenter;
        text.color = Color.black;

        return button;
    }

    private static Button CreateColorButton(RectTransform parent, string name, Vector2 position, Vector2 size, Color color)
    {
        GameObject buttonObj = new GameObject(name);
        buttonObj.transform.SetParent(parent, false);

        RectTransform rect = buttonObj.AddComponent<RectTransform>();
        rect.anchoredPosition = position;
        rect.sizeDelta = size;

        Image image = buttonObj.AddComponent<Image>();
        image.color = color;

        Button button = buttonObj.AddComponent<Button>();
        button.targetGraphic = image;

        // 不添加文字，纯色按钮

        return button;
    }

    private static GameObject CreateProgressRing(RectTransform parent, string name)
    {
        GameObject ringObj = new GameObject(name);
        ringObj.transform.SetParent(parent, false);

        RectTransform rect = ringObj.AddComponent<RectTransform>();
        rect.sizeDelta = new Vector2(80, 80);

        Image image = ringObj.AddComponent<Image>();
        image.color = new Color(0.2f, 0.8f, 1f, 0.8f);
        image.type = Image.Type.Filled;
        image.fillMethod = Image.FillMethod.Radial360;
        image.fillOrigin = (int)Image.Origin360.Top;
        image.fillClockwise = true;
        image.fillAmount = 0f;
        image.raycastTarget = false;

        ringObj.SetActive(false);

        return ringObj;
    }
#endif
}
