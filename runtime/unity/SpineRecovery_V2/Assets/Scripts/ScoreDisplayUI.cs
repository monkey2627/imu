using System.Collections;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// 简洁的分数显示UI - 评分结束时短暂显示分数
/// </summary>
public class ScoreDisplayUI : MonoBehaviour
{
    [Header("显示设置")]
    [SerializeField] private float displayDuration = 3f; // 显示时长
    [SerializeField] private float fadeInDuration = 0.3f; // 淡入时长
    [SerializeField] private float fadeOutDuration = 0.5f; // 淡出时长

    [Header("样式")]
    [SerializeField] private int fontSize = 80;
    [SerializeField] private Color scoreColor = new Color(1f, 0.85f, 0.2f); // 金色
    [SerializeField] private Font customFont; // 可选：自定义字体

    private Canvas canvas;
    private Text scoreText;
    private CanvasGroup canvasGroup;
    private Coroutine displayCoroutine;

    private void Awake()
    {
        CreateUI();
    }

    /// <summary>
    /// 显示分数
    /// </summary>
    public void ShowScore(float score)
    {
        if (displayCoroutine != null)
        {
            StopCoroutine(displayCoroutine);
        }
        displayCoroutine = StartCoroutine(ShowScoreSequence(score));
    }

    private void CreateUI()
    {
        // 创建 Canvas
        GameObject canvasGo = new GameObject("ScoreDisplayCanvas");
        canvasGo.transform.SetParent(transform, false);

        canvas = canvasGo.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 1000; // 最上层

        canvasGo.AddComponent<CanvasScaler>();
        canvasGo.AddComponent<GraphicRaycaster>();

        canvasGroup = canvasGo.AddComponent<CanvasGroup>();
        canvasGroup.alpha = 0f;
        canvasGroup.blocksRaycasts = false;

        // 创建背景面板（半透明黑色）
        GameObject panelGo = new GameObject("ScorePanel");
        panelGo.transform.SetParent(canvasGo.transform, false);

        RectTransform panelRt = panelGo.AddComponent<RectTransform>();
        panelRt.anchorMin = new Vector2(0.5f, 0.5f);
        panelRt.anchorMax = new Vector2(0.5f, 0.5f);
        panelRt.pivot = new Vector2(0.5f, 0.5f);
        panelRt.anchoredPosition = Vector2.zero;
        panelRt.sizeDelta = new Vector2(500f, 200f);

        Image panelImage = panelGo.AddComponent<Image>();
        panelImage.color = new Color(0f, 0f, 0f, 0.7f);

        // 添加圆角效果（通过 Outline 模拟）
        Outline outline = panelGo.AddComponent<Outline>();
        outline.effectColor = new Color(1f, 1f, 1f, 0.3f);
        outline.effectDistance = new Vector2(3f, 3f);

        // 创建分数文字
        GameObject textGo = new GameObject("ScoreText");
        textGo.transform.SetParent(panelGo.transform, false);

        RectTransform textRt = textGo.AddComponent<RectTransform>();
        textRt.anchorMin = Vector2.zero;
        textRt.anchorMax = Vector2.one;
        textRt.offsetMin = Vector2.zero;
        textRt.offsetMax = Vector2.zero;

        scoreText = textGo.AddComponent<Text>();
        scoreText.fontSize = fontSize;
        scoreText.color = scoreColor;
        scoreText.alignment = TextAnchor.MiddleCenter;
        scoreText.horizontalOverflow = HorizontalWrapMode.Overflow;
        scoreText.verticalOverflow = VerticalWrapMode.Overflow;

        // 设置字体
        if (customFont != null)
        {
            scoreText.font = customFont;
        }
        else
        {
            // 尝试使用系统自带的字体
            scoreText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf")
                             ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
        }

        // 添加描边效果
        Outline textOutline = textGo.AddComponent<Outline>();
        textOutline.effectColor = new Color(0f, 0f, 0f, 0.8f);
        textOutline.effectDistance = new Vector2(3f, -3f);

        // 添加阴影
        Shadow shadow = textGo.AddComponent<Shadow>();
        shadow.effectColor = new Color(0f, 0f, 0f, 0.5f);
        shadow.effectDistance = new Vector2(5f, -5f);
    }

    private IEnumerator ShowScoreSequence(float score)
    {
        // 设置文字
        scoreText.text =
            $"得分\n{FormalActionScoreMappingSettings.RoundForDisplay(score)}";

        // 淡入
        float elapsed = 0f;
        while (elapsed < fadeInDuration)
        {
            elapsed += Time.deltaTime;
            canvasGroup.alpha = Mathf.Lerp(0f, 1f, elapsed / fadeInDuration);
            yield return null;
        }
        canvasGroup.alpha = 1f;

        // 保持显示
        yield return new WaitForSeconds(displayDuration);

        // 淡出
        elapsed = 0f;
        while (elapsed < fadeOutDuration)
        {
            elapsed += Time.deltaTime;
            canvasGroup.alpha = Mathf.Lerp(1f, 0f, elapsed / fadeOutDuration);
            yield return null;
        }
        canvasGroup.alpha = 0f;

        displayCoroutine = null;
    }

    /// <summary>
    /// 立即隐藏分数显示
    /// </summary>
    public void Hide()
    {
        if (displayCoroutine != null)
        {
            StopCoroutine(displayCoroutine);
            displayCoroutine = null;
        }
        canvasGroup.alpha = 0f;
    }
}
