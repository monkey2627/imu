using System.Collections;
using System;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

/// <summary>
/// 评分窗口内常驻显示实时分数，结束后切换为平均分并自动淡出。
/// </summary>
public class ScorePopup : MonoBehaviour
{
    [SerializeField] private TMP_Text scoreText;
    [SerializeField] private CanvasGroup canvasGroup;
    [SerializeField] private float displayDuration = 3f;
    [SerializeField] private float fadeOutDuration = 0.5f;
    [SerializeField] private float fontSize = 72f; // 字体大小

    private Coroutine displayRoutine;
    private Image backgroundImage; // 背景图片引用

    public event Action DisplayFinished;
    public bool IsShowing { get; private set; }

    private void Awake()
    {
        if (canvasGroup == null)
        {
            canvasGroup = GetComponent<CanvasGroup>();
            if (canvasGroup == null)
            {
                canvasGroup = gameObject.AddComponent<CanvasGroup>();
            }
        }

        if (scoreText == null)
        {
            scoreText = GetComponentInChildren<TMP_Text>();
        }

        // 设置字体大小
        if (scoreText != null)
        {
            scoreText.fontSize = fontSize;
        }

        // 找到背景Image并设置为透明
        backgroundImage = GetComponent<Image>();
        if (backgroundImage != null)
        {
            Color c = backgroundImage.color;
            c.a = 0f; // 完全透明
            backgroundImage.color = c;
        }

        // 初始隐藏（使用alpha而不是SetActive）
        canvasGroup.alpha = 0f;
        canvasGroup.interactable = false;
        canvasGroup.blocksRaycasts = false;
    }

    /// <summary>
    /// 显示分数
    /// </summary>
    public void ShowScore(float score)
    {
        ShowScore(score, null);
    }

    /// <summary>
    /// 显示分数和一条简短、可执行的鼓励反馈。
    /// </summary>
    public void ShowScore(float score, string feedback)
    {
        ShowTimedScore("得分", score, feedback);
    }

    public void ShowLiveScore(float score)
    {
        if (!PrepareForDisplay())
        {
            return;
        }

        StopDisplayRoutine();
        scoreText.text = FormatLiveScoreText(score);
        ShowCanvas();
        IsShowing = true;
    }

    public void ShowAverageScore(float score, string feedback)
    {
        ShowTimedScore("平均分", score, feedback);
    }

    public static string FormatLiveScoreText(float score)
    {
        return $"实时分数: {FormalActionScoreMappingSettings.RoundForDisplay(score)}";
    }

    public static string FormatAverageScoreText(float score, string feedback)
    {
        int displayed =
            FormalActionScoreMappingSettings.RoundForDisplay(score);
        return string.IsNullOrWhiteSpace(feedback)
            ? $"平均分: {displayed}"
            : $"平均分: {displayed}\n<size=45%><nobr>{feedback}</nobr></size>";
    }

    public void HideImmediately()
    {
        StopDisplayRoutine();
        if (canvasGroup != null)
        {
            canvasGroup.alpha = 0f;
        }

        bool wasShowing = IsShowing;
        IsShowing = false;
        if (wasShowing)
        {
            DisplayFinished?.Invoke();
        }
    }

    private void ShowTimedScore(string label, float score, string feedback)
    {
        if (!PrepareForDisplay())
        {
            return;
        }

        StopDisplayRoutine();
        IsShowing = true;
        displayRoutine = StartCoroutine(
            DisplayScoreRoutine(label, score, feedback));
    }

    private bool PrepareForDisplay()
    {
        if (scoreText == null)
        {
            Debug.LogError("ScorePopup: scoreText 未配置！");
            return false;
        }

        if (!gameObject.activeSelf)
        {
            Debug.LogWarning("ScorePopup: GameObject被禁用，正在激活它");
            gameObject.SetActive(true);
        }

        return true;
    }

    private void StopDisplayRoutine()
    {
        if (displayRoutine == null)
        {
            return;
        }

        StopCoroutine(displayRoutine);
        displayRoutine = null;
    }

    private void ShowCanvas()
    {
        canvasGroup.alpha = 1f;
        canvasGroup.interactable = false;
        canvasGroup.blocksRaycasts = false;
    }

    private IEnumerator DisplayScoreRoutine(
        string label,
        float score,
        string feedback)
    {
        scoreText.text = label == "平均分"
            ? FormatAverageScoreText(score, feedback)
            : string.IsNullOrWhiteSpace(feedback)
                ? $"{label}: {FormalActionScoreMappingSettings.RoundForDisplay(score)}"
                : $"{label}: {FormalActionScoreMappingSettings.RoundForDisplay(score)}\n" +
                  $"<size=45%>{feedback}</size>";

        ShowCanvas();

        // 等待显示时间
        yield return new WaitForSeconds(displayDuration);

        // 淡出
        float elapsed = 0f;
        while (elapsed < fadeOutDuration)
        {
            elapsed += Time.deltaTime;
            canvasGroup.alpha = 1f - (elapsed / fadeOutDuration);
            yield return null;
        }

        // 隐藏UI（使用alpha而不是SetActive）
        canvasGroup.alpha = 0f;

        displayRoutine = null;
        IsShowing = false;
        DisplayFinished?.Invoke();
    }

    private void OnDisable()
    {
        HideImmediately();
    }
}
