using UnityEngine;
using UnityEngine.UI;
using System;

/// <summary>
/// 难度选择UI控制器 - 在练习开始前让用户选择稳定性和评分的难度等级
/// 使用手机触控按钮进行交互。
/// </summary>
public class DifficultySelectionUI : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private DifficultySettings difficultySettings;
    [SerializeField] private GameObject selectionPanel;

    [Header("One-time placement")]
    [Tooltip("Camera used only once when the selection panel opens.")]
    [SerializeField] private Camera viewCamera;
    [SerializeField] private float viewDistance = 1.5f;

    [Header("稳定性难度按钮")]
    [SerializeField] private Button stabilityRelaxedButton;
    [SerializeField] private Button stabilityMediumButton;
    [SerializeField] private Button stabilityStrictButton;

    [Header("评分难度按钮")]
    [SerializeField] private Button scoringRelaxedButton;
    [SerializeField] private Button scoringMediumButton;
    [SerializeField] private Button scoringStrictButton;

    [Header("自动关闭设置")]
    [Tooltip("选择完难度后自动关闭面板（不需要点确认按钮）")]
    [SerializeField] private bool autoCloseAfterSelection = true;
    [Tooltip("选择后延迟关闭的时间（秒）")]
    [SerializeField] private float autoCloseDelay = 0.5f;

    [Header("显示文本（可选）")]
    [SerializeField] private Text stabilityLevelText;
    [SerializeField] private Text scoringLevelText;
    [SerializeField] private Text stabilityDescriptionText;
    [SerializeField] private Text scoringDescriptionText;

    [Header("按钮高亮效果")]
    [SerializeField] private float selectedBrightness = 1.0f; // 选中时保持原色
    [SerializeField] private float unselectedBrightness = 0.5f; // 未选中时变暗
    [SerializeField] private float selectedScale = 1.2f; // 选中时放大

    // 当前选择
    private DifficultyLevel selectedStabilityLevel = DifficultyLevel.Relaxed;
    private DifficultyLevel selectedScoringLevel = DifficultyLevel.Relaxed;
    private bool stabilitySelected = false;
    private bool scoringSelected = false;

    // 事件：用户完成难度选择后触发（自动关闭时触发，不需要确认按钮）
    public event Action OnDifficultyConfirmed;

    private void Awake()
    {
        // 如果没有配置DifficultySettings，尝试从Resources加载
        if (difficultySettings == null)
        {
            difficultySettings = Resources.Load<DifficultySettings>("DifficultySettings");
        }

        // 设置按钮监听
        if (stabilityRelaxedButton != null)
            stabilityRelaxedButton.onClick.AddListener(() => SelectStabilityLevel(DifficultyLevel.Relaxed));
        if (stabilityMediumButton != null)
            stabilityMediumButton.onClick.AddListener(() => SelectStabilityLevel(DifficultyLevel.Medium));
        if (stabilityStrictButton != null)
            stabilityStrictButton.onClick.AddListener(() => SelectStabilityLevel(DifficultyLevel.Strict));

        if (scoringRelaxedButton != null)
            scoringRelaxedButton.onClick.AddListener(() => SelectScoringLevel(DifficultyLevel.Relaxed));
        if (scoringMediumButton != null)
            scoringMediumButton.onClick.AddListener(() => SelectScoringLevel(DifficultyLevel.Medium));
        if (scoringStrictButton != null)
            scoringStrictButton.onClick.AddListener(() => SelectScoringLevel(DifficultyLevel.Strict));

        // 初始化默认值（不触发选择状态）
        selectedStabilityLevel = DifficultyLevel.Relaxed;
        selectedScoringLevel = DifficultyLevel.Relaxed;
        UpdateStabilityButtons();
        UpdateScoringButtons();
        UpdateStabilityText();
        UpdateScoringText();
    }

    /// <summary>
    /// 显示难度选择面板
    /// </summary>
    public void Show()
    {
        if (selectionPanel != null)
        {
            selectionPanel.SetActive(true);
        }

        // 重置选择状态（重新打开时需要重新选择）
        stabilitySelected = false;
        scoringSelected = false;

        // 显示当前已选择的难度（但不标记为已选择）
        UpdateStabilityButtons();
        UpdateScoringButtons();
        UpdateStabilityText();
        UpdateScoringText();

        Debug.Log("难度选择界面已显示，请选择稳定性和评分难度");
    }

    /// <summary>
    /// 隐藏难度选择面板
    /// </summary>
    public void Hide()
    {
        if (selectionPanel != null)
        {
            selectionPanel.SetActive(false);
        }
    }

    /// <summary>
    /// 选择稳定性难度等级
    /// </summary>
    private void SelectStabilityLevel(DifficultyLevel level)
    {
        selectedStabilityLevel = level;
        stabilitySelected = true;
        UpdateStabilityButtons();
        UpdateStabilityText();

        // 如果开启自动关闭且两个都已选择，自动关闭
        if (autoCloseAfterSelection && stabilitySelected && scoringSelected)
        {
            StartCoroutine(AutoCloseAfterDelay());
        }
    }

    /// <summary>
    /// 选择评分难度等级
    /// </summary>
    private void SelectScoringLevel(DifficultyLevel level)
    {
        selectedScoringLevel = level;
        scoringSelected = true;
        UpdateScoringButtons();
        UpdateScoringText();

        // 如果开启自动关闭且两个都已选择，自动关闭
        if (autoCloseAfterSelection && stabilitySelected && scoringSelected)
        {
            StartCoroutine(AutoCloseAfterDelay());
        }
    }

    /// <summary>
    /// 延迟自动关闭
    /// </summary>
    private System.Collections.IEnumerator AutoCloseAfterDelay()
    {
        yield return new WaitForSeconds(autoCloseDelay);
        ApplyAndClose();
    }

    /// <summary>
    /// 应用难度并关闭面板
    /// </summary>
    private void ApplyAndClose()
    {
        Debug.Log($"难度已选择 - 稳定性: {DifficultySettings.GetDifficultyName(selectedStabilityLevel)}, " +
                  $"评分: {DifficultySettings.GetDifficultyName(selectedScoringLevel)}");

        // 触发确认事件（通知 PracticeSessionController 应用难度）
        OnDifficultyConfirmed?.Invoke();

        // 隐藏面板
        Hide();
    }

    /// <summary>
    /// 更新稳定性按钮的视觉状态
    /// </summary>
    private void UpdateStabilityButtons()
    {
        SetButtonSelected(stabilityRelaxedButton, selectedStabilityLevel == DifficultyLevel.Relaxed);
        SetButtonSelected(stabilityMediumButton, selectedStabilityLevel == DifficultyLevel.Medium);
        SetButtonSelected(stabilityStrictButton, selectedStabilityLevel == DifficultyLevel.Strict);
    }

    /// <summary>
    /// 更新评分按钮的视觉状态
    /// </summary>
    private void UpdateScoringButtons()
    {
        SetButtonSelected(scoringRelaxedButton, selectedScoringLevel == DifficultyLevel.Relaxed);
        SetButtonSelected(scoringMediumButton, selectedScoringLevel == DifficultyLevel.Medium);
        SetButtonSelected(scoringStrictButton, selectedScoringLevel == DifficultyLevel.Strict);
    }

    /// <summary>
    /// 设置按钮的视觉状态（选中/未选中）
    /// </summary>
    private void SetButtonSelected(Button button, bool selected)
    {
        if (button == null) return;

        Image buttonImage = button.GetComponent<Image>();
        if (buttonImage != null)
        {
            // 获取按钮的原始颜色
            Color originalColor = GetOriginalButtonColor(button);

            // 选中时保持原色并放大，未选中时变暗并缩小
            float brightness = selected ? selectedBrightness : unselectedBrightness;
            buttonImage.color = originalColor * brightness;
        }

        // 缩放效果
        RectTransform rectTransform = button.GetComponent<RectTransform>();
        if (rectTransform != null)
        {
            float scale = selected ? selectedScale : 1.0f;
            rectTransform.localScale = Vector3.one * scale;
        }
    }

    /// <summary>
    /// 获取按钮的原始颜色（根据按钮名称判断）
    /// </summary>
    private Color GetOriginalButtonColor(Button button)
    {
        if (button.name.Contains("Relaxed"))
            return Color.white; // 白色 = 简单
        else if (button.name.Contains("Medium"))
            return new Color(0.2f, 0.8f, 0.2f); // 绿色 = 中等
        else if (button.name.Contains("Strict"))
            return new Color(0.9f, 0.2f, 0.2f); // 红色 = 困难
        else
            return Color.white;
    }

    /// <summary>
    /// 更新稳定性文本显示
    /// </summary>
    private void UpdateStabilityText()
    {
        if (stabilityLevelText != null)
        {
            stabilityLevelText.text = $"稳定性要求: {DifficultySettings.GetDifficultyName(selectedStabilityLevel)}";
        }

        if (stabilityDescriptionText != null)
        {
            stabilityDescriptionText.text = DifficultySettings.GetDifficultyDescription(selectedStabilityLevel);
        }
    }

    /// <summary>
    /// 更新评分文本显示
    /// </summary>
    private void UpdateScoringText()
    {
        if (scoringLevelText != null)
        {
            scoringLevelText.text = $"评分严格度: {DifficultySettings.GetDifficultyName(selectedScoringLevel)}";
        }

        if (scoringDescriptionText != null)
        {
            scoringDescriptionText.text = DifficultySettings.GetDifficultyDescription(selectedScoringLevel);
        }
    }

    /// <summary>
    /// 获取选中的稳定性难度等级
    /// </summary>
    public DifficultyLevel GetSelectedStabilityLevel()
    {
        return selectedStabilityLevel;
    }

    /// <summary>
    /// 获取选中的评分难度等级
    /// </summary>
    public DifficultyLevel GetSelectedScoringLevel()
    {
        return selectedScoringLevel;
    }

    /// <summary>
    /// 获取稳定性配置
    /// </summary>
    public DifficultyConfig GetStabilityConfig()
    {
        if (difficultySettings == null)
        {
            Debug.LogWarning("DifficultySettings is null, using default medium config");
            return new DifficultyConfig();
        }
        return difficultySettings.GetConfig(selectedStabilityLevel);
    }

    /// <summary>
    /// 获取评分配置
    /// </summary>
    public DifficultyConfig GetScoringConfig()
    {
        if (difficultySettings == null)
        {
            Debug.LogWarning("DifficultySettings is null, using default medium config");
            return new DifficultyConfig();
        }
        return difficultySettings.GetConfig(selectedScoringLevel);
    }

    public DifficultySettings Settings => difficultySettings;
}
