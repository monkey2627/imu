using UnityEngine;
using UnityEngine.UI;

public class CoachKeyframeUI : MonoBehaviour
{
    [Header("Coach Reference")]
    [SerializeField] private SegmentedCoachController coachController;

    [Header("UI Elements")]
    [SerializeField] private Button nextStepButton;
    [SerializeField] private Text buttonText;

    private void Start()
    {
        // Try to find coach controller if not assigned
        if (coachController == null)
        {
            coachController = FindObjectOfType<SegmentedCoachController>();
        }

        // Setup button click
        if (nextStepButton != null)
        {
            nextStepButton.onClick.AddListener(OnNextStepClicked);
        }

        Subscribe(coachController);

        // Update button text initially
        UpdateButtonText();
    }

    private void OnDestroy()
    {
        if (nextStepButton != null)
        {
            nextStepButton.onClick.RemoveListener(OnNextStepClicked);
        }

        Unsubscribe(coachController);
    }

    private void OnNextStepClicked()
    {
        if (coachController != null)
        {
            coachController.AdvanceToNext();
            UpdateButtonText();
        }
    }

    private void HandleSegmentHold(SegmentHoldInfo info)
    {
        UpdateButtonText();
    }

    private void HandleAllComplete()
    {
        UpdateButtonText();
    }

    private void UpdateButtonText()
    {
        if (buttonText == null || coachController == null)
        {
            return;
        }

        if (coachController.IsComplete)
        {
            buttonText.text = "完成";
            return;
        }

        int segment = coachController.CurrentSegmentIndex + 1;
        int totalSegments = coachController.TotalSegments;
        int rep = coachController.CurrentRepeat + 1;
        int repTotal = coachController.CurrentSegmentRepeatCount;
        int set = coachController.CurrentSetIndex + 1;
        int totalSets = coachController.TotalSets;

        // Build a progress label, hiding the parts that are trivial (single rep / single set).
        string text = $"下一步 ({segment}/{totalSegments}";
        if (repTotal > 1)
        {
            text += $" · 第{rep}/{repTotal}次";
        }
        if (totalSets > 1)
        {
            text += $" · 组{set}/{totalSets}";
        }
        text += ")";

        buttonText.text = text;
    }

    public void SetCoachController(SegmentedCoachController controller)
    {
        if (controller == coachController)
        {
            return;
        }

        Unsubscribe(coachController);
        coachController = controller;
        Subscribe(coachController);
        UpdateButtonText();
    }

    private void Subscribe(SegmentedCoachController controller)
    {
        if (controller == null)
        {
            return;
        }

        controller.OnSegmentHold += HandleSegmentHold;
        controller.OnAllComplete += HandleAllComplete;
    }

    private void Unsubscribe(SegmentedCoachController controller)
    {
        if (controller == null)
        {
            return;
        }

        controller.OnSegmentHold -= HandleSegmentHold;
        controller.OnAllComplete -= HandleAllComplete;
    }
}
