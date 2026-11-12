using TMPro;
using UnityEngine;
using UnityEngine.UI;

// Non-interactive coach-mode progress indicator. Attach this to the existing
// follow-view feedback canvas so it remains visible without affecting gaze UI.
public class CoachSetProgressUI : MonoBehaviour
{
    [SerializeField] private SegmentedCoachController coachController;
    [SerializeField] private RectTransform progressRoot;
    [SerializeField] private TMP_Text progressText;

    [Header("Layout")]
    [SerializeField] private Vector2 anchoredPosition = new Vector2(0f, 180f);
    [SerializeField] private Vector2 size = new Vector2(300f, 64f);
    [SerializeField] private float fontSize = 42f;
    [SerializeField] private Color backgroundColor = new Color(0f, 0f, 0f, 0.58f);

    private void Awake()
    {
        ResolveController();
        EnsureVisuals();
        HideProgress();
    }

    private void OnEnable()
    {
        ResolveController();
        Subscribe();

        if (coachController != null && coachController.HasStarted && !coachController.IsComplete)
        {
            ShowProgress(coachController.CompletedSets, coachController.TotalSets);
        }
        else
        {
            HideProgress();
        }
    }

    private void OnDisable()
    {
        Unsubscribe();
    }

    private void ResolveController()
    {
        if (coachController == null)
        {
            coachController = FindObjectOfType<SegmentedCoachController>(true);
        }
    }

    private void Subscribe()
    {
        if (coachController == null)
        {
            return;
        }

        coachController.OnSetProgressChanged -= HandleSetProgressChanged;
        coachController.OnSetProgressChanged += HandleSetProgressChanged;
        coachController.OnAllComplete -= HideProgress;
        coachController.OnAllComplete += HideProgress;
    }

    private void Unsubscribe()
    {
        if (coachController == null)
        {
            return;
        }

        coachController.OnSetProgressChanged -= HandleSetProgressChanged;
        coachController.OnAllComplete -= HideProgress;
    }

    private void HandleSetProgressChanged(int completedSets, int totalSets)
    {
        if (totalSets <= 0 ||
            coachController == null ||
            !coachController.HasStarted ||
            coachController.IsComplete)
        {
            HideProgress();
            return;
        }

        ShowProgress(completedSets, totalSets);
    }

    private void ShowProgress(int completedSets, int totalSets)
    {
        EnsureVisuals();

        int safeTotal = Mathf.Max(1, totalSets);
        int safeCompleted = Mathf.Clamp(completedSets, 0, safeTotal);
        int currentSet = Mathf.Min(safeCompleted + 1, safeTotal);
        progressText.text = $"当前组数：{currentSet}/{safeTotal}";
        progressRoot.gameObject.SetActive(true);
    }

    private void HideProgress()
    {
        if (progressRoot != null)
        {
            progressRoot.gameObject.SetActive(false);
        }
    }

    private void EnsureVisuals()
    {
        if (progressRoot == null)
        {
            Transform existingRoot = transform.Find("CoachSetProgress");
            if (existingRoot != null)
            {
                progressRoot = existingRoot as RectTransform;
            }
        }

        if (progressRoot == null)
        {
            GameObject rootObject = new GameObject(
                "CoachSetProgress",
                typeof(RectTransform),
                typeof(CanvasRenderer),
                typeof(Image));
            rootObject.layer = gameObject.layer;
            progressRoot = rootObject.GetComponent<RectTransform>();
            progressRoot.SetParent(transform, false);
        }

        progressRoot.anchorMin = new Vector2(0.5f, 0.5f);
        progressRoot.anchorMax = new Vector2(0.5f, 0.5f);
        progressRoot.pivot = new Vector2(0.5f, 0.5f);
        progressRoot.anchoredPosition = anchoredPosition;
        progressRoot.sizeDelta = size;
        progressRoot.localRotation = Quaternion.identity;
        progressRoot.localScale = Vector3.one;
        progressRoot.SetAsLastSibling();

        Image background = progressRoot.GetComponent<Image>();
        if (background == null)
        {
            background = progressRoot.gameObject.AddComponent<Image>();
        }
        background.color = backgroundColor;
        background.raycastTarget = false;

        if (progressText == null)
        {
            progressText = progressRoot.GetComponentInChildren<TMP_Text>(true);
        }

        if (progressText == null)
        {
            GameObject textObject = new GameObject(
                "SetProgressText",
                typeof(RectTransform),
                typeof(CanvasRenderer),
                typeof(TextMeshProUGUI));
            textObject.layer = gameObject.layer;
            RectTransform textRect = textObject.GetComponent<RectTransform>();
            textRect.SetParent(progressRoot, false);
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.offsetMin = new Vector2(8f, 4f);
            textRect.offsetMax = new Vector2(-8f, -4f);
            textRect.localRotation = Quaternion.identity;
            textRect.localScale = Vector3.one;
            progressText = textObject.GetComponent<TextMeshProUGUI>();
        }

        progressText.alignment = TextAlignmentOptions.Center;
        progressText.fontSize = fontSize;
        progressText.fontStyle = FontStyles.Bold;
        progressText.color = Color.white;
        progressText.raycastTarget = false;
        progressText.text = "当前组数：0/1";

    }

    public RectTransform ProgressRoot => progressRoot;
    public TMP_Text ProgressText => progressText;
    public bool IsVisible => progressRoot != null && progressRoot.gameObject.activeInHierarchy;
}
