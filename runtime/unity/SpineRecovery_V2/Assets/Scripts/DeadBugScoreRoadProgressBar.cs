using System.Collections.Generic;
using UnityEngine;

// Uses SliderBG as the fixed track and Slider1 as the score-colored fill.
// Scoring/progress starts only after the coach has finished lying down because
// those events are raised by CoachActionController's scored sequence.
public class DeadBugScoreRoadProgressBar : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private DeadBugGamingPoseScorer poseScorer;
    [SerializeField] private CoachActionController coachController;
    [Tooltip("The SpriteRenderer on SliderBG. Its sprite width is the full progress length.")]
    [SerializeField] private SpriteRenderer sliderBackground;
    [Tooltip("The SpriteRenderer on Slider1. It is resized from left to right.")]
    [SerializeField] private SpriteRenderer sliderFill;
    [Tooltip("The EnergyNova star under SliderBG.")]
    [SerializeField] private Transform scoreStarTransform;

    [Header("Evaluation Score Colors")]
    [Tooltip("Scores strictly below this value are red.")]
    [Range(0f, 100f)] [SerializeField] private float redBelowScore = 50f;
    [Tooltip("Scores at or above this value are green.")]
    [Range(0f, 100f)] [SerializeField] private float greenAboveScore = 70f;
    [SerializeField] private Color needsAdjustmentColor = new Color(1f, 0.48f, 0.35f, 1f);
    [SerializeField] private Color goodColor = new Color(1f, 0.85f, 0.25f, 1f);
    [SerializeField] private Color excellentColor = new Color(0.25f, 1f, 0.55f, 1f);

    [Header("Star Movement")]
    [Min(0f)] [SerializeField] private float starEdgePadding;

    [Header("Score Segments")]
    [Tooltip("Visual gap in local units between independently scored sections.")]
    [Min(0f)] [SerializeField] private float segmentGap = 0.02f;

    private float backgroundLocalWidth;
    private float fillSpriteLocalWidth;
    private float fillLocalY;
    private float fillLocalZ;
    private float starLocalY;
    private float starLocalZ;
    private float currentProgress;
    private float currentScore;
    private DeadBugScoreStarFeedback scoreStar;
    private Sprite originalFillSprite;
    private Sprite solidColorFillSprite;
    private Texture2D solidColorFillTexture;
    private readonly List<SpriteRenderer> scoreSegments = new List<SpriteRenderer>();
    private readonly List<GameObject> generatedSegmentObjects = new List<GameObject>();
    private readonly List<Color> lockedSegmentColors = new List<Color>();
    private readonly List<bool> segmentIsLocked = new List<bool>();
    private int completedSegmentCount;
    private float fillLocalScaleY = 1f;
    private float fillLocalScaleZ = 1f;
    private Quaternion fillLocalRotation = Quaternion.identity;
    private bool fillTransformCached;

    public float CurrentProgress => currentProgress;
    public float CurrentScore => currentScore;

    private void Awake()
    {
        ResolveReferences();
        CacheSliderGeometry();
        SetProgress(0f);
        SetScore(0f);
    }

    private void OnEnable()
    {
        ResolveReferences();
        CacheSliderGeometry();
        Subscribe();
    }

    private void OnDisable()
    {
        Unsubscribe();
    }

    private void OnDestroy()
    {
        Unsubscribe();
        DestroyGeneratedSegments();

        if (sliderFill != null && sliderFill.sprite == solidColorFillSprite)
        {
            sliderFill.sprite = originalFillSprite;
        }

        if (solidColorFillSprite != null)
        {
            Destroy(solidColorFillSprite);
        }

        if (solidColorFillTexture != null)
        {
            Destroy(solidColorFillTexture);
        }
    }

    private void HandleSessionStarted()
    {
        int segmentCount = coachController != null
            ? Mathf.Max(1, coachController.CurrentTotalScoredSegments)
            : 1;
        BuildScoreSegments(segmentCount);
        SetProgress(0f);
        SetScore(0f);

        if (scoreStar != null)
        {
            scoreStarTransform.gameObject.SetActive(true);
            scoreStar.BeginSession(0f, EvaluateScoreColor(0f));
        }
    }

    private void HandleScoringProgressChanged(float progress)
    {
        SetProgress(progress);
    }

    private void HandleLiveScoreChanged(float score)
    {
        SetScore(score);
    }

    private void HandleCheckpointScored(DeadBugCheckpointScore checkpointScore)
    {
        if (completedSegmentCount >= scoreSegments.Count)
        {
            return;
        }

        lockedSegmentColors[completedSegmentCount] =
            EvaluateScoreColor(checkpointScore.averageScore);
        segmentIsLocked[completedSegmentCount] = true;
        completedSegmentCount++;

        float completedProgress = scoreSegments.Count > 0
            ? (float)completedSegmentCount / scoreSegments.Count
            : 0f;
        SetProgress(Mathf.Max(currentProgress, completedProgress));
    }

    private void HandleSessionScored(
        float sessionAverage,
        System.Collections.Generic.IReadOnlyList<DeadBugCheckpointScore> checkpointScores)
    {
        SetProgress(1f);
        SetScore(sessionAverage);
        if (scoreStar != null)
        {
            scoreStar.EndSession(sessionAverage, EvaluateScoreColor(sessionAverage));
        }
    }

    private void SetProgress(float progress)
    {
        currentProgress = Mathf.Clamp01(progress);
        if (scoreSegments.Count == 0 || backgroundLocalWidth <= 0f ||
            fillSpriteLocalWidth <= 0f)
        {
            return;
        }

        int segmentCount = scoreSegments.Count;
        float safeGap = Mathf.Min(
            segmentGap,
            segmentCount > 1 ? backgroundLocalWidth / (segmentCount - 1) : 0f);
        float segmentWidth = Mathf.Max(
            0.0001f,
            (backgroundLocalWidth - safeGap * (segmentCount - 1)) / segmentCount);
        float totalSegmentProgress = currentProgress * segmentCount;
        Color liveColor = EvaluateScoreColor(currentScore);

        for (int index = 0; index < segmentCount; index++)
        {
            SpriteRenderer segment = scoreSegments[index];
            if (segment == null)
            {
                continue;
            }

            float sectionProgress = Mathf.Clamp01(totalSegmentProgress - index);
            if (segmentIsLocked[index])
            {
                sectionProgress = 1f;
            }

            float visibleWidth = segmentWidth * sectionProgress;
            Transform segmentTransform = segment.transform;
            segmentTransform.localScale = new Vector3(
                visibleWidth / fillSpriteLocalWidth,
                fillLocalScaleY,
                fillLocalScaleZ);

            float segmentLeft = -backgroundLocalWidth * 0.5f +
                index * (segmentWidth + safeGap);
            segmentTransform.localPosition = new Vector3(
                segmentLeft + visibleWidth * 0.5f,
                fillLocalY,
                fillLocalZ);
            segment.enabled = sectionProgress > 0.0001f;
            segment.color = segmentIsLocked[index]
                ? lockedSegmentColors[index]
                : liveColor;
        }

        if (scoreStarTransform != null)
        {
            float halfWidth = Mathf.Max(0f, backgroundLocalWidth * 0.5f - starEdgePadding);
            scoreStarTransform.localPosition = new Vector3(
                Mathf.Lerp(-halfWidth, halfWidth, currentProgress),
                starLocalY,
                starLocalZ);
        }
    }

    private void SetScore(float score)
    {
        currentScore = Mathf.Clamp(score, 0f, 100f);
        Color scoreColor = EvaluateScoreColor(currentScore);
        SetProgress(currentProgress);

        if (scoreStar != null)
        {
            scoreStar.SetScore(currentScore, scoreColor);
        }
    }

    public Color GetScoreColor(float score)
    {
        return EvaluateScoreColor(score);
    }

    public bool IsRedScore(float score)
    {
        return score < redBelowScore;
    }

    public bool IsGreenScore(float score)
    {
        return score >= greenAboveScore;
    }

    private Color EvaluateScoreColor(float score)
    {
        if (score < redBelowScore)
        {
            return needsAdjustmentColor;
        }

        if (score < Mathf.Max(redBelowScore, greenAboveScore))
        {
            return goodColor;
        }

        return excellentColor;
    }

    private void CacheSliderGeometry()
    {
        if (sliderBackground == null || sliderFill == null ||
            sliderBackground.sprite == null || sliderFill.sprite == null)
        {
            return;
        }

        EnsureSolidColorFillSprite();

        // Both sprites are coplanar under the world-space canvas. Explicit
        // sorting prevents the opaque background from covering the fill when
        // viewed head-on by the game camera.
        sliderFill.sortingLayerID = sliderBackground.sortingLayerID;
        sliderFill.sortingOrder = sliderBackground.sortingOrder + 1;

        backgroundLocalWidth = sliderBackground.sprite.bounds.size.x;
        fillSpriteLocalWidth = sliderFill.sprite.bounds.size.x;
        fillLocalY = sliderFill.transform.localPosition.y;
        fillLocalZ = sliderFill.transform.localPosition.z;

        if (!fillTransformCached)
        {
            fillLocalScaleY = sliderFill.transform.localScale.y;
            fillLocalScaleZ = sliderFill.transform.localScale.z;
            fillLocalRotation = sliderFill.transform.localRotation;
            fillTransformCached = true;
        }

        if (scoreSegments.Count == 0)
        {
            BuildScoreSegments(1);
        }

        if (scoreStarTransform != null)
        {
            starLocalY = scoreStarTransform.localPosition.y;
            starLocalZ = scoreStarTransform.localPosition.z;
            scoreStar = scoreStarTransform.GetComponent<DeadBugScoreStarFeedback>();

            Renderer[] starRenderers = scoreStarTransform.GetComponentsInChildren<Renderer>(true);
            foreach (Renderer starRenderer in starRenderers)
            {
                starRenderer.sortingLayerID = sliderBackground.sortingLayerID;
                starRenderer.sortingOrder = sliderFill.sortingOrder + 1;
            }
        }
    }

    // SpriteRenderer.color multiplies the source texture color. Slider1's
    // imported texture is green, so red/orange score tints otherwise remain
    // visibly green. A white sprite with the same size lets the selected score
    // color appear exactly as authored while keeping the rounded bar shape.
    private void EnsureSolidColorFillSprite()
    {
        if (solidColorFillSprite != null || sliderFill == null || sliderFill.sprite == null)
        {
            return;
        }

        originalFillSprite = sliderFill.sprite;
        int width = Mathf.Max(2, Mathf.RoundToInt(originalFillSprite.rect.width));
        int height = Mathf.Max(2, Mathf.RoundToInt(originalFillSprite.rect.height));
        float radius = height * 0.5f;
        Color32[] pixels = new Color32[width * height];

        for (int y = 0; y < height; y++)
        {
            float py = y + 0.5f;
            for (int x = 0; x < width; x++)
            {
                float px = x + 0.5f;
                float centerX = px < radius ? radius :
                    (px > width - radius ? width - radius : px);
                float centerY = height * 0.5f;
                float dx = px - centerX;
                float dy = py - centerY;
                bool inside = dx * dx + dy * dy <= radius * radius;
                pixels[y * width + x] = inside
                    ? new Color32(255, 255, 255, 255)
                    : new Color32(255, 255, 255, 0);
            }
        }

        solidColorFillTexture = new Texture2D(width, height, TextureFormat.RGBA32, false)
        {
            name = "Runtime Solid Score Fill",
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        solidColorFillTexture.SetPixels32(pixels);
        solidColorFillTexture.Apply(false, true);

        solidColorFillSprite = Sprite.Create(
            solidColorFillTexture,
            new Rect(0f, 0f, width, height),
            new Vector2(0.5f, 0.5f),
            originalFillSprite.pixelsPerUnit);
        solidColorFillSprite.name = "Runtime Solid Score Fill";
        sliderFill.sprite = solidColorFillSprite;
    }

    private void BuildScoreSegments(int requestedCount)
    {
        if (sliderFill == null || solidColorFillSprite == null)
        {
            return;
        }

        DestroyGeneratedSegments();
        scoreSegments.Clear();
        lockedSegmentColors.Clear();
        segmentIsLocked.Clear();
        completedSegmentCount = 0;

        int count = Mathf.Max(1, requestedCount);
        scoreSegments.Add(sliderFill);
        lockedSegmentColors.Add(needsAdjustmentColor);
        segmentIsLocked.Add(false);

        for (int index = 1; index < count; index++)
        {
            GameObject segmentObject = new GameObject($"ScoreSegment_{index + 1:00}");
            segmentObject.transform.SetParent(sliderFill.transform.parent, false);
            segmentObject.transform.localRotation = fillLocalRotation;

            SpriteRenderer segment = segmentObject.AddComponent<SpriteRenderer>();
            segment.sprite = solidColorFillSprite;
            segment.sharedMaterial = sliderFill.sharedMaterial;
            segment.sortingLayerID = sliderFill.sortingLayerID;
            segment.sortingOrder = sliderFill.sortingOrder;
            segment.maskInteraction = sliderFill.maskInteraction;

            generatedSegmentObjects.Add(segmentObject);
            scoreSegments.Add(segment);
            lockedSegmentColors.Add(needsAdjustmentColor);
            segmentIsLocked.Add(false);
        }
    }

    private void DestroyGeneratedSegments()
    {
        foreach (GameObject segmentObject in generatedSegmentObjects)
        {
            if (segmentObject != null)
            {
                segmentObject.SetActive(false);
                Destroy(segmentObject);
            }
        }

        generatedSegmentObjects.Clear();
    }

    private void ResolveReferences()
    {
        if (poseScorer == null)
        {
            poseScorer = FindObjectOfType<DeadBugGamingPoseScorer>(true);
        }

        if (coachController == null)
        {
            coachController = FindObjectOfType<CoachActionController>(true);
        }

        if (sliderBackground == null || sliderFill == null)
        {
            SpriteRenderer[] spriteRenderers = FindObjectsOfType<SpriteRenderer>(true);
            foreach (SpriteRenderer spriteRenderer in spriteRenderers)
            {
                if (spriteRenderer == null)
                {
                    continue;
                }

                if (sliderBackground == null && spriteRenderer.gameObject.name == "SliderBG")
                {
                    sliderBackground = spriteRenderer;
                }
                else if (sliderFill == null && spriteRenderer.gameObject.name == "Slider1")
                {
                    sliderFill = spriteRenderer;
                }
            }
        }

        if (scoreStarTransform == null && sliderBackground != null)
        {
            DeadBugScoreStarFeedback foundStar =
                sliderBackground.GetComponentInChildren<DeadBugScoreStarFeedback>(true);
            if (foundStar != null)
            {
                scoreStarTransform = foundStar.transform;
                scoreStar = foundStar;
            }
        }
    }

    private void Subscribe()
    {
        if (poseScorer == null || coachController == null)
        {
            Debug.LogWarning(
                "DeadBugScoreRoadProgressBar needs the pose scorer and coach controller.",
                this);
            return;
        }

        poseScorer.SessionStarted -= HandleSessionStarted;
        poseScorer.LiveScoreChanged -= HandleLiveScoreChanged;
        poseScorer.CheckpointScored -= HandleCheckpointScored;
        poseScorer.SessionScored -= HandleSessionScored;
        coachController.ScoringProgressChanged -= HandleScoringProgressChanged;

        poseScorer.SessionStarted += HandleSessionStarted;
        poseScorer.LiveScoreChanged += HandleLiveScoreChanged;
        poseScorer.CheckpointScored += HandleCheckpointScored;
        poseScorer.SessionScored += HandleSessionScored;
        coachController.ScoringProgressChanged += HandleScoringProgressChanged;
    }

    private void Unsubscribe()
    {
        if (poseScorer != null)
        {
            poseScorer.SessionStarted -= HandleSessionStarted;
            poseScorer.LiveScoreChanged -= HandleLiveScoreChanged;
            poseScorer.CheckpointScored -= HandleCheckpointScored;
            poseScorer.SessionScored -= HandleSessionScored;
        }

        if (coachController != null)
        {
            coachController.ScoringProgressChanged -= HandleScoringProgressChanged;
        }
    }
}
