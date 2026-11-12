using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

[RequireComponent(typeof(Slider))]
public class PlaybackProgressSlider : MonoBehaviour, IPointerDownHandler, IPointerUpHandler, IBeginDragHandler, IDragHandler, IEndDragHandler
{
    private const string SnapHitAreaName = "Progress Snap Hit Area";

    [SerializeField] private MotionPlaybackController playbackController;
    [SerializeField] private ReplayPlayer replayPlayer;
    [SerializeField] private float snapHitHeight = 140f;

    private Slider slider;
    private RectTransform rectTransform;
    private bool isDragging;
    private bool seekActive;

    public void Configure(ReplayPlayer player)
    {
        replayPlayer = player;
        playbackController = null;
    }

    private void Awake()
    {
        slider = GetComponent<Slider>();
        rectTransform = GetComponent<RectTransform>();
        slider.minValue = 0f;
        slider.maxValue = 1f;
        slider.wholeNumbers = false;
        slider.onValueChanged.AddListener(OnSliderValueChanged);

        EnsureSnapHitArea();
    }

    private void OnDestroy()
    {
        if (slider != null)
        {
            slider.onValueChanged.RemoveListener(OnSliderValueChanged);
        }
    }

    private void Update()
    {
        if (isDragging)
        {
            return;
        }

        if (replayPlayer != null && replayPlayer.HasReplay)
        {
            slider.SetValueWithoutNotify(replayPlayer.NormalizedProgress);
            return;
        }

        if (playbackController == null || !playbackController.HasClip)
        {
            return;
        }

        slider.SetValueWithoutNotify(playbackController.NormalizedProgress);
    }

    public void OnPointerDown(PointerEventData eventData)
    {
        isDragging = true;
        BeginReplaySeek();
        UpdateSliderFromPointer(eventData);
    }

    public void OnPointerUp(PointerEventData eventData)
    {
        UpdateSliderFromPointer(eventData);
        isDragging = false;

        if (replayPlayer != null)
        {
            replayPlayer.SeekNormalized(slider.value);
            EndReplaySeek();
        }
        else if (playbackController != null)
        {
            playbackController.SeekNormalized(slider.value);
        }
    }

    public void OnBeginDrag(PointerEventData eventData)
    {
        isDragging = true;
        BeginReplaySeek();
        UpdateSliderFromPointer(eventData);
    }

    public void OnDrag(PointerEventData eventData)
    {
        UpdateSliderFromPointer(eventData);
    }

    public void OnEndDrag(PointerEventData eventData)
    {
        UpdateSliderFromPointer(eventData);
        isDragging = false;
        EndReplaySeek();
    }

    private void OnSliderValueChanged(float value)
    {
        if (replayPlayer != null)
        {
            replayPlayer.SeekNormalized(value);
        }
        else
        {
            playbackController?.SeekNormalized(value);
        }
    }

    private void BeginReplaySeek()
    {
        if (!seekActive && replayPlayer != null)
        {
            seekActive = true;
            replayPlayer.BeginSeek();
        }
    }

    private void EndReplaySeek()
    {
        if (seekActive && replayPlayer != null)
        {
            seekActive = false;
            replayPlayer.EndSeek();
        }
    }

    private void UpdateSliderFromPointer(PointerEventData eventData)
    {
        if (eventData == null || rectTransform == null || slider == null)
        {
            return;
        }

        Camera eventCamera = eventData.pressEventCamera != null
            ? eventData.pressEventCamera
            : eventData.enterEventCamera;

        if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(
                rectTransform,
                eventData.position,
                eventCamera,
                out Vector2 localPoint))
        {
            return;
        }

        Rect rect = rectTransform.rect;
        float normalized = Mathf.InverseLerp(rect.xMin, rect.xMax, localPoint.x);
        slider.value = Mathf.Clamp01(normalized);
    }

    private void EnsureSnapHitArea()
    {
        if (rectTransform == null || snapHitHeight <= 0f)
        {
            return;
        }

        Transform existing = transform.Find(SnapHitAreaName);
        RectTransform hitRect = existing as RectTransform;
        if (hitRect == null)
        {
            GameObject hitArea = new GameObject(SnapHitAreaName, typeof(RectTransform), typeof(Image));
            hitRect = hitArea.GetComponent<RectTransform>();
            hitRect.SetParent(transform, false);
        }

        hitRect.SetAsFirstSibling();
        hitRect.anchorMin = new Vector2(0f, 0.5f);
        hitRect.anchorMax = new Vector2(1f, 0.5f);
        hitRect.anchoredPosition = Vector2.zero;
        hitRect.sizeDelta = new Vector2(0f, snapHitHeight);
        hitRect.pivot = new Vector2(0.5f, 0.5f);

        Image image = hitRect.GetComponent<Image>();
        image.color = new Color(1f, 1f, 1f, 0f);
        image.raycastTarget = true;
    }
}
