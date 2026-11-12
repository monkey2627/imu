using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// 显示身体部位对齐提示图片
/// </summary>
public class BodyPartFeedbackUI : MonoBehaviour
{
    [Header("身体部位提示图片")]
    [SerializeField] private Sprite leftArmSprite;
    [SerializeField] private Sprite rightArmSprite;
    [SerializeField] private Sprite leftLegSprite;
    [SerializeField] private Sprite rightLegSprite;
    [SerializeField] private Sprite torsoSprite;

    [Header("UI组件")]
    [SerializeField] private Image feedbackImage;
    [SerializeField] private CanvasGroup canvasGroup;

    private BodyPart visiblePart = BodyPart.None;

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

        if (feedbackImage == null)
        {
            feedbackImage = GetComponent<Image>();
        }

        // 初始隐藏
        Hide();
    }

    /// <summary>
    /// 显示指定身体部位的提示图片
    /// </summary>
    public void ShowBodyPart(BodyPart part)
    {
        if (feedbackImage == null)
        {
            Debug.LogError("BodyPartFeedbackUI: feedbackImage 未配置！");
            return;
        }

        Sprite sprite = GetSpriteForBodyPart(part);
        if (sprite != null)
        {
            if (visiblePart == part && canvasGroup.alpha > 0.99f && feedbackImage.enabled)
            {
                return;
            }

            feedbackImage.sprite = sprite;
            feedbackImage.enabled = true;
            canvasGroup.alpha = 1f;
            visiblePart = part;
        }
        else
        {
            Debug.LogWarning($"BodyPartFeedbackUI: {part} 的图片未配置！");
            Hide();
        }
    }

    /// <summary>
    /// 隐藏提示图片
    /// </summary>
    public void Hide()
    {
        if (canvasGroup != null)
        {
            canvasGroup.alpha = 0f;
        }

        visiblePart = BodyPart.None;
    }

    private Sprite GetSpriteForBodyPart(BodyPart part)
    {
        switch (part)
        {
            case BodyPart.LeftArm:
                return leftArmSprite;
            case BodyPart.RightArm:
                return rightArmSprite;
            case BodyPart.LeftLeg:
                return leftLegSprite;
            case BodyPart.RightLeg:
                return rightLegSprite;
            case BodyPart.Torso:
                return torsoSprite;
            default:
                return null;
        }
    }
}

/// <summary>
/// 身体部位枚举
/// </summary>
public enum BodyPart
{
    None,
    LeftArm,
    RightArm,
    LeftLeg,
    RightLeg,
    Torso
}
