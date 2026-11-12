using System;
using UnityEngine;

[DefaultExecutionOrder(-1000)]
public sealed class JsonGamingSceneInitializer : MonoBehaviour
{
    public static readonly Type[] DisabledLegacyControllerTypes =
    {
        typeof(DeadBugStartMenuController),
        typeof(DeadBugCountdownController),
        typeof(CoachActionController),
        typeof(DeadBugGamingPoseScorer),
        typeof(RhythmModeController),
        typeof(DeadBugGamingSideFeedback)
    };

    private void Awake()
    {
        CoachMotionPackage package = SpineFlowTrainingSession.CachedMotionPackage;
        string validationError = null;

        // 检查是否有有效的 JSON motion package
        bool hasValidPackage = package != null && package.IsValid(out validationError);

        if (!hasValidPackage)
        {
            // 没有有效的 JSON package，使用传统的 Animator 驱动
            Debug.LogWarning(
                $"testGaming: No valid cached motion package ({validationError}). " +
                "Using legacy AnimatorController-driven mode.",
                this);
            // 不禁用旧控制器，让场景以传统模式运行
            return;
        }

        // 有有效的 JSON package，切换到 JSON 驱动模式
        Debug.Log($"testGaming: Using JSON-driven mode with package '{package.actionId}'", this);

        DisableLegacyControllers();
        DisableSceneObject("RhythmModeRoot");

        Animator coachAnimator = FindAnimator("coach");
        Animator userAnimator = FindAnimator("user");
        if (coachAnimator == null || userAnimator == null)
        {
            Debug.LogError("testGaming could not find both coach and user Humanoid Animators.", this);
            return;
        }

        JsonCoachGamingController jsonController =
            gameObject.AddComponent<JsonCoachGamingController>();
        JsonGamingPoseScorer jsonScorer = gameObject.AddComponent<JsonGamingPoseScorer>();
        JsonGamingUiController jsonUi = gameObject.AddComponent<JsonGamingUiController>();
        jsonController.Initialize(coachAnimator, package);
        jsonScorer.Initialize(jsonController, coachAnimator, userAnimator);
        jsonUi.Initialize(jsonController, jsonScorer, package);

        Debug.Log($"[JsonGamingSceneInitializer] JSON 游戏模式初始化完成，校准状态：{jsonScorer.IsCalibrated}");
    }

    public static void DisableLegacyControllers()
    {
        foreach (DeadBugStartMenuController controller in
            FindObjectsOfType<DeadBugStartMenuController>(true))
        {
            // 移除所有按钮监听器，防止旧控制器的校准检查
            UnityEngine.UI.Button[] buttons = controller.GetComponentsInChildren<UnityEngine.UI.Button>(true);
            foreach (var button in buttons)
            {
                button.onClick.RemoveAllListeners();
            }

            controller.enabled = false;
            Debug.Log("[JsonGamingSceneInitializer] 已禁用 DeadBugStartMenuController 并清除其按钮监听器");
        }
        foreach (DeadBugCountdownController controller in
            FindObjectsOfType<DeadBugCountdownController>(true))
        {
            controller.enabled = false;
        }
        foreach (CoachActionController controller in
            FindObjectsOfType<CoachActionController>(true))
        {
            controller.enabled = false;
        }
        foreach (DeadBugGamingPoseScorer controller in
            FindObjectsOfType<DeadBugGamingPoseScorer>(true))
        {
            controller.enabled = false;
            Debug.Log("[JsonGamingSceneInitializer] 已禁用 DeadBugGamingPoseScorer");
        }
        foreach (RhythmModeController controller in
            FindObjectsOfType<RhythmModeController>(true))
        {
            controller.enabled = false;
        }
        foreach (DeadBugGamingSideFeedback controller in
            FindObjectsOfType<DeadBugGamingSideFeedback>(true))
        {
            controller.enabled = false;
        }
    }

    private static void DisableSceneObject(string objectName)
    {
        foreach (Transform sceneTransform in FindObjectsOfType<Transform>(true))
        {
            if (sceneTransform.name == objectName)
            {
                sceneTransform.gameObject.SetActive(false);
            }
        }
    }

    private static Animator FindAnimator(string objectName)
    {
        foreach (Animator animator in FindObjectsOfType<Animator>(true))
        {
            if (animator != null &&
                string.Equals(animator.gameObject.name, objectName, StringComparison.OrdinalIgnoreCase))
            {
                return animator;
            }
        }
        return null;
    }
}
