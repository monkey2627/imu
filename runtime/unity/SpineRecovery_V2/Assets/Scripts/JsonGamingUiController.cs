using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public sealed class JsonGamingUiController : MonoBehaviour
{
    private JsonCoachGamingController controller;
    private JsonGamingPoseScorer scorer;
    private CoachMotionPackage package;
    private GameObject startMenuCanvas;
    private GameObject gamingCanvas;
    private GameObject endMenuCanvas;
    private TMP_Text actionNameText;
    private TMP_Text progressText;
    private Button startButton;
    private bool started;
    private string selectedLevel = "standard";

    public void Initialize(
        JsonCoachGamingController gamingController,
        JsonGamingPoseScorer poseScorer,
        CoachMotionPackage motionPackage)
    {
        controller = gamingController;
        scorer = poseScorer;
        package = motionPackage;
        startMenuCanvas = FindSceneObject("StartMenuCanvas");
        gamingCanvas = FindSceneObject("GamingCanvas");
        endMenuCanvas = FindSceneObject("EndMenuCanvas");
        actionNameText = FindText("ActionName");
        progressText = FindText("Progress");

        BindDifficultyButton("Simple", "simple");
        BindDifficultyButton("Standard", "standard");
        BindDifficultyButton("Hard", "hard");

        // 查找并绑定开始按钮（可能有多个不同名称）
        startButton = FindButton("StartTraining");
        if (startButton == null)
        {
            startButton = FindButton("Start");
        }
        if (startButton == null)
        {
            // 查找包含 "Start" 或 "开始" 的按钮
            foreach (UnityEngine.UI.Button button in FindObjectsOfType<UnityEngine.UI.Button>(true))
            {
                string buttonName = button.gameObject.name;
                if (buttonName.Contains("Start") || buttonName.Contains("开始") || buttonName.Contains("Training"))
                {
                    startButton = button;
                    Debug.Log($"[JsonGamingUiController] 找到开始按钮：{buttonName}");
                    break;
                }
            }
        }

        if (startButton != null)
        {
            ReplaceListeners(startButton, StartGame);
            Debug.Log($"[JsonGamingUiController] 按钮 '{startButton.name}' 已绑定到 StartGame");
        }
        else
        {
            Debug.LogWarning("[JsonGamingUiController] 未找到任何开始按钮");
        }

        if (actionNameText != null)
        {
            actionNameText.gameObject.SetActive(true);
            actionNameText.text = string.IsNullOrWhiteSpace(package.displayName)
                ? package.actionId
                : package.displayName;
        }
        if (progressText != null)
        {
            progressText.gameObject.SetActive(true);
            progressText.text = $"0 / {controller.ScoringUnitCount}";
        }

        startMenuCanvas?.SetActive(true);
        gamingCanvas?.SetActive(false);
        endMenuCanvas?.SetActive(false);
        controller.ScoringCheckpointStarted += HandleCheckpointStarted;
        controller.ScoringProgressChanged += HandleProgressChanged;
        scorer.SessionScored += HandleSessionScored;
    }

    public void StartGame()
    {
        if (started)
        {
            return;
        }

        // 检查校准状态
        if (scorer != null && !scorer.IsCalibrated)
        {
            Debug.LogWarning("[JsonGamingUiController] 用户尚未校准，开始校准流程");
            scorer.BeginCalibration();
            // 校准完成后会自动继续（编辑器模式下立即完成）
        }

        started = true;
        SpineFlowTrainingSession.BeginGameTraining(selectedLevel);
        startMenuCanvas?.SetActive(false);
        endMenuCanvas?.SetActive(false);
        gamingCanvas?.SetActive(true);

        Debug.Log($"[JsonGamingUiController] 开始游戏，难度：{selectedLevel}");
        controller.PlayAction();
    }

    private void BindDifficultyButton(string objectName, string level)
    {
        Button button = FindButton(objectName);
        if (button != null)
        {
            ReplaceListeners(button, () => selectedLevel = level);
        }
    }

    private void HandleCheckpointStarted(JsonGamingCheckpointInfo info)
    {
        if (actionNameText != null)
        {
            string actionName = string.IsNullOrWhiteSpace(package.displayName)
                ? package.actionId
                : package.displayName;
            actionNameText.text = string.IsNullOrWhiteSpace(info.label)
                ? actionName
                : $"{actionName} - {info.label}";
        }
    }

    private void HandleProgressChanged(float progress)
    {
        if (progressText != null)
        {
            int completed = Mathf.RoundToInt(progress * controller.ScoringUnitCount);
            progressText.text = $"{completed} / {controller.ScoringUnitCount}";
        }
    }

    private void HandleSessionScored(
        float average,
        IReadOnlyList<DeadBugCheckpointScore> scores)
    {
        if (progressText != null)
        {
            progressText.text = $"{Mathf.RoundToInt(average)}";
        }
        gamingCanvas?.SetActive(false);
        endMenuCanvas?.SetActive(true);
    }

    private static void ReplaceListeners(Button button, UnityEngine.Events.UnityAction listener)
    {
        button.onClick = new Button.ButtonClickedEvent();
        button.onClick.AddListener(listener);
    }

    private static GameObject FindSceneObject(string objectName)
    {
        foreach (Transform sceneTransform in FindObjectsOfType<Transform>(true))
        {
            if (sceneTransform.name == objectName)
            {
                return sceneTransform.gameObject;
            }
        }
        return null;
    }

    private static Button FindButton(string objectName)
    {
        GameObject sceneObject = FindSceneObject(objectName);
        return sceneObject != null ? sceneObject.GetComponent<Button>() : null;
    }

    private static TMP_Text FindText(string objectName)
    {
        GameObject sceneObject = FindSceneObject(objectName);
        return sceneObject != null ? sceneObject.GetComponentInChildren<TMP_Text>(true) : null;
    }

    private void OnDestroy()
    {
        if (controller != null)
        {
            controller.ScoringCheckpointStarted -= HandleCheckpointStarted;
            controller.ScoringProgressChanged -= HandleProgressChanged;
        }
        if (scorer != null)
        {
            scorer.SessionScored -= HandleSessionScored;
        }
    }
}
