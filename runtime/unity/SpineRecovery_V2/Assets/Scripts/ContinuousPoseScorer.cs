using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Continuous pose scoring for gaming mode - compares user pose to coach every frame.
/// No keyframe holding, no stability detection, just real-time scoring.
/// </summary>
public class ContinuousPoseScorer : MonoBehaviour
{
    [Header("References")]
    [Tooltip("The coach animator (plays the exercise animation continuously).")]
    [SerializeField] private Animator coachAnimator;
    [Tooltip("The user animator (driven by IMU/SlimeVR).")]
    [SerializeField] private Animator userAnimator;
    [Tooltip("Optional: UI Text to display current score.")]
    [SerializeField] private Text scoreText;

    [Header("Scoring Configuration")]
    [Tooltip("Bones compared for scoring.")]
    [SerializeField] private HumanBodyBones[] scoringBones = new HumanBodyBones[]
    {
        HumanBodyBones.Hips,
        HumanBodyBones.Spine,
        HumanBodyBones.Chest,
        HumanBodyBones.LeftUpperLeg,
        HumanBodyBones.RightUpperLeg,
        HumanBodyBones.LeftLowerLeg,
        HumanBodyBones.RightLowerLeg,
        HumanBodyBones.LeftUpperArm,
        HumanBodyBones.RightUpperArm,
        HumanBodyBones.LeftLowerArm,
        HumanBodyBones.RightLowerArm
    };

    [Tooltip("Max angle difference (degrees) that maps to score=0.")]
    [SerializeField] private float maxAngleError = 30f;
    [Tooltip("Angle difference (degrees) that maps to score=100 (perfect alignment).")]
    [SerializeField] private float perfectAngleThreshold = 5f;
    [Tooltip("Score leniency multiplier (higher = more forgiving). Useful for difficult poses.")]
    [SerializeField] private float scoreLeniency = 1f;
    [SerializeField] private RobustPoseScoringSettings robustScoring = new RobustPoseScoringSettings();

    [Header("Score Tracking")]
    [Tooltip("Window size for moving average score (number of frames).")]
    [SerializeField] private int movingAverageWindow = 30;
    [Tooltip("Update score text every N frames (to reduce UI overhead).")]
    [SerializeField] private int scoreUpdateInterval = 5;

    // Runtime state
    private Queue<float> recentScores = new Queue<float>();
    private float currentScore = 0f;
    private float averageScore = 0f;
    private int frameCounter = 0;
    private bool isScoring = false;
    private RobustPoseScoringEngine scoringEngine;

    // Public accessors
    public float CurrentScore => currentScore;
    public float AverageScore => averageScore;
    public bool IsScoring => isScoring;

    private void Awake()
    {
        scoringEngine = new RobustPoseScoringEngine(robustScoring);
    }

    private void Update()
    {
        if (!isScoring || coachAnimator == null || userAnimator == null)
        {
            return;
        }

        // Score current pose
        currentScore = ScoreCurrentPose();

        // Update moving average
        recentScores.Enqueue(currentScore);
        if (recentScores.Count > movingAverageWindow)
        {
            recentScores.Dequeue();
        }

        // Calculate average
        float sum = 0f;
        foreach (float score in recentScores)
        {
            sum += score;
        }
        averageScore = recentScores.Count > 0 ? sum / recentScores.Count : 0f;

        // Update UI periodically
        frameCounter++;
        if (frameCounter >= scoreUpdateInterval)
        {
            frameCounter = 0;
            UpdateScoreUI();
        }
    }

    /// <summary>
    /// Start continuous scoring.
    /// </summary>
    public void StartScoring()
    {
        if (coachAnimator == null || userAnimator == null)
        {
            Debug.LogError("ContinuousPoseScorer: Missing coach or user animator!");
            return;
        }

        isScoring = true;
        recentScores.Clear();
        currentScore = 0f;
        averageScore = 0f;
        frameCounter = 0;
        scoringEngine?.Reset();

        Debug.Log("ContinuousPoseScorer: Started continuous scoring.");
    }

    /// <summary>
    /// Stop continuous scoring.
    /// </summary>
    public void StopScoring()
    {
        isScoring = false;
        Debug.Log($"ContinuousPoseScorer: Stopped. Final average score: {averageScore:F1}");
    }

    /// <summary>
    /// Reset all scores.
    /// </summary>
    public void ResetScores()
    {
        recentScores.Clear();
        currentScore = 0f;
        averageScore = 0f;
        frameCounter = 0;
        UpdateScoreUI();
    }

    /// <summary>
    /// Compares user's current pose to coach's current pose. Returns a score 0-100.
    /// </summary>
    private float ScoreCurrentPose()
    {
        if (userAnimator == null || coachAnimator == null)
        {
            return 0f;
        }

        if (scoringEngine == null)
        {
            scoringEngine = new RobustPoseScoringEngine(robustScoring);
        }

        return scoringEngine.Evaluate(
            userAnimator,
            coachAnimator,
            scoringBones,
            perfectAngleThreshold,
            maxAngleError,
            scoreLeniency,
            deltaTime: Time.deltaTime).score;
    }

    private void UpdateScoreUI()
    {
        if (scoreText != null)
        {
            scoreText.text = $"当前: {currentScore:F0}\n平均: {averageScore:F0}";
        }
    }

    private void OnDestroy()
    {
        recentScores.Clear();
    }

    // Debug visualization
    private void OnGUI()
    {
        if (!isScoring || !Application.isPlaying)
        {
            return;
        }

        // Optional: draw score graph or debug info
        GUILayout.BeginArea(new Rect(10, 10, 300, 100));
        GUILayout.Label($"Current Score: {currentScore:F1}");
        GUILayout.Label($"Average Score (last {recentScores.Count} frames): {averageScore:F1}");
        GUILayout.Label($"Max Angle Error: {maxAngleError * scoreLeniency:F1}°");
        GUILayout.EndArea();
    }
}
