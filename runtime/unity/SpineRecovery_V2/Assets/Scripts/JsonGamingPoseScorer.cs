using System;
using System.Collections.Generic;
using UnityEngine;

public sealed class JsonGamingPoseScorer : MonoBehaviour
{
    public event Action<float> LiveScoreChanged;
    public event Action<DeadBugCheckpointScore> CheckpointScored;
    public event Action<float, IReadOnlyList<DeadBugCheckpointScore>> SessionScored;

    private static readonly HumanBodyBones[] EvidenceBones =
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

    private readonly List<float> checkpointFrameScores = new List<float>();
    private readonly List<DeadBugCheckpointScore> sessionScores = new List<DeadBugCheckpointScore>();
    private readonly AnimatorMotionEvidence motionEvidence = new AnimatorMotionEvidence();

    private JsonCoachGamingController controller;
    private Animator coachAnimator;
    private Animator userAnimator;
    private AnatomicalMotionScoringEngine scoringEngine;
    private JsonGamingCheckpointInfo activeCheckpoint;
    private bool sessionActive;
    private bool checkpointActive;
    private bool calibrated;

    public IReadOnlyList<DeadBugCheckpointScore> SessionScores => sessionScores;
    public float LastSessionAverage { get; private set; }
    public bool IsCalibrated => calibrated;

    public void Initialize(
        JsonCoachGamingController gamingController,
        Animator coach,
        Animator user)
    {
        Unsubscribe();
        controller = gamingController;
        coachAnimator = coach;
        userAnimator = user;
        scoringEngine = new AnatomicalMotionScoringEngine(new AnatomicalMotionScoringSettings());
        scoringEngine.ClearCalibration();

#if UNITY_EDITOR
        // 编辑器模式下自动校准，跳过站立检测
        calibrated = true;
        Debug.Log("[JsonGamingPoseScorer] 编辑器模式：自动通过校准");
#else
        calibrated = false;
#endif

        Subscribe();
    }

    public void BeginCalibration()
    {
        // TODO: 实现真实的站立检测校准
        // 目前在编辑器中自动通过
#if UNITY_EDITOR
        calibrated = true;
        Debug.Log("[JsonGamingPoseScorer] 编辑器模式：校准完成");
#else
        Debug.LogWarning("[JsonGamingPoseScorer] 校准功能尚未实现");
        calibrated = true; // 临时：自动通过
#endif
    }

    private void LateUpdate()
    {
        if (!sessionActive || !checkpointActive || coachAnimator == null || userAnimator == null)
        {
            return;
        }

        motionEvidence.Sample();
        AnatomicalPoseScore result = scoringEngine.Evaluate(
            userAnimator,
            coachAnimator,
            Time.deltaTime);
        checkpointFrameScores.Add(result.rawScore);
        LiveScoreChanged?.Invoke(result.score);
    }

    private void HandleSessionStarted()
    {
        sessionScores.Clear();
        checkpointFrameScores.Clear();
        LastSessionAverage = 0f;
        checkpointActive = false;
        scoringEngine.ResetSession();
        sessionActive = coachAnimator != null && userAnimator != null;
        if (!sessionActive)
        {
            Debug.LogError("JsonGamingPoseScorer needs both coach and user Animators.", this);
            return;
        }

        motionEvidence.Begin(coachAnimator, userAnimator, EvidenceBones);
    }

    private void HandleCheckpointStarted(JsonGamingCheckpointInfo info)
    {
        if (!sessionActive)
        {
            return;
        }

        activeCheckpoint = info;
        checkpointFrameScores.Clear();
        checkpointActive = true;
    }

    private void HandleCheckpointEnded(JsonGamingCheckpointInfo info)
    {
        if (!sessionActive || !checkpointActive || activeCheckpoint.index != info.index)
        {
            return;
        }

        checkpointActive = false;
        float best = 0f;
        foreach (float score in checkpointFrameScores)
        {
            best = Mathf.Max(best, score);
        }

        var checkpointScore = new DeadBugCheckpointScore
        {
            stateName = info.label,
            loopIndex = info.setIndex,
            loopCount = info.setCount,
            pairIndex = info.index,
            isUp = true,
            averageScore = RobustPoseScoringEngine.CalculateRobustAverage(
                checkpointFrameScores,
                0.1f),
            bestScore = best,
            holdFrameCount = checkpointFrameScores.Count,
            frameScores = new List<float>(checkpointFrameScores)
        };
        sessionScores.Add(checkpointScore);
        CheckpointScored?.Invoke(checkpointScore);
    }

    private void HandleSessionCompleted()
    {
        if (!sessionActive)
        {
            return;
        }

        checkpointActive = false;
        motionEvidence.Stop();
        sessionActive = false;

        float total = 0f;
        foreach (DeadBugCheckpointScore score in sessionScores)
        {
            total += score.averageScore;
        }
        LastSessionAverage = sessionScores.Count > 0 ? total / sessionScores.Count : 0f;

        SpineFlowTrainingSession.CompleteGameTraining(
            sessionScores,
            motionEvidence.MaximumUserMotionDegrees,
            motionEvidence.MaximumCoachMotionDegrees);
        SessionScored?.Invoke(LastSessionAverage, sessionScores);
    }

    private void Subscribe()
    {
        if (controller == null)
        {
            return;
        }

        controller.ScoringSessionStarted += HandleSessionStarted;
        controller.ScoringCheckpointStarted += HandleCheckpointStarted;
        controller.ScoringCheckpointEnded += HandleCheckpointEnded;
        controller.ScoringSessionCompleted += HandleSessionCompleted;
    }

    private void Unsubscribe()
    {
        if (controller == null)
        {
            return;
        }

        controller.ScoringSessionStarted -= HandleSessionStarted;
        controller.ScoringCheckpointStarted -= HandleCheckpointStarted;
        controller.ScoringCheckpointEnded -= HandleCheckpointEnded;
        controller.ScoringSessionCompleted -= HandleSessionCompleted;
    }

    private void OnDestroy()
    {
        Unsubscribe();
    }
}
