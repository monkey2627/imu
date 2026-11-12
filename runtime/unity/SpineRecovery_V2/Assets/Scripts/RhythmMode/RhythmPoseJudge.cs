using UnityEngine;

public sealed class RhythmPoseJudge : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private DeadBugGamingPoseScorer poseScorer;

    [Header("Waist Evaluation")]
    [Min(1f)] [SerializeField] private float waistPerfectAngle = 10f;
    [Min(5f)] [SerializeField] private float waistZeroScoreAngle = 48f;

    private AnatomicalLimbScores limbScores;
    private float waistScore;
    private bool hasWaistTracking;
    private bool awaitingSessionScores;
    GameTrainingAssessment assessment;
    public RhythmPoseDiagnostic LastDiagnostic { get; private set; }

    string Difficulty => poseScorer!=null?poseScorer.SelectedDifficultyKey:"standard";
    public float SuccessThreshold => GameDifficultyPolicy.HitThreshold(Difficulty);
    public Animator CoachAnimator => poseScorer != null ? poseScorer.CoachAnimator : null;

    private void OnEnable()
    {
        ResolveReferences();
        Subscribe();
    }

    private void OnDisable()
    {
        Unsubscribe();
    }

    private void LateUpdate()
    {
        UpdateWaistScore();
    }

    public bool TryHit(RhythmBodyTarget target, out float poseScore)
    {
        if(assessment==null)assessment=FindObjectOfType<GameTrainingAssessment>();
        return TryHit(target,assessment!=null?assessment.CaptureHitReference():null,out poseScore);
    }

    public RhythmMovementRange CaptureTarget(string state)
    {
        if(poseScorer==null)return null;
        var reference=new RhythmMovementRange();reference.Begin(poseScorer,state);return reference;
    }

    public bool TryHit(RhythmBodyTarget target,RhythmMovementRange reference,out float poseScore)
    {
        poseScore=0;
        if(poseScorer==null){ResolveReferences();Subscribe();}
        bool sessionActive=poseScorer!=null&&poseScorer.IsSessionActive;
        if(sessionActive&&!awaitingSessionScores)limbScores=poseScorer.LastLimbScores;
        bool paused=poseScorer==null||poseScorer.IsScoringPaused;
        bool fresh=poseScorer!=null&&poseScorer.HasFreshInput;
        int validSegments=ValidSegmentCount(target);
        var diagnostic=new RhythmPoseDiagnostic{target=target.ToString(),difficulty=Difficulty,
            sessionActive=sessionActive,paused=paused,freshInput=fresh,validSegmentCount=validSegments,
            tracked=sessionActive&&!paused&&fresh&&!awaitingSessionScores&&validSegments>0,poseThreshold=SuccessThreshold,
            requiredExtension=GameDifficultyPolicy.MinimumExtension(Difficulty,target),
            progress=-1,endpointQuality=-1,thighAngleDegrees=-1,shinAngleDegrees=-1};
        diagnostic.anatomicalScore=RawScore(target);
        LastDiagnostic=diagnostic;
        if(!diagnostic.tracked){diagnostic.reason=!sessionActive?"session_inactive":paused?"scoring_paused":!fresh?"stale_input":awaitingSessionScores?"waiting_for_session_scores":"limb_not_tracked";return false;}
        if(reference==null)
        {
            // Waist/legacy presentations retain their own pose measurement.
            if(target!=RhythmBodyTarget.Waist){diagnostic.reason="missing_movement_reference";return false;}
            poseScore=GetScore(target);diagnostic.poseScore=poseScore;
            diagnostic.reason=poseScore>=SuccessThreshold?"hit":"pose_quality_below_threshold";
            return poseScore>=SuccessThreshold;
        }
        // The note owns its extension pose throughout the arrival window. The
        // live coach may already be returning or starting the opposite pair.
        diagnostic.postureMatched=reference.HasMatchingPosture(poseScorer);
        diagnostic.endpointAvailable=reference.TryEndpoint(poseScorer,target,out float progress,out poseScore,
            out string failureReason,out float thighAngle,out float shinAngle);
        diagnostic.progress=progress;diagnostic.poseScore=poseScore;diagnostic.endpointQuality=poseScore;
        diagnostic.thighAngleDegrees=thighAngle;diagnostic.shinAngleDegrees=shinAngle;
        if(!diagnostic.endpointAvailable){diagnostic.reason=failureReason;return false;}
        if(progress<diagnostic.requiredExtension){diagnostic.reason="extension_below_threshold";return false;}
        if(poseScore<SuccessThreshold){diagnostic.reason="pose_quality_below_threshold";return false;}
        diagnostic.reason="hit";return true;
    }

    int ValidSegmentCount(RhythmBodyTarget target)
    {
        switch(target)
        {
            case RhythmBodyTarget.LeftHand:return limbScores.leftArm.validSegmentCount;
            case RhythmBodyTarget.RightHand:return limbScores.rightArm.validSegmentCount;
            case RhythmBodyTarget.LeftFoot:return limbScores.leftLeg.validSegmentCount;
            case RhythmBodyTarget.RightFoot:return limbScores.rightLeg.validSegmentCount;
            default:return hasWaistTracking?1:0;
        }
    }

    float RawScore(RhythmBodyTarget target)
    {
        switch(target)
        {
            case RhythmBodyTarget.LeftHand:return limbScores.leftArm.rawScore;
            case RhythmBodyTarget.RightHand:return limbScores.rightArm.rawScore;
            case RhythmBodyTarget.LeftFoot:return limbScores.leftLeg.rawScore;
            case RhythmBodyTarget.RightFoot:return limbScores.rightLeg.rawScore;
            default:return waistScore;
        }
    }

    public float GetScore(RhythmBodyTarget target)
    {
        if(poseScorer==null||!poseScorer.IsSessionActive||poseScorer.IsScoringPaused||!poseScorer.HasFreshInput||!poseScorer.HasMatchingFloorPosture)return 0;
        switch (target)
        {
            case RhythmBodyTarget.LeftHand: return limbScores.leftArm.score;
            case RhythmBodyTarget.RightHand: return limbScores.rightArm.score;
            case RhythmBodyTarget.LeftFoot: return limbScores.leftLeg.score;
            case RhythmBodyTarget.RightFoot: return limbScores.rightLeg.score;
            default: return waistScore;
        }
    }

    private bool HasTracking(RhythmBodyTarget target)
    {
        // Pending notes check posture against their captured target, even if the
        // coach has already moved into the opposite pair or the outro.
        if(poseScorer==null||!poseScorer.IsSessionActive||poseScorer.IsScoringPaused||!poseScorer.HasFreshInput)return false;
        switch (target)
        {
            case RhythmBodyTarget.LeftHand: return limbScores.leftArm.validSegmentCount > 0;
            case RhythmBodyTarget.RightHand: return limbScores.rightArm.validSegmentCount > 0;
            case RhythmBodyTarget.LeftFoot: return limbScores.leftLeg.validSegmentCount > 0;
            case RhythmBodyTarget.RightFoot: return limbScores.rightLeg.validSegmentCount > 0;
            default: return hasWaistTracking;
        }
    }

    private void UpdateWaistScore()
    {
        hasWaistTracking = false;
        waistScore = 0f;
        if (poseScorer == null || poseScorer.CoachAnimator == null || poseScorer.UserAnimator == null)
        {
            return;
        }

        Transform coachHips = poseScorer.CoachAnimator.GetBoneTransform(HumanBodyBones.Hips);
        Transform coachChest = GetChest(poseScorer.CoachAnimator);
        Transform userHips = poseScorer.UserAnimator.GetBoneTransform(HumanBodyBones.Hips);
        Transform userChest = GetChest(poseScorer.UserAnimator);
        if (coachHips == null || coachChest == null || userHips == null || userChest == null)
        {
            return;
        }

        Quaternion coachRelative = Quaternion.Inverse(coachHips.rotation) * coachChest.rotation;
        Quaternion userRelative = Quaternion.Inverse(userHips.rotation) * userChest.rotation;
        float error = Quaternion.Angle(coachRelative, userRelative);
        float normalized = Mathf.InverseLerp(waistPerfectAngle, waistZeroScoreAngle, error);
        waistScore = (1f - normalized * normalized * (3f - 2f * normalized)) * 100f;
        hasWaistTracking = true;
    }

    private static Transform GetChest(Animator animator)
    {
        return animator.GetBoneTransform(HumanBodyBones.UpperChest) ??
            animator.GetBoneTransform(HumanBodyBones.Chest) ??
            animator.GetBoneTransform(HumanBodyBones.Spine);
    }

    private void HandleLimbScoresChanged(AnatomicalLimbScores scores)
    {
        limbScores = scores;
        if(poseScorer!=null&&poseScorer.IsSessionActive)awaitingSessionScores=false;
    }

    private void HandleSessionStarted()
    {
        limbScores=default;
        awaitingSessionScores=true;
    }

    private void ResolveReferences()
    {
        if (poseScorer == null)
        {
            poseScorer = FindObjectOfType<DeadBugGamingPoseScorer>(true);
        }
    }

    private void Subscribe()
    {
        if (poseScorer == null)
        {
            return;
        }

        poseScorer.LimbScoresChanged -= HandleLimbScoresChanged;
        poseScorer.LimbScoresChanged += HandleLimbScoresChanged;
        poseScorer.SessionStarted -= HandleSessionStarted;
        poseScorer.SessionStarted += HandleSessionStarted;
        if(poseScorer.IsSessionActive)
        {
            limbScores=poseScorer.LastLimbScores;
            awaitingSessionScores=false;
        }
    }

    private void Unsubscribe()
    {
        if (poseScorer != null)
        {
            poseScorer.LimbScoresChanged -= HandleLimbScoresChanged;
            poseScorer.SessionStarted -= HandleSessionStarted;
        }
    }
}
