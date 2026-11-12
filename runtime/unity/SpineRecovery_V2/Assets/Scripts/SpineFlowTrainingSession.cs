using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using UnityEngine.SceneManagement;

public struct SpineFlowTrainingVolume
{
    public int sets;
    public int repetitionsPerSet;

    public SpineFlowTrainingVolume(int sets, int repetitionsPerSet)
    {
        this.sets = Mathf.Max(1, sets);
        this.repetitionsPerSet = Mathf.Max(1, repetitionsPerSet);
    }

    public int TotalRepetitions
    {
        get
        {
            long total = (long)sets * repetitionsPerSet;
            return total > int.MaxValue ? int.MaxValue : (int)total;
        }
    }
}

// Persists the phone-launched SpineFlow session across a supported coaching
// scene and its gaming scene, aggregates their real measurements, and returns
// one result to the phone when the user leaves the gaming completion panel.
public sealed class SpineFlowTrainingSession : MonoBehaviour
{
    private const int ProtocolVersion = 1;
    private const string PayloadExtra = "spineflow_payload_json";
    private const string ProtocolVersionExtra = "spineflow_protocol_version";
    private const string MobilePackage = "com.metaspine.mobile";
    private const string ResultAction = "com.metaspine.action.AR_SESSION_RESULT";
    private const string StartMessage = "START_SESSION";
    private const string ResultMessage = "SESSION_RESULT";
    private const string MeasuredOrigin = "ar_pose_measurement_v2";
    private const string ScoreSummaryOrigin = "ar_score_summary_v1";
    private const string MeasuredDetail = "MEASURED_AR_RESULT";
    private const string ScoreSummaryDetail = "MEASURED_AR_SCORE_SUMMARY";
    private const string MissingDetail = "AR_DETAILS_UNAVAILABLE";
    private const string DeadBugSceneName = "DeadBug";
    private const string DeadBugPracticeSceneName = "DeadBugPractice";
    private const string MaleDeadBugSceneName = "maleDeadBug 1";
    private const string MaleDeadBugPracticeSceneName = "maleDeadBugPractice";
    private const string BirdDogPracticeSceneName = "BirdDogPractice";
    private const string MaleBirdDogPracticeSceneName = "maleBirdDogPractice";
    private const string UnifiedPracticeSceneName = "testPractice";
    private const float MinimumAcceptedMotionDegrees = 8f;
    private const int MinimumAcceptedSamples = 10;
    private const float CoreActionWeight = 1f;
    private const float ReturnTransitionWeight = 0.2f;
    private const float BoundaryPoseWeight = 0.1f;
    private const float CoachModeWeight = 0.7f;
    private const float GameModeWeight = 0.3f;
    private const string ScoreMappingVersion = "formal-core-v3";

    [Serializable]
    private sealed class CallbackPayload
    {
        public string packageName;
        public string action;
    }

    [Serializable]
    private sealed class PlannedAction
    {
        public string actionId;
        public string actionName;
        public string arTaskType;
        public string level;
        public int sets;
        public int reps;
        public string motionPackageUri;
    }

    [Serializable]
    private sealed class TrainingPlanPayload
    {
        public PlannedAction[] recommendedActions;
    }

    [Serializable]
    private sealed class UserProfilePayload
    {
        public string gender;
    }

    [Serializable]
    private sealed class StartPayload
    {
        public int protocolVersion;
        public string messageType;
        public string sessionId;
        public string userId;
        public string requestToken;
        public string targetScene;
        public UserProfilePayload userProfile;
        public CallbackPayload callback;
        public TrainingPlanPayload trainingPlan;
    }

    [Serializable]
    private sealed class MetricPayload
    {
        public int speed;
        public int smoothness;
        public int stability;
        public int symmetry;
        public int holdTime;
        public int rangeOfMotion;
    }

    [Serializable]
    private sealed class WarningPayload
    {
        public string type;
        public string message;
        public int count;
    }

    [Serializable]
    private sealed class ActionResultPayload
    {
        public string actionId;
        public string actionName;
        public int overallScore;
        public int rawOverallScore;
        public string trainingMode;
        public int plannedRepetitions;
        public int completedRepetitions;
        public int durationSeconds;
        public int averageScore;
        public int rawAverageScore;
        public int bestScore;
        public int rawBestScore;
        public int scoreSampleCount;
        public string scoreMappingVersion;
        public float minimumUserMotionDegrees;
        public float minimumCoachMotionDegrees;
        public MetricPayload metrics;
        public bool metricsAvailable;
        public WarningPayload[] warnings;
        public bool passed;
        public bool cheatingDetected;
    }

    [Serializable]
    private sealed class GameResultPayload
    {
        public string gameId;
        public string gameName;
        public int score;
        public int averageScore;
        public int rawAverageScore;
        public int bestScore;
        public int rawBestScore;
        public int scoreSampleCount;
        public int completedRepetitions;
        public string scoreMappingVersion;
        public string level;
        public int durationSeconds;
    }

    [Serializable]
    private sealed class ResultPayload
    {
        public string schemaVersion = "0.1";
        public int protocolVersion = ProtocolVersion;
        public string messageType = ResultMessage;
        public string sessionId;
        public string userId;
        public string requestToken;
        public string status;
        public string endReason;
        public string createdAt;
        public string sourceApp = "ar_app";
        public string detailStatus;
        public string dataOrigin;
        public bool simulated;
        public bool completed;
        public int durationSeconds;
        public int overallScore;
        public int rawOverallScore;
        public string scoreMappingVersion;
        public ActionResultPayload[] actionResults;
        public GameResultPayload[] gameResults;
        public string[] abnormalEvents;
    }

    private sealed class ModeMeasurement
    {
        public int averageScore;
        public int rawAverageScore;
        public int bestScore;
        public int rawBestScore;
        public int scoreSampleCount;
        public int completedRepetitions;
        public float userMotionDegrees;
        public float coachMotionDegrees;
        public int durationSeconds;
        public string level;

        public bool IsAcceptable =>
            HasScoreSummary &&
            userMotionDegrees >= MinimumAcceptedMotionDegrees &&
            coachMotionDegrees >= MinimumAcceptedMotionDegrees;

        public bool HasScoreSummary =>
            durationSeconds > 0 &&
            scoreSampleCount >= MinimumAcceptedSamples &&
            averageScore >= 0 && averageScore <= 100 &&
            bestScore >= 0 && bestScore <= 100;
    }

    private static SpineFlowTrainingSession instance;

    private StartPayload startPayload;
    private ModeMeasurement coachMeasurement;
    private ModeMeasurement gameMeasurement;
    private float coachStartedAt;
    private float gameStartedAt;
    private bool coachTiming;
    private bool gameTiming;
    private bool resultSent;
    private string pendingGameLevel = "standard";
    private string lastAcceptedLaunchJson;
    private string pendingLaunchScene;
    private bool hasLoadedInitialScene;
    private bool practiceStartConfirmationAcknowledged;
    private static CoachMotionPackage cachedMotionPackage;
#if UNITY_EDITOR
    private bool editorPracticeConfirmationSessionActive;
#endif

    public static event Action PracticeStartConfirmationReset;

    public static bool HasMobileSession => instance != null && instance.HasValidStartPayload();

    public static CoachMotionPackage CachedMotionPackage => cachedMotionPackage;

    public static string CurrentMobileSessionId =>
        instance != null && instance.HasValidStartPayload()
            ? instance.startPayload.sessionId
            : null;

    public static string CurrentGender =>
        instance != null && instance.HasValidStartPayload()
            ? instance.startPayload.userProfile?.gender
            : null;

    public static string CurrentCoachActionId
    {
        get
        {
            PlannedAction action = instance != null
                ? instance.FindPlannedAction("coach")
                : null;
            return action?.actionId;
        }
    }

    public static void CacheMotionPackage(CoachMotionPackage package)
    {
        cachedMotionPackage = package;
    }

    public static bool IsMaleMobileSession => instance != null && instance.IsMaleSession();

    public static bool ShouldShowPracticeStartConfirmation =>
        ShouldShowPracticeStartConfirmationForSession(
            HasPracticeStartConfirmationSession,
            instance != null && instance.practiceStartConfirmationAcknowledged);

    private static bool HasPracticeStartConfirmationSession
    {
        get
        {
            if (HasMobileSession)
            {
                return true;
            }

#if UNITY_EDITOR
            return instance != null && instance.editorPracticeConfirmationSessionActive;
#else
            return false;
#endif
        }
    }

    public static bool ShouldShowPracticeStartConfirmationForSession(
        bool hasMobileSession,
        bool confirmationAcknowledged)
    {
        return !hasMobileSession || !confirmationAcknowledged;
    }

    public static void AcknowledgePracticeStartConfirmation()
    {
        if (instance != null && HasPracticeStartConfirmationSession)
        {
            instance.practiceStartConfirmationAcknowledged = true;
        }
    }

#if UNITY_EDITOR
    public static void BeginEditorPracticeConfirmationSession()
    {
        if (!Application.isPlaying || instance == null)
        {
            Debug.LogWarning(
                "[SpineFlow] Enter Play Mode before starting the simulated mobile session.");
            return;
        }

        instance.editorPracticeConfirmationSessionActive = true;
        instance.practiceStartConfirmationAcknowledged = false;
        PracticeStartConfirmationReset?.Invoke();
        Debug.Log(
            "[SpineFlow] Simulated mobile confirmation session started. " +
            "The practice confirmation will stay dismissed after Start Training " +
            "until Play Mode ends or F6 starts a new simulated session.",
            instance);
    }
#endif

    /// <summary>
    /// Keeps mobile-launched training in the matching avatar scenes. Scenes without a male
    /// counterpart (for example DeadBugGaming) deliberately remain shared.
    /// </summary>
    public static string ResolveSceneForMobileSession(string sceneName)
    {
        return ResolveSceneForGender(sceneName, IsMaleMobileSession);
    }

    public static string ResolveSceneForGender(string sceneName, bool isMale)
    {
        if (!isMale)
        {
            return sceneName;
        }

        if (string.Equals(sceneName, DeadBugPracticeSceneName, StringComparison.Ordinal))
        {
            return MaleDeadBugPracticeSceneName;
        }

        if (string.Equals(sceneName, DeadBugSceneName, StringComparison.Ordinal))
        {
            return MaleDeadBugSceneName;
        }

        if (string.Equals(sceneName, BirdDogPracticeSceneName, StringComparison.Ordinal))
        {
            return MaleBirdDogPracticeSceneName;
        }

        return sceneName;
    }

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void CreatePersistentSession()
    {
        if (instance != null)
        {
            return;
        }

        GameObject sessionObject = new GameObject(nameof(SpineFlowTrainingSession));
        sessionObject.hideFlags = HideFlags.HideInHierarchy;
        instance = sessionObject.AddComponent<SpineFlowTrainingSession>();
        DontDestroyOnLoad(sessionObject);
    }

    private void Awake()
    {
        if (instance != null && instance != this)
        {
            Destroy(gameObject);
            return;
        }

        instance = this;
        DontDestroyOnLoad(gameObject);
        SceneManager.sceneLoaded += OnSceneLoaded;
        ReadStartPayloadFromAndroidIntent();
    }

    private void OnDestroy()
    {
        SceneManager.sceneLoaded -= OnSceneLoaded;
        if (instance == this)
        {
            instance = null;
        }
    }

    private void OnApplicationFocus(bool hasFocus)
    {
        if (hasFocus)
        {
            ReadStartPayloadFromAndroidIntent();
        }
    }

    private void OnApplicationPause(bool paused)
    {
        if (!paused)
        {
            ReadStartPayloadFromAndroidIntent();
        }
    }

    private void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        hasLoadedInitialScene = true;
        TryLoadPendingLaunchScene(scene.name);
    }

    public static SpineFlowTrainingVolume GetPlannedTrainingVolume(
        string mode,
        int fallbackSets,
        int fallbackRepetitionsPerSet)
    {
        PlannedAction action = instance != null ? instance.FindPlannedAction(mode) : null;
        if (action == null || action.sets < 1 || action.reps < 1)
        {
            return new SpineFlowTrainingVolume(
                fallbackSets,
                fallbackRepetitionsPerSet);
        }

        return new SpineFlowTrainingVolume(action.sets, action.reps);
    }

    public static int GetPlannedRepetitions(string mode, int fallback)
    {
        SpineFlowTrainingVolume volume = GetPlannedTrainingVolume(
            mode,
            1,
            fallback);
        return volume.TotalRepetitions;
    }

    public static void BeginCoachTraining()
    {
        if (instance == null)
        {
            return;
        }

        instance.coachMeasurement = null;
        instance.coachStartedAt = Time.realtimeSinceStartup;
        instance.coachTiming = true;
    }

    public static void CompleteCoachTraining(
        IReadOnlyList<SegmentScore> scores,
        float userMotionDegrees,
        float coachMotionDegrees,
        int completedRepetitions)
    {
        if (instance == null)
        {
            return;
        }

        instance.coachMeasurement = BuildPracticeMeasurement(
            scores,
            userMotionDegrees,
            coachMotionDegrees,
            instance.ElapsedSeconds(instance.coachStartedAt, instance.coachTiming),
            "coach",
            completedRepetitions);
        instance.coachTiming = false;
    }

    public static void BeginGameTraining(string level)
    {
        if (instance == null)
        {
            return;
        }

        // A replay replaces only the game attempt; the completed practice
        // measurement remains part of this phone-launched training session.
        instance.gameMeasurement = null;
        instance.gameStartedAt = Time.realtimeSinceStartup;
        instance.gameTiming = true;
        instance.pendingGameLevel = string.IsNullOrWhiteSpace(level) ? "standard" : level;
    }

    public static void CompleteGameTraining(
        IReadOnlyList<DeadBugCheckpointScore> scores,
        float userMotionDegrees,
        float coachMotionDegrees)
    {
        if (instance == null)
        {
            return;
        }

        instance.gameMeasurement = BuildGameMeasurement(
            scores,
            userMotionDegrees,
            coachMotionDegrees,
            instance.ElapsedSeconds(instance.gameStartedAt, instance.gameTiming),
            instance.pendingGameLevel);
        instance.gameTiming = false;
    }

    public static bool TryReturnToMobile()
    {
        if (instance != null &&
            instance.HasValidStartPayload() &&
            instance.TrySendResultToMobile())
        {
            return true;
        }

        return TryLaunchMobileApp();
    }

    private int ElapsedSeconds(float startedAt, bool timing)
    {
        if (!timing)
        {
            return 0;
        }

        return Mathf.Max(0, Mathf.CeilToInt(Time.realtimeSinceStartup - startedAt));
    }

    private static ModeMeasurement BuildPracticeMeasurement(
        IReadOnlyList<SegmentScore> scores,
        float userMotionDegrees,
        float coachMotionDegrees,
        int durationSeconds,
        string level,
        int completedRepetitions)
    {
        double weightedTotal = 0d;
        double rawWeightedTotal = 0d;
        float semanticWeightTotal = 0f;
        int sampleCount = 0;
        float bestScore = 0f;
        float rawBestScore = 0f;
        if (scores != null)
        {
            foreach (SegmentScore score in scores)
            {
                int frames = Mathf.Max(0, score.frameCount);
                float weight = SegmentWeight(score.segmentKind);
                float rawAverage = ResolveRawScore(
                    score.rawAverageScore,
                    score.rawBestScore,
                    score.averageScore);
                float rawBest = ResolveRawScore(
                    score.rawBestScore,
                    score.rawAverageScore,
                    score.bestScore);
                weightedTotal += score.averageScore * weight;
                rawWeightedTotal += rawAverage * weight;
                semanticWeightTotal += weight;
                sampleCount += frames;
                bestScore = Mathf.Max(bestScore, score.bestScore);
                rawBestScore = Mathf.Max(rawBestScore, rawBest);
            }
        }

        int averageScore = semanticWeightTotal > 0f
            ? FormalActionScoreMappingSettings.RoundForDisplay(
                (float)(weightedTotal / semanticWeightTotal))
            : 0;
        int rawAverageScore = semanticWeightTotal > 0f
            ? Mathf.Clamp(
                Mathf.RoundToInt((float)(rawWeightedTotal / semanticWeightTotal)),
                0,
                100)
            : 0;
        return new ModeMeasurement
        {
            averageScore = averageScore,
            rawAverageScore = rawAverageScore,
            bestScore =
                FormalActionScoreMappingSettings.RoundForDisplay(bestScore),
            rawBestScore = Mathf.Clamp(Mathf.RoundToInt(rawBestScore), 0, 100),
            scoreSampleCount = sampleCount,
            completedRepetitions = Mathf.Max(0, completedRepetitions),
            userMotionDegrees = Mathf.Max(0f, userMotionDegrees),
            coachMotionDegrees = Mathf.Max(0f, coachMotionDegrees),
            durationSeconds = Mathf.Max(0, durationSeconds),
            level = level
        };
    }

    private static ModeMeasurement BuildGameMeasurement(
        IReadOnlyList<DeadBugCheckpointScore> scores,
        float userMotionDegrees,
        float coachMotionDegrees,
        int durationSeconds,
        string level)
    {
        double weightedTotal = 0d;
        double rawWeightedTotal = 0d;
        float semanticWeightTotal = 0f;
        int sampleCount = 0;
        float bestScore = 0f;
        float rawBestScore = 0f;
        int completedRepetitions = 0;
        FormalActionScoreMappingSettings mapping =
            new FormalActionScoreMappingSettings();
        if (scores != null)
        {
            foreach (DeadBugCheckpointScore score in scores)
            {
                int frames = score.frameScores != null
                    ? score.frameScores.Count
                    : Mathf.Max(0, score.TotalFrameCount);
                float weight = score.isUp
                    ? CoreActionWeight
                    : ReturnTransitionWeight;
                float displayedAverage = score.isUp
                    ? mapping.Map(score.averageScore)
                    : score.averageScore;
                float displayedBest = score.isUp
                    ? mapping.Map(score.bestScore)
                    : score.bestScore;
                weightedTotal += displayedAverage * weight;
                rawWeightedTotal += score.averageScore * weight;
                semanticWeightTotal += weight;
                sampleCount += frames;
                bestScore = Mathf.Max(bestScore, displayedBest);
                rawBestScore = Mathf.Max(rawBestScore, score.bestScore);
                if (score.isUp)
                {
                    completedRepetitions++;
                }
            }
        }

        int averageScore = semanticWeightTotal > 0f
            ? FormalActionScoreMappingSettings.RoundForDisplay(
                (float)(weightedTotal / semanticWeightTotal))
            : 0;
        int rawAverageScore = semanticWeightTotal > 0f
            ? Mathf.Clamp(
                Mathf.RoundToInt((float)(rawWeightedTotal / semanticWeightTotal)),
                0,
                100)
            : 0;
        return new ModeMeasurement
        {
            averageScore = averageScore,
            rawAverageScore = rawAverageScore,
            bestScore =
                FormalActionScoreMappingSettings.RoundForDisplay(bestScore),
            rawBestScore = Mathf.Clamp(Mathf.RoundToInt(rawBestScore), 0, 100),
            scoreSampleCount = sampleCount,
            completedRepetitions = completedRepetitions,
            userMotionDegrees = Mathf.Max(0f, userMotionDegrees),
            coachMotionDegrees = Mathf.Max(0f, coachMotionDegrees),
            durationSeconds = Mathf.Max(0, durationSeconds),
            level = string.IsNullOrWhiteSpace(level) ? "standard" : level
        };
    }

    private static float SegmentWeight(TrainingSegmentKind kind)
    {
        if (kind == TrainingSegmentKind.CoreAction)
        {
            return CoreActionWeight;
        }

        if (kind == TrainingSegmentKind.ReturnTransition)
        {
            return ReturnTransitionWeight;
        }

        return BoundaryPoseWeight;
    }

    private static float ResolveRawScore(
        float rawScore,
        float companionRawScore,
        float displayedFallback)
    {
        if (rawScore <= 0f && companionRawScore <= 0f && displayedFallback > 0f)
        {
            return displayedFallback;
        }

        return rawScore;
    }

    private bool TrySendResultToMobile()
    {
        if (resultSent)
        {
            return true;
        }

        if (!HasValidStartPayload())
        {
            Debug.LogWarning("[SpineFlow] No valid phone session is available for result delivery.", this);
            return false;
        }

        string resultJson = BuildResultJson();
#if UNITY_ANDROID && !UNITY_EDITOR
        try
        {
            using (AndroidJavaClass unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
            using (AndroidJavaObject activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity"))
            using (AndroidJavaObject intent = new AndroidJavaObject("android.content.Intent"))
            {
                intent.Call<AndroidJavaObject>("setAction", ResultAction);
                intent.Call<AndroidJavaObject>("setPackage", MobilePackage);
                intent.Call<AndroidJavaObject>("putExtra", ProtocolVersionExtra, ProtocolVersion);
                intent.Call<AndroidJavaObject>("putExtra", PayloadExtra, resultJson);
                intent.Call<AndroidJavaObject>("addFlags", 0x04000000 | 0x20000000);
                activity.Call("startActivity", intent);
                resultSent = true;
                activity.Call("finish");
                return true;
            }
        }
        catch (Exception error)
        {
            Debug.LogError($"[SpineFlow] Failed to return to the mobile app: {error}", this);
            return false;
        }
#else
        Debug.Log($"[SpineFlow] Mobile result preview:\n{resultJson}", this);
        return false;
#endif
    }

    private static bool TryLaunchMobileApp()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        try
        {
            using (AndroidJavaClass unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
            using (AndroidJavaObject activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity"))
            using (AndroidJavaObject launchIntent =
                new AndroidJavaObject("android.content.Intent"))
            {
                launchIntent.Call<AndroidJavaObject>(
                    "setAction",
                    "android.intent.action.MAIN");
                launchIntent.Call<AndroidJavaObject>("setPackage", MobilePackage);
                launchIntent.Call<AndroidJavaObject>(
                    "addCategory",
                    "android.intent.category.LAUNCHER");
                launchIntent.Call<AndroidJavaObject>(
                    "addFlags",
                    0x04000000 | 0x20000000);
                activity.Call("startActivity", launchIntent);
                activity.Call("finish");
                return true;
            }
        }
        catch (Exception error)
        {
            Debug.LogError($"[SpineFlow] Failed to launch the mobile app: {error}");
            return false;
        }
#else
        Debug.Log(
            "[SpineFlow] Returning to the mobile app is only available in an Android build.");
        return false;
#endif
    }

    private string BuildResultJson()
    {
        bool hasCompleteScoreSummary =
            coachMeasurement != null && coachMeasurement.HasScoreSummary &&
            gameMeasurement != null && gameMeasurement.HasScoreSummary;
        bool hasCompletePoseEvidence =
            hasCompleteScoreSummary &&
            coachMeasurement.IsAcceptable &&
            gameMeasurement.IsAcceptable;
        ResultPayload result = new ResultPayload
        {
            sessionId = startPayload.sessionId,
            userId = startPayload.userId,
            requestToken = startPayload.requestToken,
            createdAt = DateTimeOffset.Now.ToString("o", CultureInfo.InvariantCulture),
            status = hasCompleteScoreSummary ? "COMPLETED" : "INTERRUPTED",
            endReason = hasCompleteScoreSummary ? "NORMAL_COMPLETION" : "INSUFFICIENT_MEASUREMENT",
            detailStatus = hasCompletePoseEvidence
                ? MeasuredDetail
                : hasCompleteScoreSummary
                    ? ScoreSummaryDetail
                    : MissingDetail,
            dataOrigin = hasCompletePoseEvidence
                ? MeasuredOrigin
                : hasCompleteScoreSummary
                    ? ScoreSummaryOrigin
                    : string.Empty,
            completed = hasCompleteScoreSummary,
            simulated = false,
            durationSeconds = TotalMeasuredDuration(),
            overallScore = hasCompleteScoreSummary
                ? CombineModeScores(
                    coachMeasurement.averageScore,
                    gameMeasurement.averageScore)
                : 0,
            rawOverallScore = hasCompleteScoreSummary
                ? CombineModeScores(
                    coachMeasurement.rawAverageScore,
                    gameMeasurement.rawAverageScore)
                : 0,
            scoreMappingVersion = ScoreMappingVersion,
            abnormalEvents = new string[0]
        };

        if (!hasCompleteScoreSummary)
        {
            result.actionResults = new ActionResultPayload[0];
            result.gameResults = new GameResultPayload[0];
            return JsonUtility.ToJson(result);
        }

        List<ActionResultPayload> actions = new List<ActionResultPayload>();
        List<GameResultPayload> games = new List<GameResultPayload>();
        foreach (PlannedAction action in startPayload.trainingPlan.recommendedActions)
        {
            string mode = ResolveMode(action);
            ModeMeasurement measurement = mode == "coach" ? coachMeasurement : gameMeasurement;
            actions.Add(BuildActionResult(action, mode, measurement));
            if (mode == "game")
            {
                games.Add(BuildGameResult(action, measurement));
            }
        }

        result.actionResults = actions.ToArray();
        result.gameResults = games.ToArray();
        return JsonUtility.ToJson(result);
    }

    private ActionResultPayload BuildActionResult(
        PlannedAction action,
        string mode,
        ModeMeasurement measurement)
    {
        return new ActionResultPayload
        {
            actionId = action.actionId,
            actionName = action.actionName,
            overallScore = measurement.averageScore,
            rawOverallScore = measurement.rawAverageScore,
            trainingMode = mode,
            plannedRepetitions = SafeRepetitionProduct(action.sets, action.reps),
            completedRepetitions = measurement.completedRepetitions,
            durationSeconds = measurement.durationSeconds,
            averageScore = measurement.averageScore,
            rawAverageScore = measurement.rawAverageScore,
            bestScore = measurement.bestScore,
            rawBestScore = measurement.rawBestScore,
            scoreSampleCount = measurement.scoreSampleCount,
            scoreMappingVersion = ScoreMappingVersion,
            minimumUserMotionDegrees = measurement.userMotionDegrees,
            minimumCoachMotionDegrees = measurement.coachMotionDegrees,
            metrics = new MetricPayload(),
            metricsAvailable = false,
            warnings = new WarningPayload[0],
            passed = measurement.IsAcceptable,
            cheatingDetected = false
        };
    }

    private GameResultPayload BuildGameResult(PlannedAction action, ModeMeasurement measurement)
    {
        string actionId = action.actionId ?? string.Empty;
        string baseId = actionId.EndsWith("_game", StringComparison.Ordinal)
            ? actionId.Substring(0, actionId.Length - "_game".Length)
            : actionId;
        return new GameResultPayload
        {
            gameId = baseId + "_challenge",
            gameName = string.IsNullOrWhiteSpace(action.actionName)
                ? "Dead Bug Challenge"
                : action.actionName,
            score = measurement.averageScore,
            averageScore = measurement.averageScore,
            rawAverageScore = measurement.rawAverageScore,
            bestScore = measurement.bestScore,
            rawBestScore = measurement.rawBestScore,
            scoreSampleCount = measurement.scoreSampleCount,
            completedRepetitions = measurement.completedRepetitions,
            scoreMappingVersion = ScoreMappingVersion,
            level = measurement.level,
            durationSeconds = measurement.durationSeconds
        };
    }

    private static int CombineModeScores(int coachScore, int gameScore)
    {
        return FormalActionScoreMappingSettings.RoundForDisplay(
            coachScore * CoachModeWeight +
            gameScore * GameModeWeight);
    }

    private static int SafeRepetitionProduct(int sets, int repetitionsPerSet)
    {
        long total = (long)Mathf.Max(0, sets) * Mathf.Max(0, repetitionsPerSet);
        return total > int.MaxValue ? int.MaxValue : (int)total;
    }

    private int TotalMeasuredDuration()
    {
        int duration = 0;
        if (coachMeasurement != null)
        {
            duration += coachMeasurement.durationSeconds;
        }

        if (gameMeasurement != null)
        {
            duration += gameMeasurement.durationSeconds;
        }

        return Mathf.Max(0, duration);
    }

    private PlannedAction FindPlannedAction(string mode)
    {
        if (!HasValidStartPayload())
        {
            return null;
        }

        foreach (PlannedAction action in startPayload.trainingPlan.recommendedActions)
        {
            if (ResolveMode(action) == mode)
            {
                return action;
            }
        }

        return null;
    }

    private bool HasValidStartPayload()
    {
        if (startPayload == null || startPayload.protocolVersion != ProtocolVersion ||
            startPayload.messageType != StartMessage ||
            string.IsNullOrWhiteSpace(startPayload.sessionId) ||
            string.IsNullOrWhiteSpace(startPayload.userId) ||
            string.IsNullOrWhiteSpace(startPayload.requestToken) ||
            startPayload.callback == null ||
            startPayload.callback.packageName != MobilePackage ||
            startPayload.callback.action != ResultAction ||
            startPayload.trainingPlan == null ||
            startPayload.trainingPlan.recommendedActions == null ||
            startPayload.trainingPlan.recommendedActions.Length != 2)
        {
            return false;
        }

        bool hasCoach = false;
        bool hasGame = false;
        foreach (PlannedAction action in startPayload.trainingPlan.recommendedActions)
        {
            if (action == null || string.IsNullOrWhiteSpace(action.actionId) ||
                action.sets < 1 || action.sets > 20 || action.reps < 1 || action.reps > 100)
            {
                return false;
            }

            string mode = ResolveMode(action);
            if (mode == "coach")
            {
                if (hasCoach)
                {
                    return false;
                }

                hasCoach = true;
            }
            else if (mode == "game")
            {
                if (hasGame)
                {
                    return false;
                }

                hasGame = true;
            }
            else
            {
                return false;
            }
        }

        return hasCoach && hasGame;
    }

    private bool IsMaleSession()
    {
        return HasValidStartPayload() &&
            string.Equals(
                startPayload.userProfile != null ? startPayload.userProfile.gender : null,
                "male",
                StringComparison.OrdinalIgnoreCase);
    }

    private static string ResolveMode(PlannedAction action)
    {
        string searchable = ((action != null ? action.arTaskType : null) + " " +
            (action != null ? action.actionId : null)).ToLowerInvariant();
        if (searchable.Contains("coach"))
        {
            return "coach";
        }

        if (searchable.Contains("game"))
        {
            return "game";
        }

        return string.Empty;
    }

    private void ReadStartPayloadFromAndroidIntent()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        try
        {
            using (AndroidJavaClass unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
            using (AndroidJavaObject activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity"))
            using (AndroidJavaObject intent = activity.Call<AndroidJavaObject>("getIntent"))
            {
                string json = intent.Call<string>("getStringExtra", PayloadExtra);
                if (string.IsNullOrWhiteSpace(json) || json == lastAcceptedLaunchJson)
                {
                    return;
                }

                int intentProtocolVersion = intent.Call<int>(
                    "getIntExtra",
                    ProtocolVersionExtra,
                    -1);
                AcceptMobileLaunch(json, intentProtocolVersion);
            }
        }
        catch (Exception error)
        {
            Debug.LogError($"[SpineFlow] Failed to read the mobile start payload: {error}", this);
        }
#endif
    }

    private void AcceptMobileLaunch(string json, int intentProtocolVersion)
    {
        lastAcceptedLaunchJson = json;
        StartPayload candidate;
        try
        {
            candidate = JsonUtility.FromJson<StartPayload>(json);
        }
        catch (Exception error)
        {
            Debug.LogError($"[SpineFlow] Invalid mobile start JSON: {error}", this);
            return;
        }

        if (candidate == null)
        {
            Debug.LogError("[SpineFlow] Mobile start payload is empty.", this);
            return;
        }

        startPayload = candidate;
        if (!HasValidStartPayload())
        {
            Debug.LogError(
                "[SpineFlow] Mobile start payload failed validation; staying in the current scene.",
                this);
            return;
        }

        if (intentProtocolVersion != -1 && intentProtocolVersion != ProtocolVersion)
        {
            Debug.LogError(
                $"[SpineFlow] Unsupported Intent protocol version {intentProtocolVersion}.",
                this);
            return;
        }

        ResetMeasurementsForNewSession();
        string targetScene = ResolveSceneForMobileSession(startPayload.targetScene?.Trim());
        if (!IsSupportedPracticeScene(targetScene))
        {
            Debug.LogError(
                $"[SpineFlow] Unsupported mobile target scene '{startPayload.targetScene}'.",
                this);
            return;
        }

        pendingLaunchScene = targetScene;
        PublishPendingMotionPackage();
        Debug.Log(
            $"[SpineFlow] Accepted mobile session {startPayload.sessionId}; " +
            $"first scene is {targetScene}.",
            this);
        TryLoadPendingLaunchScene(SceneManager.GetActiveScene().name);
    }

    private static bool IsSupportedPracticeScene(string sceneName)
    {
        return string.Equals(sceneName, DeadBugPracticeSceneName, StringComparison.Ordinal) ||
            string.Equals(sceneName, MaleDeadBugPracticeSceneName, StringComparison.Ordinal) ||
            string.Equals(sceneName, BirdDogPracticeSceneName, StringComparison.Ordinal) ||
            string.Equals(sceneName, MaleBirdDogPracticeSceneName, StringComparison.Ordinal) ||
            string.Equals(sceneName, UnifiedPracticeSceneName, StringComparison.Ordinal);
    }

    private void PublishPendingMotionPackage()
    {
        PracticeSceneInitializer.PendingMotionPackageUri = null;
        PlannedAction[] actions = startPayload?.trainingPlan?.recommendedActions;
        if (actions == null)
        {
            return;
        }

        foreach (PlannedAction action in actions)
        {
            if (action != null && !string.IsNullOrWhiteSpace(action.motionPackageUri))
            {
                PracticeSceneInitializer.PendingMotionPackageUri = action.motionPackageUri.Trim();
                return;
            }
        }
    }

    private void ResetMeasurementsForNewSession()
    {
        coachMeasurement = null;
        gameMeasurement = null;
        coachTiming = false;
        gameTiming = false;
        resultSent = false;
        pendingGameLevel = "standard";
        practiceStartConfirmationAcknowledged = false;
        cachedMotionPackage = null;
#if UNITY_EDITOR
        editorPracticeConfirmationSessionActive = false;
#endif
        PracticeStartConfirmationReset?.Invoke();
    }

    private void TryLoadPendingLaunchScene(string currentSceneName)
    {
        if (!hasLoadedInitialScene || string.IsNullOrEmpty(pendingLaunchScene))
        {
            return;
        }

        string targetScene = pendingLaunchScene;
        pendingLaunchScene = null;
        if (string.Equals(currentSceneName, targetScene, StringComparison.Ordinal))
        {
            Debug.Log($"[SpineFlow] Mobile launch is already in {targetScene}.", this);
            return;
        }

        if (!Application.CanStreamedLevelBeLoaded(targetScene))
        {
            Debug.LogError(
                $"[SpineFlow] Target scene {targetScene} is missing from Build Settings.",
                this);
            return;
        }

        Debug.Log($"[SpineFlow] Routing {currentSceneName} -> {targetScene}.", this);
        SceneManager.LoadScene(targetScene, LoadSceneMode.Single);
    }
}
