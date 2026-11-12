using System;
using System.Text;
using UnityEngine;
using UnityEngine.SceneManagement;

public enum ReplaySessionContextState
{
    None,
    Pending,
    Committed
}

public static class ReplaySessionContext
{
    public static string SessionId { get; private set; }
    public static ReplaySessionContextState State { get; private set; }
    public static string ReturnPracticeScene { get; private set; }
    public static string ReturnGamingScene { get; private set; }

    public static void SetPending(string sessionId)
    {
        Set(sessionId, ReplaySessionContextState.Pending, null, null);
    }

    public static void SetPending(ReplayManifest manifest)
    {
        SetManifest(manifest, ReplaySessionContextState.Pending);
    }

    public static void SetCommitted(string sessionId)
    {
        Set(sessionId, ReplaySessionContextState.Committed, null, null);
    }

    public static void SetCommitted(ReplayManifest manifest)
    {
        SetManifest(manifest, ReplaySessionContextState.Committed);
    }

    public static void Clear()
    {
        SessionId = null;
        State = ReplaySessionContextState.None;
        ReturnPracticeScene = null;
        ReturnGamingScene = null;
    }

    public static void ClearIfMatches(string sessionId)
    {
        if (!string.IsNullOrWhiteSpace(sessionId) &&
            string.Equals(SessionId, sessionId, StringComparison.Ordinal))
        {
            Clear();
        }
    }

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetForDomainReload()
    {
        Clear();
    }

    private static void SetManifest(
        ReplayManifest manifest,
        ReplaySessionContextState state)
    {
        if (manifest == null)
        {
            throw new ArgumentNullException(nameof(manifest));
        }

        Set(
            manifest.sessionId,
            state,
            manifest.returnPracticeScene,
            manifest.returnGamingScene);
    }

    private static void Set(
        string sessionId,
        ReplaySessionContextState state,
        string returnPracticeScene,
        string returnGamingScene)
    {
        if (string.IsNullOrWhiteSpace(sessionId))
        {
            throw new ArgumentException("Replay sessionId is required.", nameof(sessionId));
        }

        SessionId = sessionId;
        State = state;
        ReturnPracticeScene = string.IsNullOrWhiteSpace(returnPracticeScene)
            ? null
            : returnPracticeScene.Trim();
        ReturnGamingScene = string.IsNullOrWhiteSpace(returnGamingScene)
            ? null
            : returnGamingScene.Trim();
    }
}

public sealed class ReplayContextRequest
{
    public string mobileSessionId;
    public string actionId;
    public string actionPackageId;
    public string actionPackageVersion;
    public string sourceScene;
    public string returnPracticeScene;
    public string returnGamingScene;
    public string avatarId;
    public string coachAvatarId;
    public string gender;
    public string rigId;
    public float sampleRate = 30f;
}

public static class ReplayContextResolver
{
    public static ReplayManifest CaptureCurrent(
        Animator userAnimator,
        Animator coachAnimator,
        float sampleRate)
    {
        PracticeActionConfig config = PracticeSceneInitializer.GetCurrentConfig();
        CoachMotionPackage package = SpineFlowTrainingSession.CachedMotionPackage;
        string sourceScene = SceneManager.GetActiveScene().name;
        string actionId = FirstNonEmpty(
            package?.actionId,
            SpineFlowTrainingSession.CurrentCoachActionId,
            config?.actionId,
            InferActionId(sourceScene));
        string gender = FirstNonEmpty(
            SpineFlowTrainingSession.CurrentGender,
            sourceScene.IndexOf("male", StringComparison.OrdinalIgnoreCase) >= 0
                ? "male"
                : "female");

        return BuildManifest(new ReplayContextRequest
        {
            mobileSessionId = SpineFlowTrainingSession.CurrentMobileSessionId,
            actionId = actionId,
            actionPackageId = FirstNonEmpty(
                package?.packageId,
                package?.sourceClipName,
                actionId),
            actionPackageVersion = FirstNonEmpty(
                package?.packageVersion,
                package != null ? package.formatVersion.ToString() : "builtin"),
            sourceScene = sourceScene,
            returnPracticeScene = sourceScene,
            returnGamingScene = FirstNonEmpty(
                config?.gamingSceneName,
                InferGamingScene(actionId)),
            avatarId = ResolveAvatarId(userAnimator, gender),
            coachAvatarId = ResolveAvatarId(coachAnimator, gender),
            gender = gender,
            rigId = userAnimator?.avatar != null
                ? userAnimator.avatar.name
                : ResolveAvatarId(userAnimator, gender),
            sampleRate = sampleRate
        });
    }

    public static ReplayManifest BuildManifest(ReplayContextRequest request)
    {
        if (request == null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        string actionId = NormalizeActionId(request.actionId);
        string gender = NormalizeGender(request.gender);
        string mobilePrefix = SanitizeIdentifier(request.mobileSessionId);
        string uniqueId = Guid.NewGuid().ToString("N");
        string sessionId = string.IsNullOrWhiteSpace(mobilePrefix)
            ? uniqueId
            : $"{mobilePrefix}-{uniqueId}";

        return new ReplayManifest
        {
            formatVersion = ReplayManifest.CurrentFormatVersion,
            sessionId = sessionId,
            createdAtUtc = DateTime.UtcNow.ToString("o"),
            actionId = actionId,
            actionPackageId = FirstNonEmpty(request.actionPackageId, actionId),
            actionPackageVersion = FirstNonEmpty(
                request.actionPackageVersion,
                "builtin"),
            sourceScene = FirstNonEmpty(request.sourceScene, "UnknownPractice"),
            returnPracticeScene = FirstNonEmpty(
                request.returnPracticeScene,
                request.sourceScene,
                "CoachingChoose"),
            returnGamingScene = FirstNonEmpty(
                request.returnGamingScene,
                InferGamingScene(actionId)),
            avatarId = FirstNonEmpty(request.avatarId, gender),
            coachAvatarId = FirstNonEmpty(request.coachAvatarId, gender),
            gender = gender,
            rigId = FirstNonEmpty(request.rigId, request.avatarId, gender),
            coordinateSystemVersion = ReplayManifest.RootBodyPoseCoordinateSystem,
            sampleRate = Mathf.Max(1f, request.sampleRate),
            frameCount = 0,
            duration = 0d,
            completed = false,
            payloadFile = ReplayManifest.DefaultPayloadFile,
            payloadSha256 = string.Empty
        };
    }

    public static string NormalizeActionId(string actionId)
    {
        string normalized = string.IsNullOrWhiteSpace(actionId)
            ? "unknown"
            : actionId.Trim().ToLowerInvariant().Replace("-", string.Empty).Replace("_", string.Empty);
        if (normalized.Contains("birddog"))
        {
            return "birddog";
        }

        if (normalized.Contains("deadbug"))
        {
            return "deadbug";
        }

        return string.IsNullOrWhiteSpace(actionId)
            ? "unknown"
            : actionId.Trim();
    }

    public static string NormalizeGender(string gender)
    {
        return string.Equals(gender?.Trim(), "male", StringComparison.OrdinalIgnoreCase)
            ? "male"
            : "female";
    }

    private static string ResolveAvatarId(Animator animator, string fallback)
    {
        if (animator?.avatar != null && !string.IsNullOrWhiteSpace(animator.avatar.name))
        {
            return animator.avatar.name;
        }

        return animator != null && !string.IsNullOrWhiteSpace(animator.gameObject.name)
            ? animator.gameObject.name
            : fallback;
    }

    private static string InferActionId(string sceneName)
    {
        return sceneName != null &&
            sceneName.IndexOf("bird", StringComparison.OrdinalIgnoreCase) >= 0
                ? "birddog"
                : "deadbug";
    }

    private static string InferGamingScene(string actionId)
    {
        return string.Equals(
            NormalizeActionId(actionId),
            "birddog",
            StringComparison.OrdinalIgnoreCase)
                ? "BirdDogGaming"
                : "DeadBugGaming";
    }

    private static string SanitizeIdentifier(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        var builder = new StringBuilder(value.Length);
        foreach (char character in value.Trim())
        {
            if (char.IsLetterOrDigit(character) || character == '-' || character == '_')
            {
                builder.Append(character);
            }
        }

        return builder.Length == 0 ? null : builder.ToString();
    }

    private static string FirstNonEmpty(params string[] values)
    {
        if (values != null)
        {
            for (int i = 0; i < values.Length; i++)
            {
                if (!string.IsNullOrWhiteSpace(values[i]))
                {
                    return values[i].Trim();
                }
            }
        }

        return string.Empty;
    }
}
