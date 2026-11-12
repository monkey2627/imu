using System;
using System.Collections;
using UnityEngine;

public sealed class ReplaySceneBootstrap : MonoBehaviour
{
    [SerializeField] private ReplayPlayer player;
    [SerializeField] private ReplayPlacementController placement;
    [SerializeField] private ReplayUIController ui;
    [SerializeField] private Transform replayRoot;
    [SerializeField] private Animator userAnimator;
    [SerializeField] private Animator coachAnimator;
    [SerializeField] private Transform userPresentationRoot;
    [SerializeField] private Transform coachPresentationRoot;
    [SerializeField, Min(0f)] private float pendingCommitWaitSeconds = 10f;

    private ReplayRepository repository;

    public void Configure(
        ReplayPlayer replayPlayer,
        ReplayPlacementController placementController,
        ReplayUIController uiController,
        Transform spatialRoot,
        Animator existingUserAnimator,
        Animator existingCoachAnimator)
    {
        player = replayPlayer;
        placement = placementController;
        ui = uiController;
        replayRoot = spatialRoot;
        userAnimator = existingUserAnimator;
        coachAnimator = existingCoachAnimator;
        userPresentationRoot = userAnimator != null ? userAnimator.transform : null;
        coachPresentationRoot = coachAnimator != null ? coachAnimator.transform : null;
    }

    private IEnumerator Start()
    {
        repository = new ReplayRepository();
        player.SetLoading();
        string requestedSessionId = ReplaySessionContext.SessionId;
        bool loadedFromLatest = string.IsNullOrWhiteSpace(requestedSessionId);
        ReplayLoadResult result;

        if (!loadedFromLatest)
        {
            float deadline = Time.realtimeSinceStartup + pendingCommitWaitSeconds;
            do
            {
                result = repository.Load(requestedSessionId);
                if (result.IsSuccess ||
                    ReplaySessionContext.State != ReplaySessionContextState.Pending ||
                    result.Status != ReplayLoadStatus.Unavailable ||
                    Time.realtimeSinceStartup >= deadline)
                {
                    break;
                }

                yield return null;
            }
            while (true);
        }
        else
        {
            result = repository.LoadLatestCompleted();
        }

        if (!result.IsSuccess && loadedFromLatest)
        {
            result = TryMigrateLegacy();
        }

        if (!result.IsSuccess)
        {
            if (!loadedFromLatest)
            {
                ReplaySessionContext.ClearIfMatches(requestedSessionId);
            }

            string message = result.Status == ReplayLoadStatus.Unavailable
                ? loadedFromLatest
                    ? "No completed replay is available. Complete a training session first."
                    : "This training replay was not saved. Return to training and complete it again."
                : string.IsNullOrWhiteSpace(result.Error)
                    ? "No valid replay is available."
                    : result.Error;
            player.Fail(message);
            ui?.ShowLoadError(message);
            yield break;
        }

        ReplayManifest manifest = result.Session.Manifest;
        if (string.IsNullOrWhiteSpace(manifest.actionId) ||
            manifest.actionId == "unknown")
        {
            const string message = "Replay action metadata is missing.";
            player.Fail(message);
            ui?.ShowLoadError(message);
            yield break;
        }

        if (!TryResolveAvatars(manifest, out string avatarError))
        {
            player.Fail(avatarError);
            ui?.ShowLoadError(avatarError);
            yield break;
        }

        if (!player.TryLoad(
                result.Session,
                userAnimator,
                coachAnimator,
                userPresentationRoot,
                coachPresentationRoot,
                replayRoot,
                out string loadError))
        {
            ui?.ShowLoadError(loadError);
            yield break;
        }

        // The recorder may be destroyed by the scene transition while its
        // background commit is completing. Loading the requested session is
        // the authoritative point at which that pending context becomes
        // committed and its return routes become available to the existing UI.
        ReplaySessionContext.SetCommitted(manifest);
    }

    private ReplayLoadResult TryMigrateLegacy()
    {
        ReplayManifest template = ReplayContextResolver.CaptureCurrent(
            userAnimator,
            coachAnimator,
            30f);
        // V2 did not persist action/avatar identity. The generic PlayBack
        // scene cannot safely infer them, so automatic migration remains
        // unavailable unless a caller supplies explicit known context to the
        // migrator API.
        template.actionId = "unknown";
        template.avatarId = string.Empty;
        var migrator = new LegacyReplayV2Migrator();
        if (!migrator.TryMigrate(
                template,
                userAnimator,
                coachAnimator,
                userPresentationRoot,
                coachPresentationRoot,
                replayRoot,
                repository,
                out ReplayManifest migrated,
                out string migrationError))
        {
            return ReplayLoadResult.Failure(
                ReplayLoadStatus.Unavailable,
                migrationError);
        }

        return repository.Load(migrated.sessionId);
    }

    private bool TryResolveAvatars(ReplayManifest manifest, out string error)
    {
        if (!TryResolveAvatar(
                ref userAnimator,
                ref userPresentationRoot,
                manifest.avatarId,
                manifest.gender,
                "user",
                out error))
        {
            return false;
        }

        // PlayBack owns one stable coach model for every user gender. The
        // recorded coach HumanPose is retargeted onto that authored model;
        // only the user's presentation avatar follows manifest.gender.
        if (coachAnimator == null)
        {
            error = "Replay coach Avatar is missing from the PlayBack scene.";
            return false;
        }

        coachPresentationRoot = coachAnimator.transform;
        error = null;
        return true;
    }

    private bool TryResolveAvatar(
        ref Animator animator,
        ref Transform presentationRoot,
        string avatarId,
        string gender,
        string role,
        out string error)
    {
        if (IsAvatarCompatible(animator, avatarId, gender))
        {
            presentationRoot = animator.transform;
            error = null;
            return true;
        }

        string resourceKey = ReplayContextResolver.NormalizeGender(gender);
        GameObject prefab = Resources.Load<GameObject>($"ReplayAvatars/{resourceKey}");
        if (prefab == null && !string.IsNullOrWhiteSpace(avatarId))
        {
            prefab = Resources.Load<GameObject>($"ReplayAvatars/{avatarId}");
        }

        if (prefab == null)
        {
            error =
                $"Replay {role} Avatar '{avatarId}' ({gender}) is not available in this build.";
            return false;
        }

        Transform previousTransform = animator != null ? animator.transform : null;
        Transform parent = previousTransform != null
            ? previousTransform.parent
            : replayRoot;
        GameObject instance = Instantiate(prefab, parent);
        instance.name = $"Replay{char.ToUpperInvariant(role[0])}{role.Substring(1)}Avatar";
        if (previousTransform != null)
        {
            Vector3 prefabScale = prefab.transform.localScale;
            instance.transform.SetPositionAndRotation(
                previousTransform.position,
                previousTransform.rotation);
            instance.transform.localScale = Vector3.Scale(
                previousTransform.localScale,
                prefabScale);
        }
        Animator resolved = instance.GetComponentInChildren<Animator>(true);
        if (resolved == null || resolved.avatar == null || !resolved.isHuman)
        {
            Destroy(instance);
            error = $"Replay {role} Avatar resource is not Humanoid-compatible.";
            return false;
        }

        if (animator != null)
        {
            animator.transform.gameObject.SetActive(false);
        }

        animator = resolved;
        presentationRoot = instance.transform;
        placement?.ReplaceContentTransform(previousTransform, presentationRoot);
        error = null;
        return true;
    }

    private static bool IsAvatarCompatible(
        Animator animator,
        string avatarId,
        string gender)
    {
        if (animator == null || animator.avatar == null || !animator.isHuman)
        {
            return false;
        }

        // Avatar names imported from VRM are often generic (for example
        // "created" or "humanoid") and "female" also contains the text
        // "male". Compare the actual Avatar asset used by the authoritative
        // gender prefab before considering the descriptive fallback.
        string resourceKey = ReplayContextResolver.NormalizeGender(gender);
        GameObject expectedPrefab =
            Resources.Load<GameObject>($"ReplayAvatars/{resourceKey}");
        Animator expectedAnimator = expectedPrefab != null
            ? expectedPrefab.GetComponentInChildren<Animator>(true)
            : null;
        if (expectedAnimator != null && expectedAnimator.avatar != null)
        {
            return animator.avatar == expectedAnimator.avatar;
        }

        string current = (animator.avatar.name + " " + animator.gameObject.name)
            .ToLowerInvariant();
        if (!string.IsNullOrWhiteSpace(avatarId) &&
            current.Contains(avatarId.ToLowerInvariant()))
        {
            return true;
        }

        return current.Contains(resourceKey);
    }
}
