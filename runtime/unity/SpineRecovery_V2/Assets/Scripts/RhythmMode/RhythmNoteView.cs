using UnityEngine;

public sealed class RhythmNoteView : MonoBehaviour
{
    private RhythmModeController owner;
    private RhythmBodyTarget target;
    private Vector3 spawnLocalPosition;
    private Vector3 entryLocalPosition;
    private Vector3 entryVelocity;
    private Vector3 hitLocalPosition;
    private Vector3 resultLocalPosition;
    private double hitTime;
    private float approachDuration;
    private float entryDurationSeconds;
    private float holdDurationSeconds;
    private float noteRadius;
    private RhythmNoteTargetIndicator targetIndicator;
    private double missDeadline;
    private bool waitingForHit;
    private bool resolved;

    public RhythmBodyTarget Target => target;
    public RhythmMovementRange JudgementReference { get; private set; }
    public void SetJudgementReference(RhythmMovementRange reference) { JudgementReference=reference; }
    public double HitTime => hitTime;
    public Vector3 HitLocalPosition => resultLocalPosition;

    public void Initialize(
        RhythmModeController noteOwner,
        RhythmBodyTarget noteTarget,
        Vector3 spawnPosition,
        Vector3 hitPosition,
        double targetHitTime,
        float noteApproachDuration,
        float noteEntryDurationSeconds,
        float noteHoldDurationSeconds,
        float targetRadius,
        RhythmNoteTargetIndicator arrivalIndicator,
        float visualRadius,
        Vector3 judgementResultPosition)
    {
        owner = noteOwner;
        target = noteTarget;
        spawnLocalPosition = spawnPosition;
        hitLocalPosition = hitPosition;
        resultLocalPosition = judgementResultPosition;
        hitTime = targetHitTime;
        approachDuration = Mathf.Max(0.1f, noteApproachDuration);
        entryDurationSeconds = Mathf.Max(0f, noteEntryDurationSeconds);
        holdDurationSeconds = Mathf.Max(0f, noteHoldDurationSeconds);
        targetIndicator = arrivalIndicator;
        noteRadius = Mathf.Max(0f, visualRadius);
        Vector2 approachDirection = new Vector2(
            spawnLocalPosition.x - hitLocalPosition.x,
            spawnLocalPosition.y - hitLocalPosition.y).normalized;
        float entryOffset = Mathf.Max(0f, targetRadius) + noteRadius * 0.25f;
        entryLocalPosition = hitLocalPosition + new Vector3(
            approachDirection.x * entryOffset,
            approachDirection.y * entryOffset,
            0f);
        entryVelocity = entryDurationSeconds > 0f
            ? (hitLocalPosition - entryLocalPosition) / entryDurationSeconds
            : Vector3.zero;
        missDeadline = 0d;
        waitingForHit = false;
        resolved = false;
        transform.localPosition = spawnLocalPosition;
        transform.localRotation = Quaternion.identity;
    }

    private void Update()
    {
        if (resolved || owner == null || !owner.IsRunning)
        {
            return;
        }

        double songTime = owner.SongTime;
        if (songTime < hitTime)
        {
            float progress = 1f - (float)((hitTime - songTime) / approachDuration);
            transform.localPosition = EvaluateHermite(
                spawnLocalPosition,
                Vector3.zero,
                entryLocalPosition,
                entryVelocity,
                approachDuration,
                progress);
        }
        else
        {
            float entryProgress = entryDurationSeconds <= 0f
                ? 1f
                : (float)((songTime - hitTime) / entryDurationSeconds);
            transform.localPosition = EvaluateHermite(
                entryLocalPosition,
                entryVelocity,
                hitLocalPosition,
                Vector3.zero,
                entryDurationSeconds,
                entryProgress);
        }

        float pulse = owner.UsesRingArcTarget
            ? 1f
            : 1f + Mathf.Sin((float)songTime * 10f) * 0.035f;
        transform.localScale = owner.NoteBaseScale * pulse;

        if (songTime >= hitTime)
        {
            if (!waitingForHit)
            {
                waitingForHit = true;
                missDeadline = hitTime + entryDurationSeconds + holdDurationSeconds;
            }

            RhythmNoteResult result = owner.JudgeAtLine(this);
            if (result.Judgement != RhythmJudgement.Miss || songTime >= missDeadline)
            {
                Resolve(result);
            }
        }
    }

    private static Vector3 EvaluateHermite(
        Vector3 startPosition,
        Vector3 startVelocity,
        Vector3 endPosition,
        Vector3 endVelocity,
        float duration,
        float normalizedTime)
    {
        float time = Mathf.Clamp01(normalizedTime);
        float timeSquared = time * time;
        float timeCubed = timeSquared * time;
        float safeDuration = Mathf.Max(0.0001f, duration);

        float startPositionWeight = 2f * timeCubed - 3f * timeSquared + 1f;
        float startVelocityWeight = timeCubed - 2f * timeSquared + time;
        float endPositionWeight = -2f * timeCubed + 3f * timeSquared;
        float endVelocityWeight = timeCubed - timeSquared;

        return startPositionWeight * startPosition +
            startVelocityWeight * safeDuration * startVelocity +
            endPositionWeight * endPosition +
            endVelocityWeight * safeDuration * endVelocity;
    }

    private void OnDestroy()
    {
        if (targetIndicator != null)
        {
            Destroy(targetIndicator.gameObject);
        }
    }

    private void Resolve(RhythmNoteResult result)
    {
        if (resolved)
        {
            return;
        }

        resolved = true;
        owner.ResolveNote(this, result);
    }
}
