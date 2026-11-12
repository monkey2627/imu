using UnityEngine;

/// <summary>Maps confirmed HITs to their limb tracker, or exclusively the waist in hip thrust.</summary>
public sealed class RhythmHitHapticAdapter : MonoBehaviour
{
    [SerializeField] private RhythmModeController rhythmController;
    [SerializeField] private CoachHapticFeedbackController hapticController;
    private HipThrustTaikoController hipThrustController;

    private void Awake()
    {
        ResolveReferences();
    }

    private void OnEnable()
    {
        ResolveReferences();
        if (hipThrustController != null)
        {
            hipThrustController.HitResolved -= HandleHipThrustHit;
            hipThrustController.HitResolved += HandleHipThrustHit;
        }
        else if (rhythmController != null)
        {
            rhythmController.NoteResolved -= HandleNoteResolved;
            rhythmController.NoteResolved += HandleNoteResolved;
        }
    }

    private void OnDisable()
    {
        if (hipThrustController != null) hipThrustController.HitResolved -= HandleHipThrustHit;
        if (rhythmController != null)
        {
            rhythmController.NoteResolved -= HandleNoteResolved;
        }

        hapticController?.StopAllHaptics();
    }

    private void ResolveReferences()
    {
        if (hipThrustController == null)
            hipThrustController = FindObjectOfType<HipThrustTaikoController>(true);
        if (rhythmController == null)
        {
            rhythmController = FindObjectOfType<RhythmModeController>(true);
        }

        if (hapticController == null)
        {
            hapticController = FindObjectOfType<CoachHapticFeedbackController>(true);
        }

        if (hapticController == null)
        {
            hapticController = gameObject.AddComponent<CoachHapticFeedbackController>();
        }
    }

    private void HandleNoteResolved(RhythmNoteResult result)
    {
        if (!result.IsHit || hapticController == null || hipThrustController != null)
        {
            return;
        }

        switch (result.Target)
        {
            case RhythmBodyTarget.RightHand:
                hapticController.PulseOnce(TrackerWearLocation.RightForearm);
                break;
            case RhythmBodyTarget.LeftHand:
                hapticController.PulseOnce(TrackerWearLocation.LeftForearm);
                break;
            case RhythmBodyTarget.RightFoot:
                hapticController.PulseOnce(TrackerWearLocation.RightLowerLeg);
                break;
            case RhythmBodyTarget.LeftFoot:
                hapticController.PulseOnce(TrackerWearLocation.LeftLowerLeg);
                break;
            case RhythmBodyTarget.Waist:
                hapticController.PulseOnce(TrackerWearLocation.Abdomen);
                break;
        }
    }

    private void HandleHipThrustHit(bool hit)
    {
        if (hit && hapticController != null)
            hapticController.PulseOnce(TrackerWearLocation.Abdomen);
    }
}
