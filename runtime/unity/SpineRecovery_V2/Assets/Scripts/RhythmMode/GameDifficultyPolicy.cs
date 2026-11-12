/// <summary>Game difficulty changes acceptance, never the prescribed dose or coach clock.</summary>
public static class GameDifficultyPolicy
{
    // The first release intentionally allows a useful but imperfect excursion.
    // All levels still require fresh calibrated input and the correct active limb.
    public static float HitThreshold(string level) => level=="simple"?35f:level=="hard"?60f:45f;
    public static float MinimumExtension(string level,RhythmBodyTarget target=RhythmBodyTarget.LeftHand)
    {
        // Thigh tracking is more sensitive to strap placement and floor calibration.
        // Keep a floor above the measured half-range cases, even on Simple.
        bool leg=target==RhythmBodyTarget.LeftFoot||target==RhythmBodyTarget.RightFoot;
        return leg ? (level=="simple"?.63f:level=="hard"?.72f:.66f)
            : (level=="simple"?.64f:level=="hard"?.76f:.70f);
    }
    public static float FollowingThreshold(string level) => level=="simple"?50f:level=="hard"?70f:60f;
    public static float QualifiedPhaseFraction(string level) => level=="simple"?.70f:level=="hard"?.80f:.75f;
    public static float BridgeProgressTolerance(string level) => level=="simple"?.24f:level=="hard"?.16f:.20f;
}
