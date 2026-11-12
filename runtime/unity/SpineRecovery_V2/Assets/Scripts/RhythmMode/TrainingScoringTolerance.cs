using System;

/// <summary>Bounded measurement tolerance. Missing samples never become valid time,
/// and qualification never changes the prescribed target or the full-score dose.</summary>
public static class TrainingScoringTolerance
{
    public const double ShortGapSeconds = .2;
    public const float MaximumResponseDelay = .55f;
    public static bool GapExpired(double seconds)=>seconds>ShortGapSeconds+.00001;
    public static bool HoldQualified(double measured,double target,bool continuous)
    {
        if(double.IsNaN(measured)||double.IsInfinity(measured)||target<=0)return false;
        // Continuous holds allow at most 3% missing evidence; dynamic repetitions
        // allow 10% variation in the measured plateau. Neither grants full marks.
        return measured+.0001>=target*(continuous?.97:.90);
    }
}
