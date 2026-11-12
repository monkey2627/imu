using UnityEngine;

/// <summary>
/// SDK-neutral telemetry boundary for any component that drives a humanoid Animator.
/// A future camera, file, or wearable adapter can update this component without
/// coupling gameplay code to a vendor transport.
/// </summary>
public sealed class AvatarMotionSource : MonoBehaviour
{
    [SerializeField] private GameObject model;
    [SerializeField] private bool inputAvailable = true;

    public GameObject Model => model;
    public bool Freeze { get; set; }
    public bool RootScaleOffsetSynchronize { get; set; }
    public bool SyncCalibrationModeWithScaleOffsetSynchronize { get; set; }
    public int LastBonePacketCounterInFrame { get; private set; }
    public int LastPacketframeCounterInFrame { get; private set; }
    public float RemoteTime { get; private set; }
    public float LastFrameReceivedAt { get; private set; } = float.NegativeInfinity;
    public bool HasFreshFrame => GetAvailable()!=0 && LastBonePacketCounterInFrame>0 &&
        Time.unscaledTime-LastFrameReceivedAt<=.5f;

    public int GetAvailable() => inputAvailable && !Freeze ? 1 : 0;
    public float GetRemoteTime() => RemoteTime;

    public void Configure(Animator animator)
    {
        model = animator != null ? animator.gameObject : null;
    }

    public void ReportFrame(float remoteTime, int humanoidBonePackets)
    {
        inputAvailable = true;
        RemoteTime = remoteTime;
        LastBonePacketCounterInFrame = Mathf.Max(0, humanoidBonePackets);
        LastPacketframeCounterInFrame++;
        LastFrameReceivedAt=Time.unscaledTime;
    }

    public void SetAvailable(bool available)
    {
        inputAvailable = available;
    }
}
