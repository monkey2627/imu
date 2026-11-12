using System;
using UnityEngine;

/// <summary>
/// 服务器下发的教练动作数据包。在 Unity 中从 FBX 录制导出，
/// 以 Humanoid muscle space 存储，与具体 avatar 无关（男女模型通用）。
/// </summary>
[Serializable]
public class CoachMotionPackage
{
    public const int CurrentFormatVersion = 1;

    public int formatVersion = CurrentFormatVersion;

    /// <summary>与手机端 payload 的 actionId 对应，如 "dead_bug"</summary>
    public string actionId;

    /// <summary>服务器动作包的稳定标识；旧包缺省时回退到 sourceClipName/actionId。</summary>
    public string packageId;

    /// <summary>服务器动作包业务版本；与二进制 formatVersion 分开演进。</summary>
    public string packageVersion;

    public string displayName;

    /// <summary>采样帧率</summary>
    public float fps = 30f;

    /// <summary>源 FBX 名称，仅用于溯源</summary>
    public string sourceClipName;

    /// <summary>整个动作序列重复的组数（手机端 payload 可覆盖）</summary>
    public int setCount = 1;

    public CoachMotionSegment[] segments = Array.Empty<CoachMotionSegment>();

    public bool IsValid(out string error)
    {
        if (formatVersion != CurrentFormatVersion)
        {
            error = $"formatVersion {formatVersion} 不受支持（当前 {CurrentFormatVersion}）";
            return false;
        }

        if (segments == null || segments.Length == 0)
        {
            error = "segments 为空";
            return false;
        }

        for (int i = 0; i < segments.Length; i++)
        {
            if (!segments[i].IsValid(out string segmentError))
            {
                error = $"segment[{i}] '{segments[i].label}': {segmentError}";
                return false;
            }
        }

        error = null;
        return true;
    }
}
