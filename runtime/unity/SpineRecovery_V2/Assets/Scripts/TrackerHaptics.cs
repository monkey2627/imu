using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.Networking;
// Shared transport for discrete rhythm hit feedback.

/// <summary>
/// Physical locations of the wearable trackers. This is intentionally
/// separate from BodyPart, which is the coarser visual-guidance vocabulary.
/// </summary>
public enum TrackerWearLocation
{
    LeftUpperArm,
    LeftForearm,
    RightUpperArm,
    RightForearm,
    LeftThigh,
    LeftLowerLeg,
    RightThigh,
    RightLowerLeg,
    Chest,
    Abdomen
}

[Serializable]
public sealed class TrackerMotorEndpoint
{
    public TrackerWearLocation location;
    [Tooltip("Tracker IPv4 address, for example 192.168.31.137")]
    public string address;
    [Min(1)] public int durationMs = 100;
}

[Serializable]
public sealed class SlimeVrHapticTrackerSnapshot
{
    public string hardwareId;
    public int trackerNum;
    public string name;
    public string ip;
    public string bodyPosition;
    public string status;
    public bool connected;
}

[Serializable]
public sealed class SlimeVrHapticTrackerResponse
{
    public int apiVersion;
    public SlimeVrHapticTrackerSnapshot[] trackers;
}

/// <summary>
/// Sends one non-blocking timed pulse to the tracker selected by wear location.
/// Callers are responsible for invoking PulseOnce only for a confirmed hit.
/// </summary>
public sealed class CoachHapticFeedbackController : MonoBehaviour
{
    [Header("SlimeVR Discovery")]
    [SerializeField] private string discoveryUrl =
        "http://127.0.0.1:21111/api/haptics/trackers";
    [SerializeField, Min(0.5f)] private float discoveryIntervalSeconds = 2f;

    [Header("Hit Haptics")]
    [SerializeField] private bool enabledForDemo = true;
    [SerializeField, Min(1)] private int pulseDurationMs = 100;
    [SerializeField, Min(1)] private int requestTimeoutSeconds = 1;
    [SerializeField] private TrackerMotorEndpoint[] endpoints =
    {
        new TrackerMotorEndpoint { location = TrackerWearLocation.LeftUpperArm, address = "" },
        new TrackerMotorEndpoint { location = TrackerWearLocation.LeftForearm, address = "" },
        new TrackerMotorEndpoint { location = TrackerWearLocation.RightUpperArm, address = "" },
        new TrackerMotorEndpoint { location = TrackerWearLocation.RightForearm, address = "" },
        new TrackerMotorEndpoint { location = TrackerWearLocation.LeftThigh, address = "" },
        new TrackerMotorEndpoint { location = TrackerWearLocation.LeftLowerLeg, address = "" },
        new TrackerMotorEndpoint { location = TrackerWearLocation.RightThigh, address = "" },
        new TrackerMotorEndpoint { location = TrackerWearLocation.RightLowerLeg, address = "" },
        new TrackerMotorEndpoint { location = TrackerWearLocation.Chest, address = "" },
        new TrackerMotorEndpoint { location = TrackerWearLocation.Abdomen, address = "" }
    };

    private readonly Dictionary<TrackerWearLocation, string> endpointAddresses =
        new Dictionary<TrackerWearLocation, string>();
    private readonly Dictionary<TrackerWearLocation, string> fallbackEndpointAddresses =
        new Dictionary<TrackerWearLocation, string>();
    private readonly HashSet<TrackerWearLocation> automaticallyDiscoveredLocations =
        new HashSet<TrackerWearLocation>();
    private readonly Dictionary<TrackerWearLocation, int> endpointDurations =
        new Dictionary<TrackerWearLocation, int>();
    private readonly Dictionary<TrackerWearLocation, UnityWebRequest> inFlightRequests =
        new Dictionary<TrackerWearLocation, UnityWebRequest>();
    private readonly HashSet<TrackerWearLocation> missingAddressWarnings =
        new HashSet<TrackerWearLocation>();
    private Coroutine discoveryCoroutine;
    private string lastLoggedDiscoveryStatus;

    public string DiscoveryStatus { get; private set; } =
        "Waiting for SlimeVR tracker bindings";

    private void Awake()
    {
        RebuildEndpointMap();
    }

    private void OnEnable()
    {
        if (Application.isPlaying && discoveryCoroutine == null)
        {
            discoveryCoroutine = StartCoroutine(PollTrackerBindings());
        }
    }

    private void OnDisable()
    {
        discoveryCoroutine = null;
        StopAllHaptics();
    }

    private void OnDestroy()
    {
        StopAllHaptics();
    }

    private void OnApplicationPause(bool paused)
    {
        if (paused)
        {
            StopAllHaptics();
        }
    }

    public void SetEnabled(bool enabled)
    {
        enabledForDemo = enabled;
        if (!enabled)
        {
            StopAllHaptics();
        }
    }

    public void ConfigureEndpoint(TrackerWearLocation location, string address)
    {
        if (endpointAddresses.Count == 0)
        {
            RebuildEndpointMap();
        }

        string normalizedAddress = address == null ? string.Empty : address.Trim();
        fallbackEndpointAddresses[location] = normalizedAddress;
        if (!automaticallyDiscoveredLocations.Contains(location))
        {
            endpointAddresses[location] = normalizedAddress;
        }
    }

    public void ConfigureEndpoint(
        TrackerWearLocation location,
        string address,
        int durationMs)
    {
        ConfigureEndpoint(location, address);
        endpointDurations[location] = Mathf.Clamp(durationMs, 1, 60000);
    }

    public void ConfigureDuration(TrackerWearLocation location, int durationMs)
    {
        endpointDurations[location] = Mathf.Clamp(durationMs, 1, 60000);
    }

    public string GetEndpointAddress(TrackerWearLocation location)
    {
        return endpointAddresses.TryGetValue(location, out string address)
            ? address
            : string.Empty;
    }

    public bool IsAutomaticallyDiscovered(TrackerWearLocation location)
    {
        return automaticallyDiscoveredLocations.Contains(location);
    }

    /// <summary>Triggers one pulse for a discrete hit result.</summary>
    public void PulseOnce(TrackerWearLocation location)
    {
        if (!enabledForDemo || !isActiveAndEnabled)
        {
            return;
        }

        if (endpointAddresses.Count == 0)
        {
            RebuildEndpointMap();
        }

        TrySendPulse(location);
    }

    public void StopAllHaptics()
    {
        foreach (UnityWebRequest request in inFlightRequests.Values)
        {
            request.Abort();
        }

        inFlightRequests.Clear();
    }

    private void RebuildEndpointMap()
    {
        endpointAddresses.Clear();
        fallbackEndpointAddresses.Clear();
        automaticallyDiscoveredLocations.Clear();
        endpointDurations.Clear();
        if (endpoints == null)
        {
            return;
        }

        foreach (TrackerMotorEndpoint endpoint in endpoints)
        {
            if (endpoint != null)
            {
                string address = endpoint.address == null
                    ? string.Empty
                    : endpoint.address.Trim();
                endpointAddresses[endpoint.location] = address;
                fallbackEndpointAddresses[endpoint.location] = address;
                endpointDurations[endpoint.location] =
                    Mathf.Clamp(endpoint.durationMs, 1, 60000);
            }
        }
    }

    private IEnumerator PollTrackerBindings()
    {
        while (enabled)
        {
            yield return RefreshTrackerBindings();
            yield return new WaitForSecondsRealtime(
                Mathf.Max(0.5f, discoveryIntervalSeconds));
        }
    }

    private IEnumerator RefreshTrackerBindings()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        string responseJson = null;
        try
        {
            responseJson = ReadAndroidTrackerBindings();
        }
        catch (Exception exception)
        {
            ClearAutomaticallyDiscoveredEndpoints();
            DiscoveryStatus = "Android tracker binding provider is unavailable";
            Debug.LogWarning(
                $"[Haptics] Failed to read Android tracker bindings: {exception.Message}",
                this);
            yield break;
        }

        ParseAndApplyTrackerBindings(responseJson, "Android SlimeVR");
        yield break;
#else
        UnityWebRequest request = UnityWebRequest.Get(discoveryUrl);
        request.timeout = Mathf.Max(1, requestTimeoutSeconds);
        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            request.Dispose();
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            if (TryReadDesktopSlimeVrBindings(out SlimeVrHapticTrackerSnapshot[] trackers))
            {
                ApplyDiscoveredTrackers(trackers);
                DiscoveryStatus = DiscoveryStatus.Replace(
                    "Discovered",
                    "Discovered from desktop SlimeVR");
                LogDiscoveryStatusOnce();
                yield break;
            }
#endif
            ClearAutomaticallyDiscoveredEndpoints();
            DiscoveryStatus = "SlimeVR tracker bindings are unavailable";
            LogDiscoveryStatusOnce();
            yield break;
        }

        string responseJson = request.downloadHandler.text;
        request.Dispose();
        ParseAndApplyTrackerBindings(responseJson, "Modified SlimeVR");
#endif
    }

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
    private static bool TryReadDesktopSlimeVrBindings(
        out SlimeVrHapticTrackerSnapshot[] trackers)
    {
        trackers = null;
        try
        {
            string slimeVrDirectory = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "dev.slimevr.SlimeVR");
            string configPath = Path.Combine(slimeVrDirectory, "vrconfig.yml");
            string logDirectory = Path.Combine(slimeVrDirectory, "logs");
            if (!File.Exists(configPath) || !Directory.Exists(logDirectory))
            {
                return false;
            }

            Dictionary<string, string> bodyPositions =
                ReadDesktopBodyAssignments(configPath);
            Dictionary<string, string> trackerAddresses =
                ReadDesktopTrackerAddresses(logDirectory);
            var snapshots = new List<SlimeVrHapticTrackerSnapshot>();

            foreach (KeyValuePair<string, string> binding in bodyPositions)
            {
                if (!trackerAddresses.TryGetValue(binding.Key, out string address))
                {
                    continue;
                }

                snapshots.Add(new SlimeVrHapticTrackerSnapshot
                {
                    hardwareId = binding.Key,
                    name = $"udp://{binding.Key}",
                    ip = address,
                    bodyPosition = binding.Value,
                    status = "OK",
                    connected = true
                });
            }

            trackers = snapshots.ToArray();
            return trackers.Length > 0;
        }
        catch (Exception exception)
        {
            Debug.LogWarning(
                $"[Haptics] Failed to read desktop SlimeVR bindings: {exception.Message}");
            return false;
        }
    }

    private static Dictionary<string, string> ReadDesktopBodyAssignments(
        string configPath)
    {
        var result = new Dictionary<string, string>(
            StringComparer.OrdinalIgnoreCase);
        bool readingTrackers = false;
        string currentHardwareId = null;

        foreach (string line in ReadLinesShared(configPath))
        {
            if (!readingTrackers)
            {
                readingTrackers = line.Trim() == "trackers:" &&
                    line.Length == line.Trim().Length;
                continue;
            }

            if (line.Length > 0 && !char.IsWhiteSpace(line[0]))
            {
                break;
            }

            string trimmed = line.Trim();
            if (trimmed.StartsWith("udp://", StringComparison.OrdinalIgnoreCase) &&
                trimmed.EndsWith(":", StringComparison.Ordinal))
            {
                int sensorSuffix = trimmed.LastIndexOf("/0:", StringComparison.Ordinal);
                currentHardwareId = sensorSuffix > 6
                    ? trimmed.Substring(6, sensorSuffix - 6)
                    : null;
                continue;
            }

            if (currentHardwareId == null ||
                !trimmed.StartsWith("designation:", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            string designation = trimmed.Substring("designation:".Length)
                .Trim()
                .Trim('"');
            if (!string.IsNullOrWhiteSpace(designation) &&
                !string.Equals(designation, "null", StringComparison.OrdinalIgnoreCase))
            {
                result[currentHardwareId] = designation;
            }
        }

        return result;
    }

    private static IEnumerable<string> ReadLinesShared(string path)
    {
        using (FileStream stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.ReadWrite | FileShare.Delete))
        using (StreamReader reader = new StreamReader(stream))
        {
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                yield return line;
            }
        }
    }

    private static Dictionary<string, string> ReadDesktopTrackerAddresses(
        string logDirectory)
    {
        string[] logFiles = Directory.GetFiles(
            logDirectory,
            "slimevr-server*.log",
            SearchOption.TopDirectoryOnly);
        if (logFiles.Length == 0)
        {
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        }

        string latestLog = logFiles[0];
        DateTime latestWriteTime = File.GetLastWriteTimeUtc(latestLog);
        for (int i = 1; i < logFiles.Length; i++)
        {
            DateTime writeTime = File.GetLastWriteTimeUtc(logFiles[i]);
            if (writeTime > latestWriteTime)
            {
                latestLog = logFiles[i];
                latestWriteTime = writeTime;
            }
        }

        var result = new Dictionary<string, string>(
            StringComparer.OrdinalIgnoreCase);
        Regex connectedPattern = new Regex(
            @"connected from address /(?<ip>(?:\d{1,3}\.){3}\d{1,3}):",
            RegexOptions.Compiled);
        Regex hardwarePattern = new Regex(
            @"\bmac:\s*(?<hardware>(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2})",
            RegexOptions.Compiled);
        string pendingAddress = null;

        using (FileStream stream = new FileStream(
            latestLog,
            FileMode.Open,
            FileAccess.Read,
            FileShare.ReadWrite | FileShare.Delete))
        using (StreamReader reader = new StreamReader(stream))
        {
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                Match connected = connectedPattern.Match(line);
                if (connected.Success)
                {
                    pendingAddress = connected.Groups["ip"].Value;
                    continue;
                }

                if (pendingAddress == null)
                {
                    continue;
                }

                Match hardware = hardwarePattern.Match(line.Trim());
                if (hardware.Success)
                {
                    result[hardware.Groups["hardware"].Value] = pendingAddress;
                    pendingAddress = null;
                }
            }
        }

        return result;
    }
#endif

    private void ParseAndApplyTrackerBindings(string responseJson, string sourceName)
    {
        SlimeVrHapticTrackerResponse response = null;
        try
        {
            response = JsonUtility.FromJson<SlimeVrHapticTrackerResponse>(
                responseJson);
        }
        catch (Exception exception)
        {
            Debug.LogWarning($"[Haptics] Failed to parse SlimeVR tracker bindings: {exception.Message}", this);
        }

        if (response == null || response.apiVersion != 1 || response.trackers == null)
        {
            ClearAutomaticallyDiscoveredEndpoints();
            DiscoveryStatus = $"{sourceName} tracker binding response is invalid";
            return;
        }

        ApplyDiscoveredTrackers(response.trackers);
    }

#if UNITY_ANDROID && !UNITY_EDITOR
    private static string ReadAndroidTrackerBindings()
    {
        using (AndroidJavaClass unityPlayer =
            new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
        using (AndroidJavaObject activity =
            unityPlayer.GetStatic<AndroidJavaObject>("currentActivity"))
        using (AndroidJavaObject resolver = activity.Call<AndroidJavaObject>("getContentResolver"))
        using (AndroidJavaClass uriClass = new AndroidJavaClass("android.net.Uri"))
        using (AndroidJavaObject uri = uriClass.CallStatic<AndroidJavaObject>(
            "parse",
            "content://com.metaspine.mobile.trackerbindings"))
        using (AndroidJavaObject extras = new AndroidJavaObject("android.os.Bundle"))
        using (AndroidJavaObject result = resolver.Call<AndroidJavaObject>(
            "call",
            uri,
            "getTrackers",
            string.Empty,
            extras))
        {
            if (result == null)
            {
                throw new InvalidOperationException(
                    "Android tracker binding provider returned no result");
            }

            return result.Call<string>("getString", "tracker_bindings_json");
        }
    }
#endif

    private void ApplyDiscoveredTrackers(SlimeVrHapticTrackerSnapshot[] trackers)
    {
        var discovered = new Dictionary<TrackerWearLocation, string>();
        int unassignedCount = 0;

        foreach (SlimeVrHapticTrackerSnapshot tracker in trackers)
        {
            if (tracker == null || !tracker.connected || string.IsNullOrWhiteSpace(tracker.ip))
            {
                continue;
            }

            if (!TryMapBodyPosition(tracker.bodyPosition, out TrackerWearLocation location))
            {
                if (string.IsNullOrWhiteSpace(tracker.bodyPosition))
                {
                    unassignedCount++;
                }
                continue;
            }

            if (!discovered.ContainsKey(location))
            {
                discovered.Add(location, tracker.ip.Trim());
            }
        }

        automaticallyDiscoveredLocations.Clear();
        endpointAddresses.Clear();
        foreach (KeyValuePair<TrackerWearLocation, string> fallback in fallbackEndpointAddresses)
        {
            endpointAddresses[fallback.Key] = fallback.Value;
        }
        foreach (KeyValuePair<TrackerWearLocation, string> binding in discovered)
        {
            endpointAddresses[binding.Key] = binding.Value;
            automaticallyDiscoveredLocations.Add(binding.Key);
        }

        DiscoveryStatus = $"Discovered {discovered.Count}/10 assigned trackers";
        if (unassignedCount > 0)
        {
            DiscoveryStatus += $"; {unassignedCount} trackers have no body assignment";
        }

        missingAddressWarnings.Clear();
        LogDiscoveryStatusOnce();
    }

    private void LogDiscoveryStatusOnce()
    {
        if (string.Equals(
            lastLoggedDiscoveryStatus,
            DiscoveryStatus,
            StringComparison.Ordinal))
        {
            return;
        }

        lastLoggedDiscoveryStatus = DiscoveryStatus;
        Debug.Log($"[Haptics] {DiscoveryStatus}.", this);
    }

    private void ClearAutomaticallyDiscoveredEndpoints()
    {
        automaticallyDiscoveredLocations.Clear();
        endpointAddresses.Clear();
        foreach (KeyValuePair<TrackerWearLocation, string> fallback in fallbackEndpointAddresses)
        {
            endpointAddresses[fallback.Key] = fallback.Value;
        }
    }

    private static bool TryMapBodyPosition(
        string bodyPosition,
        out TrackerWearLocation location)
    {
        string normalized = (bodyPosition ?? string.Empty).Trim().ToLowerInvariant();
        switch (normalized)
        {
            case "body:upper_chest":
            case "body:chest":
            case "upper_chest":
            case "chest":
                location = TrackerWearLocation.Chest;
                return true;
            case "body:waist":
            case "body:hip":
            case "waist":
            case "hip":
                location = TrackerWearLocation.Abdomen;
                return true;
            case "body:left_upper_arm":
            case "left_upper_arm":
            case "leftupperarm":
                location = TrackerWearLocation.LeftUpperArm;
                return true;
            case "body:right_upper_arm":
            case "right_upper_arm":
            case "rightupperarm":
                location = TrackerWearLocation.RightUpperArm;
                return true;
            case "body:left_lower_arm":
            case "left_lower_arm":
            case "left_forearm":
            case "leftforearm":
                location = TrackerWearLocation.LeftForearm;
                return true;
            case "body:right_lower_arm":
            case "right_lower_arm":
            case "right_forearm":
            case "rightforearm":
                location = TrackerWearLocation.RightForearm;
                return true;
            case "body:left_upper_leg":
            case "left_upper_leg":
            case "leftthigh":
                location = TrackerWearLocation.LeftThigh;
                return true;
            case "body:right_upper_leg":
            case "right_upper_leg":
            case "rightthigh":
                location = TrackerWearLocation.RightThigh;
                return true;
            case "body:left_lower_leg":
            case "left_lower_leg":
            case "left_shin":
                location = TrackerWearLocation.LeftLowerLeg;
                return true;
            case "body:right_lower_leg":
            case "right_lower_leg":
            case "right_shin":
                location = TrackerWearLocation.RightLowerLeg;
                return true;
            default:
                location = default;
                return false;
        }
    }

    private void TrySendPulse(TrackerWearLocation location)
    {
        if (!endpointAddresses.TryGetValue(location, out string address) ||
            string.IsNullOrWhiteSpace(address))
        {
            if (missingAddressWarnings.Add(location))
            {
                Debug.LogWarning(
                    $"[Haptics] No Tracker address is assigned for {location}. " +
                    "Automatic discovery must return a connected tracker with a body position.",
                    this);
            }
            return;
        }

        if (inFlightRequests.ContainsKey(location))
        {
            return;
        }

        int durationMs = endpointDurations.TryGetValue(location, out int configuredDuration)
            ? configuredDuration
            : pulseDurationMs;
        Debug.Log(
            $"[Haptics] HIT -> {location} ({address}), {durationMs} ms.",
            this);
        StartCoroutine(SendPulse(location, address, durationMs));
    }

    private IEnumerator SendPulse(TrackerWearLocation location, string address, int durationMs)
    {
        string normalizedAddress = address.Trim().TrimEnd('/');
        if (normalizedAddress.StartsWith("http://", StringComparison.OrdinalIgnoreCase))
        {
            normalizedAddress = normalizedAddress.Substring("http://".Length);
        }
        else if (normalizedAddress.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
        {
            normalizedAddress = normalizedAddress.Substring("https://".Length);
        }

        string uri = string.Format(
            "http://{0}/motor?duration_ms={1}",
            normalizedAddress,
            Mathf.Clamp(durationMs, 1, 60000));
        UnityWebRequest request = UnityWebRequest.Get(uri);
        request.timeout = Mathf.Max(1, requestTimeoutSeconds);
        inFlightRequests[location] = request;

        yield return request.SendWebRequest();

        bool isCurrentRequest = inFlightRequests.TryGetValue(
            location,
            out UnityWebRequest trackedRequest) && trackedRequest == request;
        if (isCurrentRequest)
        {
            inFlightRequests.Remove(location);
        }

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogWarning(
                $"[Haptics] Tracker {location} request failed: {request.error}",
                this);
        }

        request.Dispose();
    }
}
