using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Threading.Tasks;
using UnityEngine;
using Debug = UnityEngine.Debug;

/// <summary>
/// Starts the local MobilePose Python bridge for a playable Unity scene.
/// A process is only launched when the configured local WebSocket port is not
/// already listening, so a manually started bridge remains supported.
/// Every calibration (MobilePoseCalibrationController.BeginCalibration) restarts the bridge through
/// RestartBridge, so a calibration and the recording it starts always run the current Python code.
/// </summary>
[DisallowMultipleComponent]
[DefaultExecutionOrder(-900)]
public sealed class MobilePosePythonLauncher : MonoBehaviour
{
    [SerializeField] private bool autoStart = true;
    [Tooltip("Optional Python executable. Empty checks MOTION_RECOVER_PYTHON, the documented local conda env, then PATH.")]
    [SerializeField] private string pythonExecutable = "";
    [Tooltip("Also open the browser diagnostic server on port 8765.")]
    [SerializeField] private bool startWebViewer;

    public const float RestartConnectTimeoutSeconds = 120f;   // model loading + SlimeVR connection

    private static Process ownedProcess;
    private static bool launchAttempted;
    // Launched by Unity and not yet used by a calibration: the first calibration after Play needs no restart.
    private static bool freshProcess;
    private bool exitReported;

    public bool IsRestarting { get; private set; }
    public string RestartError { get; private set; }

    /// <summary>True once: the bridge was just launched by Unity and no calibration has used it yet.</summary>
    public static bool ConsumeFreshProcess()
    {
        bool fresh = freshProcess && ownedProcess != null && !ownedProcess.HasExited;
        freshProcess = false;
        return fresh;
    }

    private IEnumerator Start()
    {
        if (!autoStart || Environment.GetEnvironmentVariable("MOTION_RECOVER_DISABLE_AUTOSTART") == "1")
            yield break;

        MobilePoseWebSocketSource source = GetComponent<MobilePoseWebSocketSource>();
        if (source == null) source = FindObjectOfType<MobilePoseWebSocketSource>(true);
        if (source == null || !IsLocalHost(source.ServerHost)) yield break;

        // Give an already-running bridge a moment to accept connections before
        // deciding that Unity owns a new Python process.
        yield return new WaitForSecondsRealtime(0.25f);
        if (source.IsConnected || IsPortListening(source.ServerHost, source.ServerPort, 150)) yield break;
        if (ownedProcess != null && !ownedProcess.HasExited) yield break;
        if (launchAttempted) yield break;
        launchAttempted = true;
        if (Launch(source)) freshProcess = true;
        else launchAttempted = false;
    }

    private bool Launch(MobilePoseWebSocketSource source)
    {
        string pythonRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "..", "python"));
        string executable = ResolvePythonExecutable();
        string arguments = "-m production.run --model " + QuoteArgument(source.SelectedModelId);
        if (startWebViewer) arguments += " --web";

        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = executable,
                Arguments = arguments,
                WorkingDirectory = pythonRoot,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            ownedProcess = Process.Start(startInfo);
            if (ownedProcess == null) throw new InvalidOperationException("Process.Start returned null");
            exitReported = false;
            Debug.Log("Started MobilePose Python bridge: " + executable + " " + arguments);
            return true;
        }
        catch (Exception exception)
        {
            Debug.LogError("Could not start MobilePose Python bridge. Set MOTION_RECOVER_PYTHON or the launcher Python path. " +
                           exception.Message, this);
            return false;
        }
    }

    /// <summary>Stop whatever Python bridge listens on the source's port (Unity's own process, or one whose
    /// handle was lost after a script recompile), start a new one and wait until Unity is connected to it.
    /// Sets RestartError on failure.  `report` receives progress text for the calibration panel.</summary>
    public IEnumerator RestartBridge(MobilePoseWebSocketSource source, Action<string> report)
    {
        IsRestarting = true;
        RestartError = null;
        try
        {
            if (source == null || !IsLocalHost(source.ServerHost))
            {
                RestartError = "Python 不在本机，无法自动重启";
                yield break;
            }
            report?.Invoke("正在关闭旧的 Python 进程…");
            KillOwnedProcess();
            Task<int> lookup = Task.Run(() => FindListeningPid(source.ServerPort));
            while (!lookup.IsCompleted) yield return null;
            int pid = lookup.Status == TaskStatus.RanToCompletion ? lookup.Result : -1;
            if (pid > 0)
            {
                string killError = KillPythonProcess(pid);
                if (killError != null)
                {
                    RestartError = killError;
                    yield break;
                }
            }
            float deadline = Time.realtimeSinceStartup + 8f;
            while (IsPortListening(source.ServerHost, source.ServerPort, 150))
            {
                if (Time.realtimeSinceStartup > deadline)
                {
                    RestartError = "端口 " + source.ServerPort + " 仍被占用，旧的 Python 未能关闭";
                    yield break;
                }
                yield return new WaitForSecondsRealtime(0.2f);
            }
            while (source.IsConnected && Time.realtimeSinceStartup < deadline)
                yield return null;
            source.ResetBridgeState();

            report?.Invoke("正在启动新的 Python（加载模型并连接 SlimeVR）…");
            launchAttempted = true;
            freshProcess = false;
            if (!Launch(source))
            {
                launchAttempted = false;
                RestartError = "无法启动 Python，请检查 MOTION_RECOVER_PYTHON 或启动器中的 Python 路径";
                yield break;
            }
            // The WebSocket opens right after the model loads, but Python handles calibration commands only
            // once its main loop runs (SlimeVR layout resolved): wait for its pose frames, not the socket.
            deadline = Time.realtimeSinceStartup + RestartConnectTimeoutSeconds;
            while (!IsBridgeReady(source))
            {
                if (ownedProcess == null || ownedProcess.HasExited)
                {
                    RestartError = "新的 Python 启动后退出了，请查看 Unity 日志";
                    yield break;
                }
                if (Time.realtimeSinceStartup > deadline)
                {
                    RestartError = source.IsConnected
                        ? "Python 已启动，但一直没有收到姿态数据（SlimeVR 是否已连接、追踪器是否已绑定？）"
                        : "等待新的 Python 启动超时";
                    yield break;
                }
                report?.Invoke(source.IsConnected
                    ? "Python 已启动，正在等待 SlimeVR 数据…"
                    : "正在启动新的 Python（加载模型），请稍候…");
                yield return new WaitForSecondsRealtime(0.2f);
            }
            Debug.Log("MobilePose Python bridge restarted for calibration (pid " + ownedProcess.Id + ").");
        }
        finally
        {
            IsRestarting = false;
        }
    }

    /// <summary>Python is running its main loop: connected and streaming pose frames (~0.5 s of them).</summary>
    public static bool IsBridgeReady(MobilePoseWebSocketSource source) =>
        source != null && source.IsConnected && source.PoseFramesSinceReset >= ReadyPoseFrames;

    public const int ReadyPoseFrames = 15;

    private void Update()
    {
        if (!exitReported && ownedProcess != null && ownedProcess.HasExited)
        {
            exitReported = true;
            Debug.LogError("MobilePose Python bridge exited with code " + ownedProcess.ExitCode +
                           ". Check the Unity Editor log or start production.run in a terminal for details.", this);
        }
    }

    private static void KillOwnedProcess()
    {
        if (ownedProcess == null) return;
        try
        {
            if (!ownedProcess.HasExited)
            {
                ownedProcess.Kill();
                ownedProcess.WaitForExit(3000);
            }
        }
        catch (InvalidOperationException) { }
        catch (System.ComponentModel.Win32Exception) { }
        finally
        {
            ownedProcess.Dispose();
            ownedProcess = null;
        }
    }

    /// <summary>Kill the process listening on the bridge port, but only if it is a Python process.
    /// Returns an error message, or null.</summary>
    private static string KillPythonProcess(int pid)
    {
        try
        {
            using (Process process = Process.GetProcessById(pid))
            {
                if (!process.ProcessName.StartsWith("python", StringComparison.OrdinalIgnoreCase))
                    return "端口被非 Python 进程占用（" + process.ProcessName + "，pid " + pid + "），未关闭";
                process.Kill();
                process.WaitForExit(3000);
                Debug.Log("Stopped previous MobilePose Python bridge (pid " + pid + ").");
            }
        }
        catch (ArgumentException) { }                     // already exited
        catch (InvalidOperationException) { }
        catch (System.ComponentModel.Win32Exception exception)
        {
            return "无法关闭旧的 Python（pid " + pid + "）：" + exception.Message;
        }
        return null;
    }

    /// <summary>PID listening on the TCP port (from netstat -ano), or -1.</summary>
    private static int FindListeningPid(int port)
    {
        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = "netstat",
                Arguments = "-ano -p TCP",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                CreateNoWindow = true
            };
            using (Process netstat = Process.Start(startInfo))
            {
                string output = netstat.StandardOutput.ReadToEnd();
                netstat.WaitForExit(3000);
                string suffix = ":" + port;
                foreach (string line in output.Split('\n'))
                {
                    // TCP  127.0.0.1:21200  0.0.0.0:0  LISTENING  39664   (state text may be localized)
                    string[] parts = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 5 && parts[0] == "TCP" &&
                        parts[1].EndsWith(suffix, StringComparison.Ordinal) &&
                        parts[2].EndsWith(":0", StringComparison.Ordinal) &&
                        int.TryParse(parts[parts.Length - 1], out int pid) && pid > 0)
                        return pid;
                }
            }
        }
        catch (Exception exception)
        {
            Debug.LogWarning("netstat lookup of the Python bridge failed: " + exception.Message);
        }
        return -1;
    }

    private string ResolvePythonExecutable()
    {
        if (!string.IsNullOrWhiteSpace(pythonExecutable))
            return pythonExecutable.Trim();

        string configured = Environment.GetEnvironmentVariable("MOTION_RECOVER_PYTHON");
        if (!string.IsNullOrWhiteSpace(configured)) return configured.Trim();

        const string documentedEnvironment = "E:/dyh/conda_envs/mobileposer/python.exe";
        if (File.Exists(documentedEnvironment)) return documentedEnvironment;

        string condaPrefix = Environment.GetEnvironmentVariable("CONDA_PREFIX");
        if (!string.IsNullOrWhiteSpace(condaPrefix))
        {
            string condaPython = Path.Combine(condaPrefix, "python.exe");
            if (File.Exists(condaPython)) return condaPython;
        }
        return "python";
    }

    private static bool IsLocalHost(string host)
    {
        return string.IsNullOrWhiteSpace(host) || host == "127.0.0.1" ||
               host.Equals("localhost", StringComparison.OrdinalIgnoreCase) || host == "::1";
    }

    private static bool IsPortListening(string host, int port, int timeoutMilliseconds)
    {
        try
        {
            using (var client = new TcpClient())
            {
                IAsyncResult result = client.BeginConnect(host, port, null, null);
                bool connected = result.AsyncWaitHandle.WaitOne(timeoutMilliseconds);
                if (!connected) return false;
                client.EndConnect(result);
                return client.Connected;
            }
        }
        catch (SocketException) { return false; }
        catch (ObjectDisposedException) { return false; }
    }

    private static string QuoteArgument(string value)
    {
        return "\"" + (value ?? string.Empty).Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    }

    private void OnApplicationQuit()
    {
        KillOwnedProcess();
        launchAttempted = false;
        freshProcess = false;
    }
}
