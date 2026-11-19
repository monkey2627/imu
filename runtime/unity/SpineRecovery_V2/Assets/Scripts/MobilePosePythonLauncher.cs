using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using UnityEngine;
using Debug = UnityEngine.Debug;

/// <summary>
/// Starts the local MobilePose Python bridge for a playable Unity scene.
/// A process is only launched when the configured local WebSocket port is not
/// already listening, so a manually started bridge remains supported.
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

    private static Process ownedProcess;
    private static bool launchAttempted;
    private bool exitReported;

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
            Debug.Log("Started MobilePose Python bridge: " + executable + " " + arguments);
        }
        catch (Exception exception)
        {
            launchAttempted = false;
            Debug.LogError("Could not start MobilePose Python bridge. Set MOTION_RECOVER_PYTHON or the launcher Python path. " +
                           exception.Message, this);
        }
    }

    private void Update()
    {
        if (!exitReported && ownedProcess != null && ownedProcess.HasExited)
        {
            exitReported = true;
            Debug.LogError("MobilePose Python bridge exited with code " + ownedProcess.ExitCode +
                           ". Check the Unity Editor log or start production.run in a terminal for details.", this);
        }
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
        if (ownedProcess == null)
        {
            launchAttempted = false;
            return;
        }
        try
        {
            if (!ownedProcess.HasExited) ownedProcess.Kill();
        }
        catch (InvalidOperationException) { }
        catch (System.ComponentModel.Win32Exception) { }
        finally
        {
            ownedProcess.Dispose();
            ownedProcess = null;
            launchAttempted = false;
        }
    }
}
