using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public sealed class ReplayRepository
{
    public const string RepositoryDirectoryName = "Replays";
    public const string ManifestFileName = "manifest.json";
    public const string CommitMarkerFileName = "commit.ok";
    public const string LatestCompletedFileName = "latest_completed.json";

    [Serializable]
    private sealed class LatestCompletedIndex
    {
        public int formatVersion;
        public string sessionId;
        public string payloadSha256;
    }

    public string RootPath { get; }

    public ReplayRepository(string rootPath = null)
    {
        RootPath = string.IsNullOrWhiteSpace(rootPath)
            ? Path.Combine(Application.persistentDataPath, RepositoryDirectoryName)
            : Path.GetFullPath(rootPath);
    }

    public Task<ReplayManifest> CommitCompletedAsync(
        ReplayManifest manifest,
        IReadOnlyList<ReplayFrame> frames)
    {
        ReplayManifest manifestSnapshot = manifest?.Clone();
        var frameSnapshot = CloneFrames(frames);
        return Task.Run(() => CommitCompleted(manifestSnapshot, frameSnapshot));
    }

    public ReplayManifest CommitCompleted(
        ReplayManifest manifest,
        IReadOnlyList<ReplayFrame> frames)
    {
        if (manifest == null)
        {
            throw new ArgumentNullException(nameof(manifest));
        }

        Directory.CreateDirectory(RootPath);
        ReplayManifest committedManifest = PrepareManifest(manifest, frames, true);
        string temporaryDirectory = GetRecordingDirectory(committedManifest.sessionId);
        string finalDirectory = GetSessionDirectory(committedManifest.sessionId);
        EnsurePublicationTargetsAreNew(temporaryDirectory, finalDirectory);
        Directory.CreateDirectory(temporaryDirectory);

        try
        {
            string payloadPath = Path.Combine(
                temporaryDirectory,
                ReplayManifest.DefaultPayloadFile);
            ReplayBinarySerializer.WriteFile(payloadPath, frames);
            committedManifest.payloadSha256 =
                ReplayBinarySerializer.ComputeFileSha256(payloadPath);

            string manifestPath = Path.Combine(temporaryDirectory, ManifestFileName);
            WriteNewTextFile(
                manifestPath,
                JsonUtility.ToJson(committedManifest, false));

            ReplayLoadResult validation = LoadFromDirectory(
                temporaryDirectory,
                committedManifest.sessionId,
                requireCommitMarker: false,
                requireCompleted: true);
            if (!validation.IsSuccess)
            {
                throw new InvalidDataException(
                    $"Replay failed pre-commit validation: {validation.Error}");
            }

            WriteNewTextFile(
                Path.Combine(temporaryDirectory, CommitMarkerFileName),
                BuildCommitMarker(committedManifest));

            Directory.Move(temporaryDirectory, finalDirectory);
            UpdateLatestCompleted(committedManifest);
            return committedManifest;
        }
        catch
        {
            // A temporary directory is intentionally retained as diagnostic
            // evidence. It has no commit marker and can never become latest.
            throw;
        }
    }

    public ReplayManifest SaveIncomplete(
        ReplayManifest manifest,
        IReadOnlyList<ReplayFrame> frames)
    {
        if (manifest == null)
        {
            throw new ArgumentNullException(nameof(manifest));
        }

        Directory.CreateDirectory(RootPath);
        ReplayManifest partialManifest = PrepareManifest(manifest, frames, false);
        string temporaryDirectory = GetRecordingDirectory(partialManifest.sessionId);
        if (Directory.Exists(temporaryDirectory))
        {
            throw new IOException(
                $"Replay recording directory already exists: {temporaryDirectory}");
        }

        Directory.CreateDirectory(temporaryDirectory);
        string payloadPath = Path.Combine(
            temporaryDirectory,
            ReplayManifest.DefaultPayloadFile);
        ReplayBinarySerializer.WriteFile(payloadPath, frames);
        partialManifest.payloadSha256 =
            ReplayBinarySerializer.ComputeFileSha256(payloadPath);
        WriteNewTextFile(
            Path.Combine(temporaryDirectory, ManifestFileName),
            JsonUtility.ToJson(partialManifest, false));
        return partialManifest;
    }

    public ReplayLoadResult Load(string sessionId)
    {
        if (string.IsNullOrWhiteSpace(sessionId))
        {
            return ReplayLoadResult.Failure(
                ReplayLoadStatus.Unavailable,
                "No replay sessionId was supplied.");
        }

        try
        {
            return LoadFromDirectory(
                GetSessionDirectory(sessionId),
                sessionId,
                requireCommitMarker: true,
                requireCompleted: true);
        }
        catch (Exception exception)
        {
            return ReplayLoadResult.Failure(
                ClassifyException(exception),
                $"Replay '{sessionId}' could not be loaded: {exception.Message}");
        }
    }

    public ReplayLoadResult LoadLatestCompleted()
    {
        string indexPath = Path.Combine(RootPath, LatestCompletedFileName);
        if (!File.Exists(indexPath))
        {
            return ReplayLoadResult.Failure(
                ReplayLoadStatus.Unavailable,
                "No completed replay is available.");
        }

        try
        {
            LatestCompletedIndex index = JsonUtility.FromJson<LatestCompletedIndex>(
                File.ReadAllText(indexPath, Encoding.UTF8));
            if (index == null ||
                index.formatVersion != ReplayManifest.CurrentFormatVersion ||
                string.IsNullOrWhiteSpace(index.sessionId) ||
                string.IsNullOrWhiteSpace(index.payloadSha256))
            {
                return ReplayLoadResult.Failure(
                    ReplayLoadStatus.Corrupt,
                    "The latest completed replay index is invalid.");
            }

            ReplayLoadResult result = Load(index.sessionId);
            if (result.IsSuccess && !string.Equals(
                    result.Session.Manifest.payloadSha256,
                    index.payloadSha256,
                    StringComparison.OrdinalIgnoreCase))
            {
                return ReplayLoadResult.Failure(
                    ReplayLoadStatus.Corrupt,
                    "The latest replay index checksum does not match its session.");
            }

            return result;
        }
        catch (Exception exception)
        {
            return ReplayLoadResult.Failure(
                ClassifyException(exception),
                $"The latest completed replay index could not be loaded: {exception.Message}");
        }
    }

    public string GetSessionDirectory(string sessionId)
    {
        ValidateSessionId(sessionId);
        return Path.Combine(RootPath, sessionId);
    }

    public string GetRecordingDirectory(string sessionId)
    {
        ValidateSessionId(sessionId);
        return Path.Combine(RootPath, $"recording-{sessionId}.tmp");
    }

    private ReplayLoadResult LoadFromDirectory(
        string directory,
        string expectedSessionId,
        bool requireCommitMarker,
        bool requireCompleted)
    {
        if (!Directory.Exists(directory))
        {
            return ReplayLoadResult.Failure(
                ReplayLoadStatus.Unavailable,
                $"Replay directory does not exist: {directory}");
        }

        string manifestPath = Path.Combine(directory, ManifestFileName);
        if (!File.Exists(manifestPath))
        {
            return ReplayLoadResult.Failure(
                ReplayLoadStatus.Corrupt,
                "Replay manifest is missing.");
        }

        ReplayManifest manifest;
        try
        {
            manifest = JsonUtility.FromJson<ReplayManifest>(
                File.ReadAllText(manifestPath, Encoding.UTF8));
        }
        catch (Exception exception)
        {
            return ReplayLoadResult.Failure(
                ReplayLoadStatus.Corrupt,
                $"Replay manifest JSON is invalid: {exception.Message}");
        }

        if (manifest == null)
        {
            return ReplayLoadResult.Failure(
                ReplayLoadStatus.Corrupt,
                "Replay manifest is empty.");
        }

        if (manifest.formatVersion != ReplayManifest.CurrentFormatVersion)
        {
            return ReplayLoadResult.Failure(
                ReplayLoadStatus.UnsupportedVersion,
                $"Replay format {manifest.formatVersion} is not supported.");
        }

        if (!string.Equals(
                manifest.sessionId,
                expectedSessionId,
                StringComparison.Ordinal))
        {
            return ReplayLoadResult.Failure(
                ReplayLoadStatus.Corrupt,
                "Replay manifest sessionId does not match its directory.");
        }

        if (requireCompleted && !manifest.completed)
        {
            return ReplayLoadResult.Failure(
                ReplayLoadStatus.Incomplete,
                "Replay recording was not completed.");
        }

        if (!string.Equals(
                manifest.payloadFile,
                Path.GetFileName(manifest.payloadFile),
                StringComparison.Ordinal) ||
            !string.Equals(
                manifest.payloadFile,
                ReplayManifest.DefaultPayloadFile,
                StringComparison.Ordinal))
        {
            return ReplayLoadResult.Failure(
                ReplayLoadStatus.Corrupt,
                "Replay payload path is unsafe or unsupported.");
        }

        string payloadPath = Path.Combine(directory, manifest.payloadFile);
        if (!File.Exists(payloadPath))
        {
            return ReplayLoadResult.Failure(
                ReplayLoadStatus.Corrupt,
                "Replay payload is missing.");
        }

        string actualSha = ReplayBinarySerializer.ComputeFileSha256(payloadPath);
        if (string.IsNullOrWhiteSpace(manifest.payloadSha256) ||
            !string.Equals(
                actualSha,
                manifest.payloadSha256,
                StringComparison.OrdinalIgnoreCase))
        {
            return ReplayLoadResult.Failure(
                ReplayLoadStatus.Corrupt,
                "Replay payload checksum validation failed.");
        }

        List<ReplayFrame> frames;
        try
        {
            frames = ReplayBinarySerializer.ReadFile(payloadPath);
        }
        catch (NotSupportedException exception)
        {
            return ReplayLoadResult.Failure(
                ReplayLoadStatus.UnsupportedVersion,
                exception.Message);
        }
        catch (Exception exception)
        {
            return ReplayLoadResult.Failure(
                ReplayLoadStatus.Corrupt,
                $"Replay payload is invalid: {exception.Message}");
        }

        if (!ReplayDataValidator.TryValidate(
                manifest,
                frames,
                requireCompleted,
                out string validationError))
        {
            ReplayLoadStatus status = manifest.completed
                ? ReplayLoadStatus.Corrupt
                : ReplayLoadStatus.Incomplete;
            return ReplayLoadResult.Failure(status, validationError);
        }

        if (requireCommitMarker)
        {
            string markerPath = Path.Combine(directory, CommitMarkerFileName);
            if (!File.Exists(markerPath) ||
                !string.Equals(
                    File.ReadAllText(markerPath, Encoding.UTF8),
                    BuildCommitMarker(manifest),
                    StringComparison.Ordinal))
            {
                return ReplayLoadResult.Failure(
                    ReplayLoadStatus.Corrupt,
                    "Replay commit marker is missing or invalid.");
            }
        }

        return ReplayLoadResult.Success(new ReplaySession(manifest, frames));
    }

    private void UpdateLatestCompleted(ReplayManifest manifest)
    {
        var index = new LatestCompletedIndex
        {
            formatVersion = ReplayManifest.CurrentFormatVersion,
            sessionId = manifest.sessionId,
            payloadSha256 = manifest.payloadSha256
        };

        string destination = Path.Combine(RootPath, LatestCompletedFileName);
        string temporary = destination + ".tmp-" + Guid.NewGuid().ToString("N");
        WriteNewTextFile(temporary, JsonUtility.ToJson(index, false));
        AtomicReplaceFile(temporary, destination);
    }

    private static ReplayManifest PrepareManifest(
        ReplayManifest source,
        IReadOnlyList<ReplayFrame> frames,
        bool completed)
    {
        if (frames == null || frames.Count == 0)
        {
            throw new InvalidDataException("A replay cannot be saved without frames.");
        }

        ReplayManifest result = source.Clone();
        result.formatVersion = ReplayManifest.CurrentFormatVersion;
        result.completed = completed;
        result.frameCount = frames.Count;
        result.duration = frames[frames.Count - 1].timestamp;
        result.payloadFile = ReplayManifest.DefaultPayloadFile;
        result.payloadSha256 = string.Empty;
        if (string.IsNullOrWhiteSpace(result.createdAtUtc))
        {
            result.createdAtUtc = DateTime.UtcNow.ToString("o");
        }

        if (!ReplayDataValidator.TryValidate(
                result,
                frames,
                requireCompleted: completed,
                out string error))
        {
            throw new InvalidDataException(error);
        }

        return result;
    }

    private static List<ReplayFrame> CloneFrames(IReadOnlyList<ReplayFrame> frames)
    {
        if (frames == null)
        {
            throw new ArgumentNullException(nameof(frames));
        }

        var copy = new List<ReplayFrame>(frames.Count);
        for (int i = 0; i < frames.Count; i++)
        {
            copy.Add(frames[i]?.Clone());
        }

        return copy;
    }

    private static void EnsurePublicationTargetsAreNew(
        string temporaryDirectory,
        string finalDirectory)
    {
        if (Directory.Exists(temporaryDirectory))
        {
            throw new IOException(
                $"Replay recording directory already exists: {temporaryDirectory}");
        }

        if (Directory.Exists(finalDirectory))
        {
            throw new IOException(
                $"Replay session already exists: {finalDirectory}");
        }
    }

    private static void WriteNewTextFile(string path, string contents)
    {
        using (var stream = new FileStream(
                   path,
                   FileMode.CreateNew,
                   FileAccess.Write,
                   FileShare.None,
                   4096,
                   FileOptions.WriteThrough))
        using (var writer = new StreamWriter(stream, new UTF8Encoding(false)))
        {
            writer.Write(contents ?? string.Empty);
            writer.Flush();
            stream.Flush(true);
        }
    }

    private static void AtomicReplaceFile(string temporary, string destination)
    {
        if (!File.Exists(destination))
        {
            File.Move(temporary, destination);
            return;
        }

        string backup = destination + ".bak-" + Guid.NewGuid().ToString("N");
        File.Replace(temporary, destination, backup, true);
        if (File.Exists(backup))
        {
            File.Delete(backup);
        }
    }

    private static string BuildCommitMarker(ReplayManifest manifest) =>
        $"{manifest.sessionId}\n{manifest.payloadSha256}\n";

    private static void ValidateSessionId(string sessionId)
    {
        if (string.IsNullOrWhiteSpace(sessionId) ||
            sessionId.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 ||
            sessionId == "." || sessionId == "..")
        {
            throw new ArgumentException("Replay sessionId is missing or unsafe.", nameof(sessionId));
        }
    }

    private static ReplayLoadStatus ClassifyException(Exception exception)
    {
        if (exception is NotSupportedException)
        {
            return ReplayLoadStatus.UnsupportedVersion;
        }

        if (exception is DirectoryNotFoundException ||
            exception is FileNotFoundException)
        {
            return ReplayLoadStatus.Unavailable;
        }

        return ReplayLoadStatus.Corrupt;
    }
}
