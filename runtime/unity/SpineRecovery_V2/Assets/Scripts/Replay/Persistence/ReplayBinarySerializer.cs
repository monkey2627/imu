using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

public static class ReplayBinarySerializer
{
    private const int Magic = 0x33525053; // SPR3 in little-endian byte order.
    private const int MaximumFrameCount = 1_000_000;

    public static byte[] Serialize(IReadOnlyList<ReplayFrame> frames)
    {
        if (frames == null)
        {
            throw new ArgumentNullException(nameof(frames));
        }

        using (var stream = new MemoryStream())
        {
            Write(stream, frames);
            return stream.ToArray();
        }
    }

    public static List<ReplayFrame> Deserialize(byte[] bytes)
    {
        if (bytes == null)
        {
            throw new ArgumentNullException(nameof(bytes));
        }

        using (var stream = new MemoryStream(bytes, false))
        {
            return Read(stream);
        }
    }

    public static void WriteFile(string path, IReadOnlyList<ReplayFrame> frames)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Replay payload path is required.", nameof(path));
        }

        using (var stream = new FileStream(
                   path,
                   FileMode.CreateNew,
                   FileAccess.Write,
                   FileShare.None,
                   64 * 1024,
                   FileOptions.WriteThrough))
        {
            Write(stream, frames);
            stream.Flush(true);
        }
    }

    public static List<ReplayFrame> ReadFile(string path)
    {
        using (var stream = new FileStream(
                   path,
                   FileMode.Open,
                   FileAccess.Read,
                   FileShare.Read,
                   64 * 1024,
                   FileOptions.SequentialScan))
        {
            return Read(stream);
        }
    }

    public static string ComputeSha256(byte[] bytes)
    {
        using (SHA256 sha = SHA256.Create())
        {
            return ToLowerHex(sha.ComputeHash(bytes));
        }
    }

    public static string ComputeFileSha256(string path)
    {
        using (var stream = new FileStream(
                   path,
                   FileMode.Open,
                   FileAccess.Read,
                   FileShare.Read,
                   64 * 1024,
                   FileOptions.SequentialScan))
        using (SHA256 sha = SHA256.Create())
        {
            return ToLowerHex(sha.ComputeHash(stream));
        }
    }

    private static void Write(Stream stream, IReadOnlyList<ReplayFrame> frames)
    {
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
        {
            writer.Write(Magic);
            writer.Write(ReplayManifest.CurrentFormatVersion);
            writer.Write(ReplayDataValidator.HumanMuscleCount);
            writer.Write(frames.Count);

            for (int i = 0; i < frames.Count; i++)
            {
                ReplayFrame frame = frames[i] ??
                    throw new InvalidDataException($"Replay frame {i} is null.");
                writer.Write(frame.timestamp);
                WriteActor(writer, frame.user, i, "user");
                WriteActor(writer, frame.coach, i, "coach");
            }
        }
    }

    private static List<ReplayFrame> Read(Stream stream)
    {
        using (var reader = new BinaryReader(stream, Encoding.UTF8, true))
        {
            if (reader.ReadInt32() != Magic)
            {
                throw new InvalidDataException("Replay payload magic is invalid.");
            }

            int formatVersion = reader.ReadInt32();
            if (formatVersion != ReplayManifest.CurrentFormatVersion)
            {
                throw new NotSupportedException(
                    $"Replay payload format {formatVersion} is not supported.");
            }

            int muscleCount = reader.ReadInt32();
            if (muscleCount != ReplayDataValidator.HumanMuscleCount)
            {
                throw new InvalidDataException(
                    $"Replay payload muscle count {muscleCount} is incompatible with this Unity runtime.");
            }

            int frameCount = reader.ReadInt32();
            if (frameCount <= 0 || frameCount > MaximumFrameCount)
            {
                throw new InvalidDataException($"Replay payload frame count {frameCount} is invalid.");
            }

            var frames = new List<ReplayFrame>(frameCount);
            for (int i = 0; i < frameCount; i++)
            {
                frames.Add(new ReplayFrame
                {
                    timestamp = reader.ReadDouble(),
                    user = ReadActor(reader),
                    coach = ReadActor(reader)
                });
            }

            if (stream.Position != stream.Length)
            {
                throw new InvalidDataException("Replay payload has unexpected trailing bytes.");
            }

            return frames;
        }
    }

    private static void WriteActor(
        BinaryWriter writer,
        ReplayActorPose actor,
        int frameIndex,
        string actorName)
    {
        if (actor == null || actor.muscles == null ||
            actor.muscles.Length != ReplayDataValidator.HumanMuscleCount)
        {
            throw new InvalidDataException(
                $"Replay frame {frameIndex} {actorName} pose is invalid.");
        }

        Quaternion bodyRotation = ReplayDataValidator.NormalizeFinite(actor.bodyRotation);
        Quaternion rootRotation = ReplayDataValidator.NormalizeFinite(
            actor.presentationRootRotation);
        WriteVector3(writer, actor.bodyPosition);
        WriteQuaternion(writer, bodyRotation);
        WriteVector3(writer, actor.presentationRootPosition);
        WriteQuaternion(writer, rootRotation);
        writer.Write(actor.trackingValid);
        writer.Write(actor.trackingConfidence);
        for (int i = 0; i < actor.muscles.Length; i++)
        {
            writer.Write(actor.muscles[i]);
        }
    }

    private static ReplayActorPose ReadActor(BinaryReader reader)
    {
        var actor = new ReplayActorPose
        {
            bodyPosition = ReadVector3(reader),
            bodyRotation = ReplayDataValidator.NormalizeFinite(ReadQuaternion(reader)),
            presentationRootPosition = ReadVector3(reader),
            presentationRootRotation = ReplayDataValidator.NormalizeFinite(ReadQuaternion(reader)),
            trackingValid = reader.ReadBoolean(),
            trackingConfidence = reader.ReadSingle(),
            muscles = new float[ReplayDataValidator.HumanMuscleCount]
        };

        for (int i = 0; i < actor.muscles.Length; i++)
        {
            actor.muscles[i] = reader.ReadSingle();
        }

        return actor;
    }

    private static void WriteVector3(BinaryWriter writer, Vector3 value)
    {
        writer.Write(value.x);
        writer.Write(value.y);
        writer.Write(value.z);
    }

    private static Vector3 ReadVector3(BinaryReader reader) =>
        new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());

    private static void WriteQuaternion(BinaryWriter writer, Quaternion value)
    {
        writer.Write(value.x);
        writer.Write(value.y);
        writer.Write(value.z);
        writer.Write(value.w);
    }

    private static Quaternion ReadQuaternion(BinaryReader reader) =>
        new Quaternion(
            reader.ReadSingle(),
            reader.ReadSingle(),
            reader.ReadSingle(),
            reader.ReadSingle());

    private static string ToLowerHex(byte[] bytes)
    {
        var builder = new StringBuilder(bytes.Length * 2);
        for (int i = 0; i < bytes.Length; i++)
        {
            builder.Append(bytes[i].ToString("x2"));
        }

        return builder.ToString();
    }
}
