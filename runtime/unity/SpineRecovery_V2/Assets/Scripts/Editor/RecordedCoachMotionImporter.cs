using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

// The FBX export loses rotations on intermediate OpenSim wrist/ankle bones.
// Import the accompanying original HumanPose recording directly instead.
[ScriptedImporter(2, "coachrecording")]
public sealed class RecordedCoachMotionImporter : ScriptedImporter
{
    public override void OnImportAsset(AssetImportContext context)
    {
        string phasePath = Path.ChangeExtension(context.assetPath, ".phases.json");
        context.DependsOnSourceAsset(phasePath);
        var map = JsonUtility.FromJson<RecordedCoachSceneTools.PhaseMap>(File.ReadAllText(phasePath));
        var package = JsonUtility.FromJson<CoachMotionPackage>(File.ReadAllText(context.assetPath));
        if (!package.IsValid(out string error)) throw new InvalidDataException(error);
        var frames = package.segments.Single().frames;
        int cycleCount = map.peaks.Length;
        if (cycleCount < 1 || map.boundaries.Length != cycleCount + 1)
            throw new InvalidDataException("Phase boundaries must bracket every recorded cycle.");
        string prefix = map.action.ToLowerInvariant();
        var info = new TextAsset($"{map.action}: original HumanPose recording, {map.fps} fps, {3 + cycleCount * 2} phase clips.") { name = map.action };
        context.AddObjectToAsset("source", info);
        context.SetMainObject(info);
        Add(prefix + "_routine", map.routineStart, map.routineEnd);
        Add(prefix + "_start", map.routineStart, map.boundaries[0]);
        Add(prefix + "_end", map.boundaries[cycleCount], map.routineEnd);
        for (int i = 0; i < cycleCount; i++)
        {
            Add(prefix + "_" + (i + 1) + "_up", map.boundaries[i], map.peaks[i]);
            Add(prefix + "_" + (i + 1) + "_down", map.peaks[i], map.boundaries[i + 1]);
        }

        void Add(string name, int first, int last)
        {
            if (first < 0 || last <= first || last >= frames.Length) throw new InvalidDataException("Invalid phase " + name);
            var clip = Bake(name, frames, first, last, map.fps, map.bodyYawDegrees, map.mirrorArmChannels);
            context.AddObjectToAsset(name, clip);
        }
    }

    // The bird-dog source raises the same-side arm and leg. Mirror only the anatomical
    // arm muscle channels (shoulder through fingers), retaining leg motion and timing.
    // Humanoid left/right muscle channels express the same anatomical range on each side.
    public static int SourceMuscleIndex(int index, bool mirrorArms)
    {
        if (!mirrorArms) return index;
        string name = HumanTrait.MuscleName[index];
        bool left = name.StartsWith("Left");
        bool right = name.StartsWith("Right");
        if (!left && !right) return index;
        string suffix = name.Substring(left ? 4 : 5);
        string part = suffix.TrimStart();
        if (!(part.StartsWith("Shoulder") || part.StartsWith("Arm") || part.StartsWith("Forearm") ||
            part.StartsWith("Hand") || part.StartsWith("Thumb") || part.StartsWith("Index") ||
            part.StartsWith("Middle") || part.StartsWith("Ring") || part.StartsWith("Little"))) return index;
        int other = Array.IndexOf(HumanTrait.MuscleName, (left ? "Right" : "Left") + suffix);
        if (other < 0) throw new InvalidDataException("Missing mirrored muscle: " + name);
        return other;
    }

    public static AnimationClip Bake(string name, CoachMotionFrame[] frames, int first, int last, float fps, float yaw, bool mirrorArms)
    {
        var clip = new AnimationClip { name = name, frameRate = fps };
        string[] properties = HumanTrait.MuscleName.Concat(new[] { "RootT.x", "RootT.y", "RootT.z", "RootQ.x", "RootQ.y", "RootQ.z", "RootQ.w" }).ToArray();
        var values = properties.Select(_ => new float[last - first + 1]).ToArray();
        var sourceMuscles = Enumerable.Range(0, HumanTrait.MuscleCount).Select(m => SourceMuscleIndex(m,mirrorArms)).ToArray();
        Quaternion heading = Quaternion.Euler(0, yaw, 0), previous = Quaternion.identity;
        for (int f = first; f <= last; f++)
        {
            int index = f - first;
            for (int m = 0; m < HumanTrait.MuscleCount; m++) values[m][index] = frames[f].muscles[sourceMuscles[m]];
            Vector3 position = heading * frames[f].bodyPosition;
            Quaternion rotation = heading * frames[f].bodyRotation;
            if (index > 0 && Quaternion.Dot(previous, rotation) < 0) rotation = new Quaternion(-rotation.x, -rotation.y, -rotation.z, -rotation.w);
            previous = rotation;
            float[] root = { position.x, position.y, position.z, rotation.x, rotation.y, rotation.z, rotation.w };
            for (int m = 0; m < root.Length; m++) values[HumanTrait.MuscleCount + m][index] = root[m];
        }
        for (int m = 0; m < properties.Length; m++)
        {
            // Keep every recorded sample; collapse only exactly constant curves.
            bool constant = values[m].All(v => v == values[m][0]);
            var keys = new List<Keyframe>();
            for (int f = 0; f < values[m].Length; f++)
                if (!constant || f == 0 || f == values[m].Length - 1) keys.Add(new Keyframe(f / fps, values[m][f]));
            var curve = new AnimationCurve(keys.ToArray());
            for (int k = 0; k < curve.length; k++)
            {
                AnimationUtility.SetKeyLeftTangentMode(curve, k, AnimationUtility.TangentMode.Linear);
                AnimationUtility.SetKeyRightTangentMode(curve, k, AnimationUtility.TangentMode.Linear);
            }
            AnimationUtility.SetEditorCurve(clip, EditorCurveBinding.FloatCurve("", typeof(Animator), properties[m]), curve);
        }
        var settings = AnimationUtility.GetAnimationClipSettings(clip);
        settings.loopTime = false;
        settings.keepOriginalOrientation = settings.keepOriginalPositionY = settings.keepOriginalPositionXZ = true;
        settings.loopBlendOrientation = settings.loopBlendPositionY = settings.loopBlendPositionXZ = true;
        AnimationUtility.SetAnimationClipSettings(clip, settings);
        return clip;
    }
}
