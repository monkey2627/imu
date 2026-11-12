using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

/// <summary>Registers the original recorded-coach games and their new bridge variants.</summary>
public static class GameSceneOrganizer
{
    [MenuItem("Tools/Ring Skin C/Organize Game Scenes")]
    public static void Run()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode)
            throw new InvalidOperationException("Exit Play mode before organizing scenes.");

        var paths = new[] { RecordedCoachSceneTools.Recorded("BirdDog"),
            RecordedCoachSceneTools.Recorded("DeadBug"), RingBridgeGameBuilder.ScenePath,
            HipThrustNewRulesBuilder.NewScore, HipThrustNewRulesBuilder.Hold30,
            SideBridgeGameBuilder.Dynamic, SideBridgeGameBuilder.Hold30 };
        // Do not resurrect scenes deliberately removed from the project.
        EditorBuildSettings.scenes = paths.Where(File.Exists).Select(p => new EditorBuildSettingsScene(p, true)).ToArray();
        Debug.Log("GAME_SCENES_ORGANIZED: " + EditorBuildSettings.scenes.Length + " games; existing scene files preserved.");
    }
}
