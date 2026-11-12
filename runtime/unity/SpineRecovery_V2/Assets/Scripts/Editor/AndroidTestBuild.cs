using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

public static class AndroidTestBuild
{
    public static void Build()
    {
        Directory.CreateDirectory("output/android-test");
        UserRigVerification.ValidateBuildRigs();
        var scenes = GameTestMenu.Scenes.Select(s => "Assets/Scenes/" + s + ".unity").ToArray();
        foreach (var scene in scenes) if (!File.Exists(scene)) throw new Exception("Missing test scene: " + scene);
        foreach (var path in AssetDatabase.GetDependencies(scenes, true)) {
            var material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material != null && material.shader != null && material.shader.name.StartsWith("VRM10/Universal Render Pipeline/"))
                throw new Exception("Scene requires URP, which is not installed: " + path);
        }
        EditorBuildSettings.scenes = scenes.Select(s => new EditorBuildSettingsScene(s, true)).ToArray();
        PlayerSettings.companyName = "MetaSpine";
        PlayerSettings.productName = "脊柱康复·六项测试";
        PlayerSettings.SetApplicationIdentifier(BuildTargetGroup.Android, "com.metaspine.recovery.gamingtest");
        PlayerSettings.bundleVersion = "0.2.1-test.20261030";
        PlayerSettings.Android.bundleVersionCode = 2026103002;
        PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel26;
        PlayerSettings.Android.targetSdkVersion = (AndroidSdkVersions)35;
        PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64 | AndroidArchitecture.ARMv7;
        PlayerSettings.SetScriptingBackend(BuildTargetGroup.Android, ScriptingImplementation.IL2CPP);
        PlayerSettings.SetManagedStrippingLevel(BuildTargetGroup.Android, ManagedStrippingLevel.Minimal);
        PlayerSettings.defaultInterfaceOrientation = UIOrientation.LandscapeLeft;
        PlayerSettings.runInBackground = true;
        PlayerSettings.Android.forceInternetPermission = true;
        PlayerSettings.insecureHttpOption = InsecureHttpOption.AlwaysAllowed;
        PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
        PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { UnityEngine.Rendering.GraphicsDeviceType.OpenGLES3 });
        EditorUserBuildSettings.buildAppBundle = false;
        PlayerSettings.Android.useCustomKeystore = false;
        var settings = Resources.FindObjectsOfTypeAll<PlayerSettings>().FirstOrDefault();
        if (settings == null) throw new Exception("PlayerSettings unavailable");
        var serialized = new SerializedObject(settings);
        var input = serialized.FindProperty("activeInputHandler");
        if (input == null) throw new Exception("Active input setting unavailable");
        input.intValue = 0; // Existing touch UI uses StandaloneInputModule; Both is unsupported on Android.
        serialized.ApplyModifiedPropertiesWithoutUndo();
        string apk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "SpineRecovery_六项游戏测试_评分修正版_20261030.apk");
        var result = BuildPipeline.BuildPlayer(new BuildPlayerOptions {
            scenes = scenes, target = BuildTarget.Android, locationPathName = apk,
            extraScriptingDefines = new[] { "SPINE_TEST_BUILD" },
            options = BuildOptions.Development | BuildOptions.CompressWithLz4
        });
        File.WriteAllText("output/android-test/build.txt", $"Result: {result.summary.result}\nAPK: {apk}\nSize: {result.summary.totalSize}\nErrors: {result.summary.totalErrors}\nDuration: {result.summary.totalTime}\n");
        if (result.summary.result != BuildResult.Succeeded) throw new Exception("Android build failed");
    }
}

// The vendored VRM package ships optional URP Resources shaders. This project uses
// the built-in pipeline; skip these unused variants instead of compiling missing URP includes.
public sealed class AndroidBuiltInShaderFilter : UnityEditor.Build.IPreprocessShaders
{
    public int callbackOrder => 0;
    public void OnProcessShader(Shader shader, UnityEditor.Rendering.ShaderSnippetData snippet,
        System.Collections.Generic.IList<UnityEditor.Rendering.ShaderCompilerData> variants)
    {
        if (EditorUserBuildSettings.activeBuildTarget == BuildTarget.Android &&
            UnityEngine.Rendering.GraphicsSettings.currentRenderPipeline == null &&
            shader.name.StartsWith("VRM10/Universal Render Pipeline/")) variants.Clear();
    }
}
