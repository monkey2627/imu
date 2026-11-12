using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

[DisallowMultipleComponent]
public sealed class SkinnedMeshColorAnchorInstaller : MonoBehaviour
{
    private const string TargetShaderName = "Knife/Built-in/Transparent Energy Body";
    private const int AnchorUvChannel = 2;
    private const float RescanInterval = 1f;

    private sealed class AnchoredMesh
    {
        public Mesh original;
        public Mesh runtimeCopy;
    }

    private readonly Dictionary<SkinnedMeshRenderer, AnchoredMesh> anchoredMeshes =
        new Dictionary<SkinnedMeshRenderer, AnchoredMesh>();
    private float nextScanTime;

    public static void EnsureOn(GameObject host)
    {
        if (host != null && host.GetComponent<SkinnedMeshColorAnchorInstaller>() == null)
        {
            host.AddComponent<SkinnedMeshColorAnchorInstaller>();
        }
    }

    private void OnEnable()
    {
        ApplyToScene();
    }

    private void Update()
    {
        if (Time.realtimeSinceStartup < nextScanTime)
        {
            return;
        }

        ApplyToScene();
    }

    private void OnDisable()
    {
        foreach (KeyValuePair<SkinnedMeshRenderer, AnchoredMesh> pair in anchoredMeshes)
        {
            SkinnedMeshRenderer renderer = pair.Key;
            AnchoredMesh anchored = pair.Value;
            if (renderer != null && renderer.sharedMesh == anchored.runtimeCopy)
            {
                renderer.sharedMesh = anchored.original;
                SetVertexAnchorEnabled(renderer, false);
            }

            if (anchored.runtimeCopy != null)
            {
                Destroy(anchored.runtimeCopy);
            }
        }

        anchoredMeshes.Clear();
    }

    private void ApplyToScene()
    {
        nextScanTime = Time.realtimeSinceStartup + RescanInterval;
        Scene scene = gameObject.scene;
        if (!scene.IsValid() || !scene.isLoaded)
        {
            return;
        }

        foreach (GameObject root in scene.GetRootGameObjects())
        {
            SkinnedMeshRenderer[] renderers =
                root.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            foreach (SkinnedMeshRenderer renderer in renderers)
            {
                TryAnchor(renderer);
            }
        }
    }

    private void TryAnchor(SkinnedMeshRenderer renderer)
    {
        if (renderer == null || renderer.sharedMesh == null || !UsesTargetShader(renderer))
        {
            return;
        }

        if (anchoredMeshes.TryGetValue(renderer, out AnchoredMesh existing))
        {
            if (renderer.sharedMesh == existing.runtimeCopy)
            {
                SetVertexAnchorEnabled(renderer, true);
                return;
            }

            if (existing.runtimeCopy != null)
            {
                Destroy(existing.runtimeCopy);
            }

            anchoredMeshes.Remove(renderer);
        }

        Mesh original = renderer.sharedMesh;
        Mesh runtimeCopy = Instantiate(original);
        runtimeCopy.name = original.name + " (Color Anchored)";
        runtimeCopy.hideFlags = HideFlags.DontSave;

        Vector3[] vertices = original.vertices;
        float centerX = original.bounds.center.x;
        List<Vector2> anchorCoordinates = new List<Vector2>(vertices.Length);
        foreach (Vector3 vertex in vertices)
        {
            anchorCoordinates.Add(new Vector2(vertex.x - centerX, 0f));
        }

        runtimeCopy.SetUVs(AnchorUvChannel, anchorCoordinates);
        renderer.sharedMesh = runtimeCopy;
        anchoredMeshes.Add(renderer, new AnchoredMesh
        {
            original = original,
            runtimeCopy = runtimeCopy
        });
        SetVertexAnchorEnabled(renderer, true);
    }

    private static bool UsesTargetShader(SkinnedMeshRenderer renderer)
    {
        foreach (Material material in renderer.sharedMaterials)
        {
            if (material != null && material.shader != null &&
                material.shader.name == TargetShaderName)
            {
                return true;
            }
        }

        return false;
    }

    private static void SetVertexAnchorEnabled(Renderer renderer, bool enabled)
    {
        foreach (Material material in renderer.sharedMaterials)
        {
            if (material != null && material.shader != null &&
                material.shader.name == TargetShaderName)
            {
                material.SetFloat("_VertexAnchorEnabled", enabled ? 1f : 0f);
            }
        }
    }
}
