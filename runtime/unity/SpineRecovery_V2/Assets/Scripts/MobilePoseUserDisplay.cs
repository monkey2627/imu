using UnityEngine;

/// <summary>Manages the scene-authored live user preview.</summary>
[DefaultExecutionOrder(160)]
[DisallowMultipleComponent]
public sealed class MobilePoseUserDisplay : MonoBehaviour
{
    [SerializeField] private Animator userAnimator;
    [SerializeField] private Camera userCamera;
    [SerializeField] private Canvas previewCanvas;
    [SerializeField] private int displayLayer = 31;

    private GameObject userRoot;
    private bool initialized;
    private bool missingReferencesReported;
    private readonly System.Collections.Generic.Dictionary<Transform, int> originalLayers =
        new System.Collections.Generic.Dictionary<Transform, int>();

    private void Awake()
    {
        FindNamedReferences();
        TryInitialize();
    }

    public void Configure(Animator animator)
    {
        if (animator != null) userAnimator = animator;
        FindNamedReferences();
        TryInitialize();
    }

    private void LateUpdate()
    {
        if (!initialized)
        {
            if (userAnimator == null)
            {
                DeadBugGamingPoseScorer scorer = FindObjectOfType<DeadBugGamingPoseScorer>(true);
                if (scorer != null) userAnimator = scorer.UserAnimator;
            }
            FindNamedReferences();
            TryInitialize();
            return;
        }

        if (!previewCanvas.gameObject.activeSelf)
            previewCanvas.gameObject.SetActive(true);
        if (!previewCanvas.enabled) previewCanvas.enabled = true;
        if (!userCamera.enabled) userCamera.enabled = true;
    }

    private void FindNamedReferences()
    {
        if (userCamera == null)
        {
            foreach (Camera candidate in FindObjectsOfType<Camera>(true))
            {
                if (candidate.gameObject.name == "MobilePose User Preview Camera")
                {
                    userCamera = candidate;
                    break;
                }
            }
        }

        if (previewCanvas == null)
        {
            foreach (Canvas candidate in FindObjectsOfType<Canvas>(true))
            {
                if (candidate.gameObject.name == "MobilePose User Preview")
                {
                    previewCanvas = candidate;
                    break;
                }
            }
        }
    }

    private void TryInitialize()
    {
        if (initialized) return;
        if (userAnimator == null || !userAnimator.isHuman || userCamera == null || previewCanvas == null)
        {
            if (!missingReferencesReported)
            {
                Debug.LogWarning("MobilePose preview is waiting for the user Animator, preview camera, and preview canvas.", this);
                missingReferencesReported = true;
            }
            return;
        }

        userRoot = userAnimator.transform.root.gameObject;
        SetLayerRecursively(userRoot, displayLayer);
        userCamera.cullingMask = 1 << displayLayer;
        userCamera.enabled = true;
        previewCanvas.enabled = true;
        previewCanvas.gameObject.SetActive(true);
        initialized = true;
    }

    private void OnDestroy()
    {
        foreach (var entry in originalLayers)
        {
            if (entry.Key != null) entry.Key.gameObject.layer = entry.Value;
        }
    }

    private void SetLayerRecursively(GameObject root, int layer)
    {
        if (!originalLayers.ContainsKey(root.transform))
        {
            originalLayers.Add(root.transform, root.layer);
        }
        root.layer = layer;
        foreach (Transform child in root.transform)
        {
            SetLayerRecursively(child.gameObject, layer);
        }
    }
}
