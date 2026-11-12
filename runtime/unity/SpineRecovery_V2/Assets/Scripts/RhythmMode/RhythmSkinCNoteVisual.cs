using UnityEngine;

/// <summary>Only animates sprite opacity and the outward-pointing trail. Never moves the gameplay note.</summary>
public sealed class RhythmSkinCNoteVisual : MonoBehaviour
{
    private RhythmModeController controller;
    private SpriteRenderer trail;
    private SpriteRenderer glow;
    private SpriteRenderer body;
    private double spawnTime;

    public void Initialize(RhythmModeController owner, SpriteRenderer trailRenderer,
        SpriteRenderer glowRenderer, SpriteRenderer bodyRenderer)
    {
        controller = owner;
        trail = trailRenderer;
        glow = glowRenderer;
        body = bodyRenderer;
        spawnTime = owner.SongTime;
    }

    private void LateUpdate()
    {
        if (controller == null) return;
        Vector3 p = transform.localPosition;
        trail.transform.localRotation = Quaternion.Euler(0f, 0f, Mathf.Atan2(p.y, p.x) * Mathf.Rad2Deg);
        float alpha = Mathf.Clamp01((float)(controller.SongTime - spawnTime) / 0.24f);
        RingSkinC.SpriteAlpha.Set(trail, alpha);
        RingSkinC.SpriteAlpha.Set(glow, alpha);
        RingSkinC.SpriteAlpha.Set(body, alpha);
    }
}
