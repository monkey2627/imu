using UnityEngine;
using TaikoSelf;

/// <summary>Presentation only: observes the existing bridge sample; never advances or judges the game.</summary>
[DefaultExecutionOrder(400)]
public sealed class RecordedDrumArtwork : MonoBehaviour
{
    public HipThrustRingBridgeView source;
    public SpriteRenderer background;
    public SpriteRenderer liftFill, liftProgress, holdFill, returnFill, comet, dot, coreGlow, constellation;
    public SpriteRenderer captionLift, captionHold, captionReturn;
    public SpriteRenderer[] stars, notes, noteRings, dust;
    public Transform dotPivot, core;
    MaterialPropertyBlock arcProperties;
    MaterialPropertyBlock backgroundProperties;
    SpriteRenderer holdProgress;

    public void LateUpdate() => Render(source.CurrentSample);

    public void Render(HipThrustPhaseTimeline.Sample sample)
    {
        // The delivered background includes the drum rim. Extend its edge pixels
        // across the viewport while keeping the baked rim at the design scale.
        if(backgroundProperties==null) backgroundProperties=new MaterialPropertyBlock();
        background.GetPropertyBlock(backgroundProperties);
        backgroundProperties.SetTexture("_MainTex",background.sprite.texture);
        backgroundProperties.SetFloat("_BackgroundScale",background.transform.localScale.x);
        background.SetPropertyBlock(backgroundProperties);
        float p = sample.progress;
        float duration = 1;
        if (sample.cycle >= 0)
        {
            var timeline = source.transport.PhaseTimeline;
            var cycle = timeline.cycles[sample.cycle];
            duration = (sample.phase == 0 ? cycle.holdStartFrame-cycle.liftStartFrame :
                sample.phase == 1 ? cycle.lowerStartFrame-cycle.holdStartFrame : cycle.lowerEndFrame-cycle.lowerStartFrame) / timeline.sourceClip.frameRate;
            if (source.transport.Hold30Timeline != null) duration = source.transport.PresentationPhaseDuration(sample);
        }
        float since = p * duration;
        bool lift = sample.phase == 0, hold = sample.phase == 1, lower = sample.phase == 2;
        // Every stage boundary remains the existing timeline's boundary. All fades are cosmetic.
        SpriteAlpha.Set(liftFill, lift ? .95f*p : 0);
        SpriteAlpha.Set(liftProgress, lift ? Mathf.Clamp01(since/.1f) : 0);
        Clip(liftProgress, SelfLayout.LiftStart, SelfLayout.LiftSweep, p, 0);
        SpriteAlpha.Set(holdFill, hold ? .95f*p : 0);
        RenderHoldProgress(hold, p);
        SpriteAlpha.Set(coreGlow, hold ? Mathf.Min(.9f,.3f+p*.84f) : 0);
        // Fit the delivered six-star reveal to the existing hold interval, without extending it.
        for (int i=0;i<stars.Length;i++)
        {
            float d = p*2.8f - (.35f+.4f*i);
            SpriteAlpha.Set(stars[i], hold ? Mathf.Clamp01(d/.12f) : 0);
            float scale = d < 0 ? .4f : d < .12f ? Mathf.Lerp(.4f,1.35f,d/.12f) : Mathf.Lerp(1.35f,1,Mathf.Clamp01((d-.12f)/.18f));
            stars[i].transform.localScale = new Vector3(scale,scale,1);
        }
        float visible = lower ? Mathf.Clamp01(since/.15f) : 0;
        SpriteAlpha.Set(returnFill, visible*(.05f+.9f*p));
        float sweep = p*SelfLayout.ReturnSweep;
        comet.transform.localRotation = Quaternion.Euler(0,0,-sweep);
        dotPivot.localRotation = Quaternion.Euler(0,0,-sweep);
        SpriteAlpha.Set(comet,visible); SpriteAlpha.Set(dot,visible);
        // The rotating comet is clipped in drum coordinates, revealing only the traversed sector.
        Clip(comet,SelfLayout.ReturnStart,SelfLayout.ReturnSweep,p,sweep);

        for(int i=0;i<notes.Length;i++)
        {
            float threshold=(SelfLayout.NoteAngles[i]-SelfLayout.ReturnStart)/SelfLayout.ReturnSweep;
            float d=lower ? (p-threshold)*duration : -1;
            float alpha=.25f, scale=1, ring=0, ringScale=.6f;
            if(d>=0)
            {
                if(d<.08f) { alpha=Mathf.Lerp(.25f,1,d/.08f); scale=Mathf.Lerp(1,1.6f,d/.08f); }
                else if(d<.5f) { alpha=Mathf.Lerp(1,.85f,(d-.08f)/.42f); scale=Mathf.Lerp(1.6f,1,(d-.08f)/.42f); }
                else alpha=.85f;
                if(d<.5f) { ring=.9f*(1-d/.5f); ringScale=Mathf.Lerp(.6f,2,Ease.OutQuad(d/.5f)); }
            }
            SpriteAlpha.Set(notes[i],alpha); notes[i].transform.localScale=new Vector3(scale,scale,1);
            SpriteAlpha.Set(noteRings[i],ring); noteRings[i].transform.localScale=new Vector3(ringScale,ringScale,1);
        }
        for(int i=0;i<dust.Length;i++)
        {
            float threshold=(SelfLayout.DustAngles[i]-SelfLayout.ReturnStart)/SelfLayout.ReturnSweep;
            float d=lower ? (p-threshold)*duration : -1;
            float alpha = d<0 || d>1.3f ? 0 : d<.05f ? d/.05f : 1-(d-.05f)/1.25f;
            SpriteAlpha.Set(dust[i],alpha);
            Vector2 position=SelfLayout.Dust[i];
            dust[i].transform.localPosition=DesignSpace.ToWorld(position.x,position.y)+Vector3.down*Mathf.Lerp(2,26,Ease.OutQuad(Mathf.Max(0,d)/1.3f))*.01f;
        }
        SpriteAlpha.Set(captionLift,lift?1:0);
        SpriteAlpha.Set(captionHold,hold?1:0);
        SpriteAlpha.Set(captionReturn,lower?1:0);
        float age=Mathf.Min(sample.completedSeconds.x,sample.completedSeconds.z);
        float bump=age<.06f ? Mathf.Lerp(1,1.1f,age/.06f) : age<.35f ? Mathf.Lerp(1.1f,1,(age-.06f)/.29f) : 1;
        core.localScale=new Vector3(bump,bump,1);
        // Decorative acknowledgement of an existing hit; no new reward condition or score writes.
        float after=sample.completedSeconds.z;
        float glow=after<.8f ? Mathf.Clamp01((after-.2f)/.25f)*Mathf.Clamp01((.8f-after)/.3f) : 0;
        SpriteAlpha.Set(constellation,source.transport.ConsecutiveHits>0 ? glow : 0);
    }

    void Clip(SpriteRenderer renderer,float start,float sweep,float progress,float angleOffset)
    {
        if(arcProperties==null) arcProperties=new MaterialPropertyBlock();
        renderer.GetPropertyBlock(arcProperties);
        arcProperties.SetTexture("_MainTex",renderer.sprite.texture);
        arcProperties.SetFloat("_UseArc",1);
        arcProperties.SetFloat("_ArcStart",start);
        arcProperties.SetFloat("_ArcSweep",sweep);
        arcProperties.SetFloat("_Progress",progress);
        arcProperties.SetFloat("_AngleOffset",angleOffset);
        renderer.SetPropertyBlock(arcProperties);
    }

    void RenderHoldProgress(bool visible, float progress)
    {
        // Shared presentation adapter: all bridge variants use their own existing
        // phase progress, including the 30-second hold. No separate visual timer.
        if (holdProgress == null)
        {
            const string name = "Hold Progress Arc";
            Transform existing = holdFill.transform.parent.Find(name);
            holdProgress = existing != null ? existing.GetComponent<SpriteRenderer>() : null;
            if (holdProgress == null)
            {
                var root = new GameObject(name);
                root.layer = holdFill.gameObject.layer;
                root.transform.SetParent(holdFill.transform.parent, false);
                root.transform.localPosition = holdFill.transform.localPosition;
                root.transform.localRotation = holdFill.transform.localRotation;
                root.transform.localScale = holdFill.transform.localScale;
                holdProgress = root.AddComponent<SpriteRenderer>();
                // The existing full-rect sprite supplies a quad around the drum.
                // The shader draws the gold arc analytically, keeping edges sharp.
                holdProgress.sprite = holdFill.sprite;
                holdProgress.sharedMaterial = holdFill.sharedMaterial;
                holdProgress.sortingLayerID = holdFill.sortingLayerID;
                holdProgress.sortingOrder = liftProgress.sortingOrder;
            }
        }
        SpriteAlpha.Set(holdProgress, visible ? 1 : 0);
        Clip(holdProgress, SelfLayout.HoldStart, SelfLayout.HoldSweep, Mathf.Clamp01(progress), 0);
        holdProgress.GetPropertyBlock(arcProperties);
        arcProperties.SetFloat("_HoldProgressArc", 1);
        holdProgress.SetPropertyBlock(arcProperties);
    }
}
