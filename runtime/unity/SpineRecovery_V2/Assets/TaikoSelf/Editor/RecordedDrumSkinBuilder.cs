using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using TMPro;
using TaikoSelf;

/// <summary>Replaces artwork inside the existing scene. No game/controller/timeline changes.</summary>
public static class RecordedDrumSkinBuilder
{
    public const string ScenePath="Assets/Scenes/HipThrust_RingBridge_Recorded.unity";
    public const string Output="output/taikoself/";
    const string Root="Assets/TaikoSelf/";
    static Material normal, white;

    [MenuItem("Tools/Taiko Self/Apply Artwork to Recorded Bridge")]
    public static void Apply()
    {
        if(EditorApplication.isPlayingOrWillChangePlaymode) throw new InvalidOperationException("Exit Play first.");
        if(UnityEngine.SceneManagement.SceneManager.GetActiveScene().isDirty) throw new InvalidOperationException("Save the current scene first.");
        ConfigureSprites();
        Directory.CreateDirectory(Root+"RecordedVisuals/Materials"); AssetDatabase.Refresh();
        normal=MaterialFor("Artwork",0); white=MaterialFor("WhiteSparkles",1);
        var scene=EditorSceneManager.OpenScene(ScenePath);
        var view=scene.GetRootGameObjects().SelectMany(r=>r.GetComponentsInChildren<HipThrustRingBridgeView>(true)).Single();
        foreach(var text in scene.GetRootGameObjects().SelectMany(r=>r.GetComponentsInChildren<TMP_Text>(true)))
            if(text.name=="Description") text.text=text.text.Replace("蓝色放下","紫色放下");
        Transform existing=view.artwork.transform;
        var previous=existing.Find("Recorded Drum Artwork");
        if(previous!=null) UnityEngine.Object.DestroyImmediate(previous.gameObject);
        // Keep the existing view references and hidden chart intact. Only its visible artwork is replaced.
        existing.Find("Ring").gameObject.SetActive(false);
        existing.Find("HUD_Sprites").gameObject.SetActive(false);
        var background=existing.Find("Background");
        foreach(Transform child in background) child.gameObject.SetActive(child.name=="rb_bg");
        var bg=background.Find("rb_bg").GetComponent<SpriteRenderer>(); bg.sprite=Sprite("ts_bg"); bg.sharedMaterial=normal;
        var root=Group("Recorded Drum Artwork",existing);
        var visual=root.gameObject.AddComponent<RecordedDrumArtwork>(); visual.source=view; visual.background=bg;
        BuildBackground(root);
        Transform head=Group("DrumHead",root);
        Place(head,"ts_bases");
        visual.liftFill=Place(head,"ts_lift_fill"); visual.liftProgress=Place(head,"ts_lift_prog");
        visual.holdFill=Place(head,"ts_hold_fill"); visual.returnFill=Place(head,"ts_ret_fill");
        visual.comet=Place(head,"ts_ret_comet");
        visual.stars=new SpriteRenderer[SelfLayout.Stars.Length];
        for(int i=0;i<visual.stars.Length;i++)
        {
            PlaceAt(head,"Star Slot "+i,"ts_star_slot",SelfLayout.Stars[i],13);
            visual.stars[i]=PlaceAt(head,"Star "+i,"ts_star_lit",SelfLayout.Stars[i],14);
        }
        visual.notes=new SpriteRenderer[SelfLayout.Notes.Length]; visual.noteRings=new SpriteRenderer[visual.notes.Length];
        for(int i=0;i<visual.notes.Length;i++)
        {
            visual.notes[i]=PlaceAt(head,"Note "+i,"ts_note",SelfLayout.Notes[i],16);
            visual.noteRings[i]=PlaceAt(head,"Note Ring "+i,"ts_note_ring",SelfLayout.Notes[i],17);
        }
        visual.dotPivot=Group("Dot Pivot",head); visual.dotPivot.localPosition=DesignSpace.RingCenter;
        var dot=SelfLayout.Layers["ts_ret_dot"];
        visual.dot=PlaceAt(visual.dotPivot,"Dot","ts_ret_dot",new Vector2(dot.x,dot.y),dot.order);
        visual.dot.transform.localPosition-=DesignSpace.RingCenter;
        visual.dust=new SpriteRenderer[SelfLayout.Dust.Length];
        for(int i=0;i<visual.dust.Length;i++) visual.dust[i]=PlaceAt(head,"Dust "+i,"ts_dust_"+SelfLayout.DustVariant[i],SelfLayout.Dust[i],19);
        visual.coreGlow=Place(head,"ts_core_glow"); visual.core=Place(head,"ts_core").transform;
        Place(head,"ts_sheen"); visual.constellation=Place(head,"ts_const");
        var hud=Group("HUD Sprites",root); Place(hud,"ts_hud_pause");
        visual.captionLift=Place(hud,"ts_cap_lift"); visual.captionHold=Place(hud,"ts_cap_hold"); visual.captionReturn=Place(hud,"ts_cap_ret");
        visual.Render(view.transport.PhaseTimeline.Evaluate(0));
        foreach(Transform child in root.GetComponentsInChildren<Transform>(true)) child.gameObject.layer=29;
        EditorSceneManager.MarkSceneDirty(scene); EditorSceneManager.SaveScene(scene); AssetDatabase.SaveAssets();
        Directory.CreateDirectory(Output);
        File.WriteAllText(Output+"build-result.txt","Updated artwork in "+ScenePath+". Existing controller, timeline, animation, menu, score and build entries preserved.\n");
    }

    static void BuildBackground(Transform parent)
    {
        Transform bg=Group("Background Effects",parent);
        for(int i=0;i<SelfLayout.Sparkles.Length;i++)
        {
            var sprite=Place(bg,"ts_sparkle_"+i); sprite.sharedMaterial=white;
            var tw=sprite.gameObject.AddComponent<TaikoSelf.Twinkle>(); tw.period=SelfLayout.Sparkles[i].period; tw.phase=SelfLayout.Sparkles[i].phase;
        }
        Place(bg,"ts_star_glow").gameObject.AddComponent<TaikoSelf.Twinkle>().period=3;
        var meteor=Place(bg,"ts_meteor").gameObject.AddComponent<TaikoSelf.MeteorLoop>();
        meteor.headToSpriteCenterPx=new Vector2(SelfLayout.MeteorCenterOffsetX,SelfLayout.MeteorCenterOffsetY);
        SpriteAlpha.Set(meteor.GetComponent<SpriteRenderer>(),0);
        Place(bg,"ts_lanterns").gameObject.AddComponent<TaikoSelf.FloatBob>();
        Place(bg,"ts_spin_ring").gameObject.AddComponent<TaikoSelf.SpinForever>().secondsPerTurn=40;
    }
    static Transform Group(string name,Transform parent) { var go=new GameObject(name); go.transform.SetParent(parent,false); return go.transform; }
    static Sprite Sprite(string name) => AssetDatabase.LoadAssetAtPath<Sprite>(Root+"Sprites/"+name+".png");
    static SpriteRenderer Place(Transform parent,string name)
    {
        var layer=SelfLayout.Layers[name]; return PlaceAt(parent,name,name,new Vector2(layer.x,layer.y),layer.order);
    }
    static SpriteRenderer PlaceAt(Transform parent,string name,string image,Vector2 point,int order)
    {
        var t=Group(name,parent); t.localPosition=DesignSpace.ToWorld(point.x,point.y);
        var r=t.gameObject.AddComponent<SpriteRenderer>(); r.sprite=Sprite(image); r.sortingOrder=order; r.sharedMaterial=normal; return r;
    }
    static Material MaterialFor(string name,float mask)
    {
        string path=Root+"RecordedVisuals/Materials/"+name+".mat";
        var material=AssetDatabase.LoadAssetAtPath<Material>(path);
        if(material==null) { material=new Material(Shader.Find("TaikoSelf/RecordedSprite")); AssetDatabase.CreateAsset(material,path); }
        material.SetFloat("_WhiteMask",mask); EditorUtility.SetDirty(material); return material;
    }
    static void ConfigureSprites()
    {
        foreach(string guid in AssetDatabase.FindAssets("t:Texture2D",new[]{Root+"Sprites"}))
        {
            var ti=(TextureImporter)AssetImporter.GetAtPath(AssetDatabase.GUIDToAssetPath(guid));
            ti.textureType=TextureImporterType.Sprite; ti.textureShape=TextureImporterShape.Texture2D;
            ti.spriteImportMode=SpriteImportMode.Single; ti.spritePixelsPerUnit=200; ti.mipmapEnabled=false;
            ti.alphaIsTransparency=true; ti.wrapMode=TextureWrapMode.Clamp; ti.filterMode=FilterMode.Bilinear;
            ti.npotScale=TextureImporterNPOTScale.None; ti.maxTextureSize=4096; ti.textureCompression=TextureImporterCompression.Uncompressed;
            var settings=new TextureImporterSettings(); ti.ReadTextureSettings(settings);
            settings.spriteMeshType=SpriteMeshType.FullRect; settings.spriteAlignment=(int)SpriteAlignment.Center;
            ti.SetTextureSettings(settings); ti.SaveAndReimport();
        }
    }
}
