using TMPro;
using UnityEngine;
using UnityEngine.UI;

/// <summary>Shared percentage settlement for the six game scenes, replacing the
/// authored placeholder score. Visibility still follows the existing end menu.</summary>
public sealed class TrainingResultView : MonoBehaviour
{
    GameTrainingAssessment assessment;
    GameObject oldEnd,root;
    CanvasScaler scaler;
    TMP_Text title,score,quality,holdTime,followingTime,repetitions;
    bool wasVisible;
    public GameObject Root => root;
    public TMP_Text ScoreLabel => score;
    public void Initialize(GameTrainingAssessment source){assessment=source;}
    void Start()
    {
        foreach(var canvas in FindObjectsOfType<Canvas>(true))
            if(canvas.gameObject.scene==gameObject.scene&&canvas.name=="EndMenuCanvas")oldEnd=canvas.gameObject;
        if(oldEnd==null)return;
        // Hide authored placeholder graphics; keep their controls and the original
        // end-canvas lifecycle intact for existing menu and prescription listeners.
        foreach(var graphic in oldEnd.GetComponentsInChildren<Graphic>(true))graphic.enabled=false;
        root=new GameObject("Training Quality Result",typeof(RectTransform),typeof(Canvas),typeof(CanvasScaler),typeof(GraphicRaycaster));
        var canvasRoot=root.GetComponent<Canvas>();canvasRoot.renderMode=RenderMode.ScreenSpaceOverlay;canvasRoot.sortingOrder=1200;
        scaler=root.GetComponent<CanvasScaler>();scaler.uiScaleMode=CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution=new Vector2(1920,1080);
        var bg=Box("Background",root.transform,Vector2.zero,new Vector2(1920,1080),new Color(.045f,.04f,.14f));
        bg.anchorMin=Vector2.zero;bg.anchorMax=Vector2.one;bg.sizeDelta=Vector2.zero;
        Box("Card",root.transform,new Vector2(0,0),new Vector2(1660,920),new Color(.12f,.10f,.25f));
        title=Label("本组训练报告",new Vector2(0,375),new Vector2(1500,70),48);
        title.color=new Color(.94f,.89f,1);
        score=Label("—",new Vector2(-540,65),new Vector2(480,200),150);
        score.color=new Color(1,.82f,.39f);
        Label("总成绩 / 100",new Vector2(-540,-70),new Vector2(480,65),32);
        quality=Metric("动作质量",new Vector2(0,230));
        holdTime=Metric("合格保持时长",new Vector2(0,80));
        followingTime=Metric("合格跟随时长",new Vector2(0,-70));
        repetitions=Metric("合格次数",new Vector2(0,-220));
        var buttonRect=Box("Return",root.transform,new Vector2(0,-390),new Vector2(360,68),new Color(.40f,.31f,.67f));
        var button=buttonRect.gameObject.AddComponent<Button>();button.targetGraphic=buttonRect.GetComponent<Image>();
        button.onClick.AddListener(()=>{oldEnd.SetActive(false);assessment.Entry.StopTraining();});
        var caption=Label("返回选择",new Vector2(0,-390),new Vector2(350,65),30);caption.raycastTarget=false;
        root.SetActive(false);
    }
    TMP_Text Metric(string label,Vector2 position)
    {
        Box(label,root.transform,new Vector2(250,position.y),new Vector2(970,125),new Color(.19f,.16f,.34f));
        var text=Label(label,new Vector2(250,position.y),new Vector2(910,110),34);
        text.alignment=TextAlignmentOptions.MidlineLeft;return text;
    }
    void LateUpdate()
    {
        if(root==null)return;
        bool visible=oldEnd.activeInHierarchy&&!assessment.Entry.menu.IsTrainingMenuVisible;
        root.SetActive(visible);scaler.matchWidthOrHeight=Screen.width/(float)Mathf.Max(1,Screen.height)<1920f/1080?0:1;
        if(visible&&!wasVisible)Refresh();wasVisible=visible;
    }
    public void Refresh()
    {
        if(score==null)return;
        var s=assessment.Summary;bool available=assessment.IsFinished&&s.HasAssessment;
        title.text=GameWorkoutPlan.ActionName(assessment.Entry.actionId)+" · 本组训练报告";
        score.text=available?s.Score.ToString():"—";
        quality.text=$"动作质量   {(available?s.Quality.ToString("0.0"):"—")} / 100";
        holdTime.text=s.IsHold?$"合格保持时长   {s.CreditedSeconds:0.0} / {s.TargetSeconds:0.0} 秒":"合格保持时长   —";
        followingTime.text=$"合格跟随时长   {s.QualifiedFollowingSeconds:0.0} / {s.ExpectedFollowingSeconds:0.0} 秒";
        repetitions.text=$"合格次数   {s.Qualified} / {s.Planned} 次";
    }
    RectTransform Box(string name,Transform parent,Vector2 position,Vector2 size,Color color)
    {
        var rect=new GameObject(name,typeof(RectTransform),typeof(Image)).GetComponent<RectTransform>();rect.SetParent(parent,false);
        rect.anchorMin=rect.anchorMax=new Vector2(.5f,.5f);rect.sizeDelta=size;rect.anchoredPosition=position;
        rect.GetComponent<Image>().color=color;return rect;
    }
    TMP_Text Label(string text,Vector2 position,Vector2 size,float fontSize)
    {
        var rect=new GameObject("Result Label",typeof(RectTransform)).GetComponent<RectTransform>();rect.SetParent(root.transform,false);
        rect.anchorMin=rect.anchorMax=new Vector2(.5f,.5f);rect.sizeDelta=size;rect.anchoredPosition=position;
        var label=rect.gameObject.AddComponent<TextMeshProUGUI>();label.font=assessment.Entry.font;label.text=text;
        label.fontSize=fontSize;label.color=new Color(.90f,.86f,.97f);label.alignment=TextAlignmentOptions.Center;
        label.enableWordWrapping=true;label.raycastTarget=false;return label;
    }
    void OnDestroy(){if(root!=null)Destroy(root);}
}
