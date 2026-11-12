using System;
using UnityEngine;

/// <summary>D01 V2 prescription -> exact bridge dose. No difficulty-to-count conversion.</summary>
public sealed class HipThrustPrescriptionDose : MonoBehaviour
{
    [Tooltip("Explicit editor/demo prescription, replaced by SetPrescriptionJson when a real prescription arrives.")]
    public TextAsset previewPrescription;
    public HipThrustTaikoController transport;
    public DeadBugStartMenuController menu;
    public TMPro.TMP_Text menuDescription;
    public string prescriptionActionId = "5";
    public string actionDisplayName = "臀桥";
    public bool preserveRecordedTiming;
    public bool skipRecordedPlateauAfterHold;
    public bool preserveSourceGaps;
    public int Sets { get; private set; }
    public int Reps { get; private set; }
    public int Total => Sets * Reps;
    public bool IsPreview { get; private set; }
    public string Label => Total>0 ? $"{(IsPreview ? "演示处方" : "处方")} {Sets} 组 × {Reps} 次 · {(preserveRecordedTiming ? "录制节奏" : "保持 30 秒")}" : $"等待{actionDisplayName}处方";
    public string LastError { get; private set; }
    [Serializable] sealed class Prescription { public int prescriptionContractVersion; public Item[] items; }
    [Serializable] sealed class Item { public long itemId; public string actionId; public int sets; public int reps; }

    void Awake()
    {
        if(previewPrescription!=null && SetPrescriptionJson(previewPrescription.text,0,out _)) IsPreview=true;
    }

    void LateUpdate()
    { RefreshDisplay(); }

    void RefreshDisplay()
    {
        if(menu==null || !menu.IsTrainingMenuVisible) return;
        if(menu.TimesText!=null) menu.TimesText.text=Total.ToString();
        if(menuDescription!=null) menuDescription.text=Label;
    }

    // Main thread. itemId is required when a prescription contains repeated bridge items.
    public bool SetPrescriptionJson(string json,long itemId,out string error)
    {
        try
        {
            var prescription=JsonUtility.FromJson<Prescription>(json);
            if(prescription==null || prescription.prescriptionContractVersion!=2 || prescription.items==null)
                throw new ArgumentException("需要 D01 V2 处方及 items。");
            Item selected=null;
            foreach(var item in prescription.items)
            {
                if(item==null || item.actionId!=prescriptionActionId || (itemId!=0 && item.itemId!=itemId)) continue;
                if(selected!=null) throw new ArgumentException($"处方包含多个{actionDisplayName}项目，请指定 itemId。");
                selected=item;
            }
            if(selected==null) throw new ArgumentException($"处方中没有指定的{actionDisplayName}动作（actionId={prescriptionActionId}）。");
            return SetDose(selected.sets,selected.reps,out error);
        }
        catch(Exception ex) { error=LastError=ex.Message; return false; }
    }

    public bool SetDose(int sets,int reps,out string error)
    {
        error=null;
        if(transport!=null && (transport.IsRunning || (menu!=null && !menu.IsTrainingMenuVisible)))
            error="请先返回菜单，再更换处方。";
        else if(sets<1 || sets>20 || reps<1 || reps>100) error="处方组数须为 1–20，次数须为 1–100。";
        if(error!=null) { LastError=error; return false; }
        Sets=sets; Reps=reps; IsPreview=false; LastError=null; RefreshDisplay(); return true;
    }
}
