using System;
using UnityEngine;

/// <summary>Samples a recording on the patient's avatar proportions during synchronous
/// reference capture. Every original transform (including wrists/ankles) is restored
/// before control returns; no patient pose is changed during gameplay.</summary>
public sealed class PatientPoseReference : IDisposable
{
    public Animator Animator { get; }
    readonly HumanPoseHandler source,destination;
    readonly Transform[] transforms;
    readonly Vector3[] positions;
    readonly Quaternion[] rotations;
    public PatientPoseReference(Animator coach,Animator patient)
    {
        Animator=patient!=null&&patient.isHuman?patient:coach;
        if(Animator==coach)return;
        transforms=Animator.GetComponentsInChildren<Transform>(true);positions=new Vector3[transforms.Length];rotations=new Quaternion[transforms.Length];
        for(int i=0;i<transforms.Length;i++){positions[i]=transforms[i].localPosition;rotations[i]=transforms[i].localRotation;}
        source=new HumanPoseHandler(coach.avatar,coach.transform);destination=new HumanPoseHandler(Animator.avatar,Animator.transform);
    }
    public void Apply(){if(source==null)return;var pose=new HumanPose();source.GetHumanPose(ref pose);destination.SetHumanPose(ref pose);}
    public void Dispose()
    {
        if(transforms!=null)for(int i=0;i<transforms.Length;i++)if(transforms[i]!=null)
        {transforms[i].localPosition=positions[i];transforms[i].localRotation=rotations[i];}
        source?.Dispose();destination?.Dispose();
    }
}
