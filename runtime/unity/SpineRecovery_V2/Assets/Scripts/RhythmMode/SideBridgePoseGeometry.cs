using UnityEngine;

/// <summary>Left-side support geometry. Translation, body scale and horizontal heading invariant.</summary>
public sealed class SideBridgePoseGeometry
{
    public sealed class Pose
    {
        public float lift, trunkAngle, kneeAngle, elbowAngle, shoulderStack, pelvisStack, torsoVertical;
        public float shoulderOverElbow;
        public Vector3 thigh, shin, shoulderAxis, pelvisAxis;
        public bool IsLeftSupport => shoulderStack > .45f && pelvisStack > .5f && torsoVertical < .6f;
    }
    public Pose Rest { get; private set; }
    public Pose Top { get; private set; }
    public float RestLift { get; private set; }
    public float TopLift { get; private set; }

    public bool SetTargets(Pose rest, Pose top, float restLift, float topLift)
    {
        Rest=rest; Top=top; RestLift=restLift; TopLift=topLift;
        return rest!=null && top!=null && rest.IsLeftSupport && top.IsLeftSupport &&
            Finite(restLift) && Finite(topLift) && topLift-restLift>.015f;
    }

    public float RawProgress(Pose pose) => (pose.lift-RestLift)/(TopLift-RestLift);
    public float Progress(Pose pose) => Mathf.Clamp01(RawProgress(pose));

    public static bool HasStableSupport(Pose pose,Pose target) => pose!=null&&target!=null&&pose.IsLeftSupport&&
        pose.shoulderAxis.sqrMagnitude>.5f&&pose.pelvisAxis.sqrMagnitude>.5f&&
        Mathf.Abs(pose.elbowAngle-target.elbowAngle)<=25 &&
        Mathf.Abs(pose.shoulderOverElbow-target.shoulderOverElbow)<=.10f &&
        Vector3.Angle(pose.shoulderAxis,target.shoulderAxis)<=25 &&
        Vector3.Angle(pose.pelvisAxis,target.pelvisAxis)<=25 &&
        Mathf.Abs(Vector3.Angle(pose.shoulderAxis,pose.pelvisAxis)-Vector3.Angle(target.shoulderAxis,target.pelvisAxis))<=20;

    public float Quality(Pose pose, Pose target)
    {
        // No free-arm or bilateral-symmetry term: a side bridge is intentionally asymmetric.
        float trunk=Score(Mathf.Abs(pose.trunkAngle-target.trunkAngle),4,25);
        float leg=.5f*Score(Mathf.Abs(pose.kneeAngle-target.kneeAngle),7,35)+
            .25f*Score(Vector3.Angle(pose.thigh,target.thigh),8,35)+
            .25f*Score(Vector3.Angle(pose.shin,target.shin),8,35);
        float support=.5f*Score(Mathf.Abs(pose.elbowAngle-target.elbowAngle),10,40)+
            .5f*Score(Mathf.Abs(pose.shoulderOverElbow-target.shoulderOverElbow),.025f,.16f);
        float roll=.5f*Score(Mathf.Abs(pose.shoulderStack-target.shoulderStack),.07f,.35f)+
            .5f*Score(Mathf.Abs(pose.pelvisStack-target.pelvisStack),.07f,.35f);
        return .45f*trunk+.25f*leg+.2f*support+.1f*roll;
    }

    public static bool TryMeasure(Animator avatar, out Pose pose)
    {
        pose=null;
        if(avatar==null || !avatar.isHuman) return false;
        var hip=avatar.GetBoneTransform(HumanBodyBones.Hips);
        var chest=avatar.GetBoneTransform(HumanBodyBones.Chest)??avatar.GetBoneTransform(HumanBodyBones.Spine);
        var ls=avatar.GetBoneTransform(HumanBodyBones.LeftUpperArm);var rs=avatar.GetBoneTransform(HumanBodyBones.RightUpperArm);
        var elbow=avatar.GetBoneTransform(HumanBodyBones.LeftLowerArm);var wrist=avatar.GetBoneTransform(HumanBodyBones.LeftHand);
        var lt=avatar.GetBoneTransform(HumanBodyBones.LeftUpperLeg);var rt=avatar.GetBoneTransform(HumanBodyBones.RightUpperLeg);
        var knee=avatar.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
        var lf=avatar.GetBoneTransform(HumanBodyBones.LeftFoot);var rf=avatar.GetBoneTransform(HumanBodyBones.RightFoot);
        if(hip==null || chest==null || ls==null || rs==null || elbow==null || wrist==null || lt==null || rt==null || knee==null || lf==null || rf==null) return false;
        Vector3 shoulder=(ls.position+rs.position)*.5f,feet=(lf.position+rf.position)*.5f;
        float span=Vector3.Distance(shoulder,feet);
        Vector3 support=lf.position-elbow.position;
        Vector3 along=Vector3.ProjectOnPlane(feet-shoulder,Vector3.up);
        if(span<.1f || support.sqrMagnitude<.01f || along.sqrMagnitude<.01f) return false;
        along.Normalize();Vector3 across=Vector3.Cross(Vector3.up,along);
        Vector3 Local(Vector3 v) { v.Normalize();return new Vector3(Vector3.Dot(v,along),Vector3.Dot(v,Vector3.up),Vector3.Dot(v,across)); }
        // A line through the supporting elbow and ankle remains well defined when
        // the feet overlap. The old chest/two-feet plane degenerates in this pose.
        Vector3 onSupport=elbow.position+support*Vector3.Dot(hip.position-elbow.position,support)/support.sqrMagnitude;
        pose=new Pose {
            lift=Vector3.Dot(hip.position-onSupport,Vector3.up)/span,
            trunkAngle=Vector3.Angle(chest.position-hip.position,lf.position-hip.position),
            kneeAngle=Vector3.Angle(lt.position-knee.position,lf.position-knee.position),
            elbowAngle=Vector3.Angle(ls.position-elbow.position,wrist.position-elbow.position),
            shoulderStack=Vector3.Dot((rs.position-ls.position).normalized,Vector3.up),
            pelvisStack=Vector3.Dot((rt.position-lt.position).normalized,Vector3.up),
            torsoVertical=Mathf.Abs(Vector3.Dot((chest.position-hip.position).normalized,Vector3.up)),
            shoulderOverElbow=Vector3.ProjectOnPlane(ls.position-elbow.position,Vector3.up).magnitude/span,
            thigh=Local(knee.position-lt.position),shin=Local(lf.position-knee.position),
            shoulderAxis=Local(rs.position-ls.position),pelvisAxis=Local(rt.position-lt.position)
        };
        return Finite(pose.lift)&&Finite(pose.trunkAngle)&&Finite(pose.kneeAngle)&&Finite(pose.elbowAngle)&&Finite(pose.shoulderStack)&&Finite(pose.pelvisStack)&&Finite(pose.torsoVertical)&&Finite(pose.shoulderOverElbow);
    }
    static bool Finite(float value)=>!float.IsNaN(value)&&!float.IsInfinity(value);
    static float Score(float error,float perfect,float zero)
    { float t=Mathf.InverseLerp(perfect,zero,error); return 100*(1-t*t*(3-2*t)); }
}
