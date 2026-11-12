using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;

/// <summary>Inspects the actual patient rig, including its bind pose, rather than only packet delivery.</summary>
public static class UserRigVerification
{
    const string Output = "output/android-test/";
    static int checks;
    static readonly HumanBodyBones[] Required = {
        HumanBodyBones.Hips,HumanBodyBones.Spine,HumanBodyBones.Chest,HumanBodyBones.Head,
        HumanBodyBones.LeftShoulder,HumanBodyBones.RightShoulder,
        HumanBodyBones.LeftUpperArm,HumanBodyBones.RightUpperArm,HumanBodyBones.LeftLowerArm,HumanBodyBones.RightLowerArm,
        HumanBodyBones.LeftHand,HumanBodyBones.RightHand,HumanBodyBones.LeftUpperLeg,HumanBodyBones.RightUpperLeg,
        HumanBodyBones.LeftLowerLeg,HumanBodyBones.RightLowerLeg,HumanBodyBones.LeftFoot,HumanBodyBones.RightFoot,
        HumanBodyBones.LeftToes,HumanBodyBones.RightToes
    };
    static void Check(bool condition,string message) { checks++; if(!condition) throw new Exception("User rig: "+message); }

    // Run on authored scene poses before building, never on a live tracked pose.
    public static void ValidateBuildRigs()
    {
        for(int i=0;i<UnityEngine.SceneManagement.SceneManager.sceneCount;i++)
            if(UnityEngine.SceneManagement.SceneManager.GetSceneAt(i).isDirty)throw new Exception("Save scene changes before validating a build.");
        var setup=EditorSceneManager.GetSceneManagerSetup();
        try {
            foreach(var name in GameTestMenu.Scenes) {
                EditorSceneManager.OpenScene("Assets/Scenes/"+name+".unity");
                ValidateFrozenRig(UnityEngine.Object.FindObjectOfType<DeadBugGamingPoseScorer>(true).UserAnimator);
            }
        } finally {
            if(setup.Any(s=>s.isLoaded&&s.isActive))EditorSceneManager.RestoreSceneManagerSetup(setup);
            else EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
        }
    }

    static void ValidateFrozenRig(Animator animator)
    {
        Check(animator!=null&&animator.avatar!=null&&animator.avatar.isValid&&animator.isHuman,"invalid humanoid Avatar");
        var mapped=Required.Select(animator.GetBoneTransform).ToArray();
        Check(mapped.All(t=>t!=null)&&mapped.Distinct().Count()==mapped.Length,"missing/duplicated humanoid bones");
        foreach(var bone in Required) {
            var t=animator.GetBoneTransform(bone);
            Check(Quaternion.Angle(t.localRotation,Quaternion.identity)<.1f,"Freeze T-Pose required: "+bone);
            Check(Quaternion.Angle(Quaternion.Inverse(animator.transform.rotation)*t.rotation,Quaternion.identity)<.1f,"non-normalized ancestor: "+bone);
            var scale=t.lossyScale;var rootScale=animator.transform.lossyScale;
            Check(Vector3.Distance(scale,rootScale)<.001f,"bone has a different scale: "+bone);
            if(bone.ToString().StartsWith("Left"))Check(t.name.EndsWith("_l"),"left/right anatomical mapping: "+bone);
            if(bone.ToString().StartsWith("Right"))Check(t.name.EndsWith("_r"),"left/right anatomical mapping: "+bone);
        }
        Vector3 Pos(HumanBodyBones b)=>animator.transform.InverseTransformPoint(animator.GetBoneTransform(b).position);
        Check(Pos(HumanBodyBones.LeftHand).x<Pos(HumanBodyBones.LeftUpperArm).x-.2f&&Pos(HumanBodyBones.RightHand).x>Pos(HumanBodyBones.RightUpperArm).x+.2f,"arms not in T-Pose / mirrored");
        Check(Mathf.Abs(Pos(HumanBodyBones.LeftHand).y-Pos(HumanBodyBones.LeftUpperArm).y)<.04f&&Mathf.Abs(Pos(HumanBodyBones.RightHand).y-Pos(HumanBodyBones.RightUpperArm).y)<.04f,"arms not horizontal");
        Check(Pos(HumanBodyBones.LeftFoot).x<Pos(HumanBodyBones.RightFoot).x,"legs swapped");
        Check(Pos(HumanBodyBones.LeftToes).z>Pos(HumanBodyBones.LeftFoot).z+.03f&&Pos(HumanBodyBones.RightToes).z>Pos(HumanBodyBones.RightFoot).z+.03f,"feet point backwards");
        foreach(var mesh in animator.GetComponentsInChildren<SkinnedMeshRenderer>(true))Check(BindClosureError(mesh)<.002f,"skin bind matrices disagree with rest pose: "+mesh.name);
        Check(animator.GetComponentsInChildren<Renderer>(true).Length>0,"model has no visible geometry");
    }
    public static void Inspect()
    {
        Directory.CreateDirectory(Output);
        checks=0;var report = new StringBuilder("Authored scene rig + real EVMC4U handler tests; synthetic poses, no physical trackers.\n");
        foreach (string name in GameTestMenu.Scenes)
        {
            EditorSceneManager.OpenScene("Assets/Scenes/" + name + ".unity");
            var patient = UnityEngine.Object.FindObjectOfType<DeadBugGamingPoseScorer>(true).UserAnimator;
            ValidateFrozenRig(patient);
            var avatarPath=AssetDatabase.GetAssetPath(patient.avatar);
            report.AppendLine("SCENE " + name + " MODEL " + patient.name + " AVATAR " + (string.IsNullOrEmpty(avatarPath)?"(scene embedded)":avatarPath));
            var clone = UnityEngine.Object.Instantiate(patient.gameObject);
            clone.SetActive(true);
            clone.transform.SetParent(null); clone.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            clone.transform.localScale = Vector3.one;
            var animator = clone.GetComponent<Animator>(); animator.enabled = false;
            foreach (var component in clone.GetComponentsInChildren<MonoBehaviour>(true)) component.enabled = false;
            foreach (var t in clone.GetComponentsInChildren<Transform>(true)) t.gameObject.layer = 31;
            var meshes = clone.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            var rigidMeshes=clone.GetComponentsInChildren<MeshFilter>(true);
            report.AppendLine($"Geometry: {meshes.Length} skinned meshes, {rigidMeshes.Length} rigid meshes.");
            foreach(var mesh in meshes)report.AppendLine($"  {mesh.name}: vertices={mesh.sharedMesh.vertexCount}, bones={mesh.bones.Length}, binds={mesh.sharedMesh.bindposes.Length}, all bones in model={mesh.bones.All(b=>b==null||b.IsChildOf(clone.transform))}");
            foreach (var mesh in meshes) { mesh.gameObject.SetActive(true); mesh.enabled = true; mesh.updateWhenOffscreen = true; }
            var before = meshes.Select(Bake).ToArray();
            for (int i=0; i<(int)HumanBodyBones.LastBone; i++)
            {
                var bone = animator.GetBoneTransform((HumanBodyBones)i); if(bone==null) continue;
                report.AppendLine($"{(HumanBodyBones)i}: {bone.name}; local-angle={Quaternion.Angle(bone.localRotation,Quaternion.identity):F4}; root-angle={Quaternion.Angle(bone.rotation,Quaternion.identity):F4}; position={bone.position:F4}; scale={bone.lossyScale:F4}");
            }
            report.AppendLine("Mesh bind closure max " + meshes.Select(BindClosureError).DefaultIfEmpty(0).Max());
            if(name==GameTestMenu.Scenes[0]) Capture(clone, name+"-authored");
            for(int i=0;i<(int)HumanBodyBones.LastBone;i++) { var bone=animator.GetBoneTransform((HumanBodyBones)i);if(bone!=null)bone.localRotation=Quaternion.identity; }
            float displacement=0;
            for(int i=0;i<meshes.Length;i++) { var after=Bake(meshes[i]); for(int v=0;v<after.Length;v++)displacement=Mathf.Max(displacement,Vector3.Distance(before[i][v],after[v])); }
            report.AppendLine("Neutral VMC max skin vertex displacement from authored pose: "+displacement);
            Check(displacement<.00001f,"neutral VMC changes the rest skin");
            if(name==GameTestMenu.Scenes[0])Capture(clone,name+"-neutral");
            VerifyReceiver(animator,name,report);
            // Regression: the gate must reject a hand that was not frozen, not silently accept it.
            var hand=animator.GetBoneTransform(HumanBodyBones.LeftHand);hand.localRotation=Quaternion.Euler(0,0,30);
            bool rejected=false;try{ValidateFrozenRig(animator);}catch(Exception){rejected=true;}
            Check(rejected,"non-frozen wrist regression was not detected");
            report.AppendLine("PASS: normalized rest, anatomical left/right mapping, forward feet, isolated joints, preserved geometry, no double application, non-frozen wrist rejected.");
            UnityEngine.Object.DestroyImmediate(clone);
        }
        report.AppendLine("Assertions: "+checks);
        File.WriteAllText(Output+"user-rig-inspection.txt",report.ToString());
    }
    static void VerifyReceiver(Animator animator,string scene,StringBuilder report)
    {
        var host=new GameObject("Rig receiver verification");var receiver=host.AddComponent<EVMC4U.ExternalReceiver>();
        receiver.Model=animator.gameObject;receiver.BonePositionSynchronize=false;receiver.RootPositionSynchronize=false;receiver.RootRotationSynchronize=false;receiver.RootScaleOffsetSynchronize=false;
        receiver.BonePositionFilterEnable=false;receiver.BoneRotationFilterEnable=false;receiver.CutBonesEnable=false;
        receiver.Start();receiver.Process();
        var transforms=animator.GetComponentsInChildren<Transform>(true);
        var positions=transforms.Select(t=>t.localPosition).ToArray();var scales=transforms.Select(t=>t.localScale).ToArray();
        void Send(Dictionary<HumanBodyBones,Quaternion> pose) {
            foreach(var bone in Required) {
                var q=pose.TryGetValue(bone,out var value)?value:Quaternion.identity;
                var message=new uOSC.Message {address="/VMC/Ext/Bone/Pos",values=new object[]{bone.ToString(),100f,-100f,100f,q.x,q.y,q.z,q.w}};
                receiver.MessageDaisyChain(ref message,0);
            }
            receiver.Process();
        }
        var neutral=new Dictionary<HumanBodyBones,Quaternion>();Send(neutral);
        var restVertices=WorldVertices(animator.gameObject);
        // Independently rotate every tracked anatomical joint, including both wrists, ankles and toes.
        foreach(var moving in Required) {
            var rotation=Quaternion.Euler(17,29,-23);Send(new Dictionary<HumanBodyBones,Quaternion>{{moving,rotation}});
            for(int i=0;i<transforms.Length;i++) {
                Check(Vector3.Distance(transforms[i].localPosition,positions[i])<.00001f,"incoming positions stretched the skeleton");
                Check(Vector3.Distance(transforms[i].localScale,scales[i])<.00001f,"incoming pose scaled a bone");
            }
            foreach(var observed in Required) {
                var target=animator.GetBoneTransform(observed);
                Check(Quaternion.Angle(target.localRotation,observed==moving?rotation:Quaternion.identity)<.05f,"wrong target or duplicate rotation: "+moving+" -> "+observed);
            }
            var vertices=WorldVertices(animator.gameObject);
            Check(vertices.All(v=>!float.IsNaN(v.x)&&!float.IsInfinity(v.x)&&!float.IsNaN(v.y)&&!float.IsInfinity(v.y)&&!float.IsNaN(v.z)&&!float.IsInfinity(v.z)),"non-finite mesh vertices");
            float displacement=vertices.Zip(restVertices,(a,b)=>Vector3.Distance(a,b)).Max();
            report.AppendLine($"  {moving}: max vertex movement={displacement:F5} m");
            Check(displacement>.001f,"bone rotated but visible geometry did not move: "+moving);
            Check(displacement<2,"mesh exploded from a moderate joint rotation");
            Send(neutral);
            Check(WorldVertices(animator.gameObject).Zip(restVertices,(a,b)=>Vector3.Distance(a,b)).Max()<.00001f,"reset accumulated pose/mesh drift");
        }
        // Anatomically useful asymmetric posture: left elbow flexion, right knee flexion, wrist and ankle offsets.
        Send(new Dictionary<HumanBodyBones,Quaternion> {
            {HumanBodyBones.LeftUpperArm,Quaternion.Euler(0,0,70)}, {HumanBodyBones.RightUpperArm,Quaternion.Euler(0,0,-70)},
            {HumanBodyBones.LeftLowerArm,Quaternion.Euler(0,70,0)}, {HumanBodyBones.RightLowerLeg,Quaternion.Euler(65,0,0)},
            {HumanBodyBones.LeftHand,Quaternion.Euler(0,0,-15)}, {HumanBodyBones.RightFoot,Quaternion.Euler(-15,0,0)}
        });
        if(scene==GameTestMenu.Scenes[0])Capture(animator.gameObject,scene+"-asymmetric");
        Send(neutral);
        // Replay real exercise poses through VMC and compare their rendered geometry
        // with Unity's independent Humanoid retargeter on the same patient proportions.
        var coach=UnityEngine.Object.FindObjectOfType<DeadBugGamingPoseScorer>(true).CoachAnimator;
        var clips=coach.runtimeAnimatorController.animationClips.Where(c=>c.humanMotion&&c.length>.1f).Distinct().ToArray();
        Check(clips.Length>0,"missing exercise reference clips");
        var template=UnityEngine.Object.Instantiate(animator.gameObject);var templateAnimator=template.GetComponent<Animator>();templateAnimator.enabled=true;
        var templateTransforms=template.GetComponentsInChildren<Transform>(true);var templatePositions=templateTransforms.Select(t=>t.localPosition).ToArray();
        foreach(var transform in templateTransforms)transform.gameObject.layer=30;
        var graph=PlayableGraph.Create("User rig exercise verification");graph.SetTimeUpdateMode(DirectorUpdateMode.Manual);
        var output=AnimationPlayableOutput.Create(graph,"Patient pose",templateAnimator);
        try {
            int samples=0;
            foreach(var clip in clips) {
                var playable=AnimationClipPlayable.Create(graph,clip);playable.SetApplyFootIK(false);playable.SetApplyPlayableIK(false);playable.SetSpeed(0);output.SetSourcePlayable(playable);graph.Play();
                foreach(float fraction in new[]{.2f,.5f,.8f}) {
                    playable.SetTime(clip.length*fraction);graph.Evaluate(0);
                    // VMC intentionally keeps this patient's proportions and fixed scene placement.
                    for(int i=0;i<templateTransforms.Length;i++)templateTransforms[i].localPosition=templatePositions[i];
                    var expected=WorldVertices(template);
                    Send(Required.ToDictionary(b=>b,b=>templateAnimator.GetBoneTransform(b).localRotation));
                    float meshError=WorldVertices(animator.gameObject).Zip(expected,(a,b)=>Vector3.Distance(a,b)).Max();
                    Check(meshError<.0001f,"VMC diverges from retargeted exercise mesh: "+clip.name+" error="+meshError);
                    if(samples==1)Capture(animator.gameObject,scene+"-exercise");
                    samples++;
                }
                playable.Destroy();
            }
            report.AppendLine($"PASS {samples} exercise samples through VMC match independent Humanoid retargeting within 0.1 mm (rotation-only placement).");
        } finally { graph.Destroy();UnityEngine.Object.DestroyImmediate(template); }
        Send(neutral);
        report.AppendLine("PASS actual EVMC4U receiver: 20 independently addressed bones; rotation-only input (oversized sender positions ignored); reset restores all geometry.");
        UnityEngine.Object.DestroyImmediate(host);
    }
    static Vector3[] WorldVertices(GameObject model)
    {
        var result=new List<Vector3>();
        foreach(var mesh in model.GetComponentsInChildren<SkinnedMeshRenderer>(true))result.AddRange(Bake(mesh));
        foreach(var mesh in model.GetComponentsInChildren<MeshFilter>(true))if(mesh.sharedMesh!=null)result.AddRange(mesh.sharedMesh.vertices.Select(mesh.transform.TransformPoint));
        Check(result.Count>0,"no vertices tested");return result.ToArray();
    }
    static Vector3[] Bake(SkinnedMeshRenderer renderer)
    {
        var mesh=new Mesh();renderer.BakeMesh(mesh);var result=mesh.vertices.Select(renderer.transform.TransformPoint).ToArray();UnityEngine.Object.DestroyImmediate(mesh);return result;
    }
    static float BindClosureError(SkinnedMeshRenderer renderer)
    {
        float error=0;var binds=renderer.sharedMesh.bindposes;
        for(int i=0;i<binds.Length;i++)
        {
            var closure=renderer.transform.worldToLocalMatrix*renderer.bones[i].localToWorldMatrix*binds[i];
            for(int r=0;r<4;r++)for(int c=0;c<4;c++)error=Mathf.Max(error,Mathf.Abs(closure[r,c]-(r==c?1:0)));
        }
        return error;
    }
    static void Capture(GameObject model,string name)
    {
        // Bake the exact tested pose for edit-mode rendering; GPU skinning refreshes on player frames.
        var bakedObjects=new List<GameObject>();var bakedMeshes=new List<Mesh>();
        var skins=model.GetComponentsInChildren<SkinnedMeshRenderer>(true);
        foreach(var skin in skins) {
            var obj=new GameObject("Baked "+skin.name,typeof(MeshFilter),typeof(MeshRenderer));obj.layer=31;
            obj.transform.SetParent(skin.transform,false);var mesh=new Mesh();skin.BakeMesh(mesh);
            obj.GetComponent<MeshFilter>().sharedMesh=mesh;obj.GetComponent<MeshRenderer>().sharedMaterials=skin.sharedMaterials;
            skin.enabled=false;bakedObjects.Add(obj);bakedMeshes.Add(mesh);
        }
        var vertices=WorldVertices(model);var bounds=new Bounds(vertices[0],Vector3.zero);
        foreach(var vertex in vertices)bounds.Encapsulate(vertex);
        var lightObject=new GameObject("Rig verification light");var light=lightObject.AddComponent<Light>();light.type=LightType.Directional;light.intensity=1.3f;light.cullingMask=1<<31;light.transform.rotation=Quaternion.Euler(35,-20,0);
        var host=new GameObject("Rig verification camera");var camera=host.AddComponent<Camera>();camera.cullingMask=1<<31;camera.clearFlags=CameraClearFlags.SolidColor;camera.backgroundColor=new Color(.07f,.09f,.14f);camera.orthographic=true;camera.orthographicSize=Mathf.Max(bounds.extents.y,bounds.extents.x/1.5f)*1.15f;
        camera.transform.position=bounds.center+(name.EndsWith("-exercise")?new Vector3(3,2,4):new Vector3(0,0,4));camera.transform.LookAt(bounds.center);
        float projected=0;
        for(int x=-1;x<=1;x+=2)for(int y=-1;y<=1;y+=2)for(int z=-1;z<=1;z+=2) {
            var corner=camera.transform.InverseTransformPoint(bounds.center+Vector3.Scale(bounds.extents,new Vector3(x,y,z)));
            projected=Mathf.Max(projected,Mathf.Abs(corner.y),Mathf.Abs(corner.x)/1.5f);
        }
        camera.orthographicSize=projected*1.15f;
        var texture=RenderTexture.GetTemporary(1500,1000,24);var previous=RenderTexture.active;
        try { camera.targetTexture=texture;camera.Render();RenderTexture.active=texture;var image=new Texture2D(1500,1000,TextureFormat.RGB24,false);image.ReadPixels(new Rect(0,0,1500,1000),0,0);image.Apply();File.WriteAllBytes(Output+name+".png",image.EncodeToPNG());UnityEngine.Object.DestroyImmediate(image); }
        finally { RenderTexture.active=previous;camera.targetTexture=null;RenderTexture.ReleaseTemporary(texture);UnityEngine.Object.DestroyImmediate(host);UnityEngine.Object.DestroyImmediate(lightObject);foreach(var obj in bakedObjects)UnityEngine.Object.DestroyImmediate(obj);foreach(var mesh in bakedMeshes)UnityEngine.Object.DestroyImmediate(mesh);foreach(var skin in skins)skin.enabled=true; }
    }
}
