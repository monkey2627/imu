using System;
using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;

// Main-thread integration boundary. No dependency on the phone/session protocol.
public sealed class GameWorkoutSession : MonoBehaviour
{
    private static GameWorkoutSession instance;
    public static GameWorkoutSession Instance
    {
        get
        {
            if (instance == null) instance = new GameObject("Game Workout Session").AddComponent<GameWorkoutSession>();
            return instance;
        }
    }
    public event Action Changed;
    public event Action Finished;
    public event Action<string> Failed;
    public bool IsRunning { get; private set; }
    public bool IsComplete { get; private set; }
    public int ActionIndex { get; private set; }
    public int CompletedSets { get; private set; }
    public string Status { get; private set; } = "未设置动作串";
    public GameWorkoutPlan Plan => plan?.ValidatedCopy();
    private GameWorkoutPlan plan;
    private GameWorkoutScene scene;
    private Coroutine transition;
    private bool awaitingCompletion;
    private bool startingSet;
    private bool useRecordedCoachAnimations;
    private string hipThrustSceneOverride;
    private string SceneFor(string action) => action == "hipthrust" && !string.IsNullOrEmpty(hipThrustSceneOverride)
        ? hipThrustSceneOverride : GameWorkoutPlan.SceneFor(action, useRecordedCoachAnimations);

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetStatic() { instance = null; }

    private void Awake()
    {
        if (instance != null && instance != this) { Destroy(gameObject); return; }
        instance = this;
        DontDestroyOnLoad(gameObject);
    }

    public bool SetPlanJson(string json, out string error)
    {
        try { return SetPlan(JsonUtility.FromJson<GameWorkoutPlan>(json), out error); }
        catch (Exception ex) { error = ex.Message; return false; }
    }

    public bool SetPlan(GameWorkoutPlan value, out string error)
    {
        error = null;
        if (IsRunning) { error = "请先停止当前动作串。"; return false; }
        try
        {
            var validated = value?.ValidatedCopy() ?? throw new ArgumentException("动作串不能为空。");
            plan = validated;
            ActionIndex = CompletedSets = 0;
            IsComplete = false;
            Status = "动作串已就绪";
            Changed?.Invoke();
            return true;
        }
        catch (Exception ex) { error = ex.Message; return false; }
    }

    public bool StartPlan(out string error)
    {
        error = null;
        if (IsRunning || plan == null) { error = IsRunning ? "动作串已在运行。" : "请先传入动作串参数。"; return false; }
        if (scene != null && scene.IsPlaying) { error = "请先结束当前单项训练。"; return false; }
        useRecordedCoachAnimations = scene != null && scene.useRecordedCoachAnimations;
        hipThrustSceneOverride = scene != null ? scene.hipThrustSceneOverride : null;
        foreach (var item in plan.actions)
            if (!Application.CanStreamedLevelBeLoaded(SceneFor(item.action)))
            { error = "动作场景未加入构建：" + item.action; return false; }
        ActionIndex = CompletedSets = 0;
        IsRunning = true;
        IsComplete = false;
        transition = StartCoroutine(PrepareNextSet());
        return true;
    }

    public void Cancel()
    {
        IsRunning = awaitingCompletion = startingSet = false;
        if (transition != null) StopCoroutine(transition);
        transition = null;
        if (scene != null) scene.StopTraining();
        Status = "动作串已停止";
        Changed?.Invoke();
    }

    public void Attach(GameWorkoutScene value) { scene = value; Changed?.Invoke(); }
    public void Detach(GameWorkoutScene value) { if (scene == value) scene = null; }
    public bool AllowsStart(GameWorkoutScene sender) => !IsRunning || (sender == scene && startingSet);

    public void OnSetStarted(GameWorkoutScene sender)
    {
        if (!IsRunning || sender != scene || !startingSet) return;
        awaitingCompletion = true;
        startingSet = false;
        Status = $"第 {CompletedSets + 1} / {plan.actions[ActionIndex].sets} 组";
        Changed?.Invoke();
    }

    public void OnSetCompleted(GameWorkoutScene sender)
    {
        if (!IsRunning || sender != scene || !awaitingCompletion) return;
        awaitingCompletion = false;
        CompletedSets++;
        if (CompletedSets >= plan.actions[ActionIndex].sets)
        {
            ActionIndex++;
            CompletedSets = 0;
        }
        if (ActionIndex == plan.actions.Length)
        {
            IsRunning = false;
            IsComplete = true;
            Status = "动作串全部完成";
            Changed?.Invoke();
            Finished?.Invoke();
        }
        else transition = StartCoroutine(PrepareNextSet());
    }

    private IEnumerator PrepareNextSet()
    {
        // Let completion listeners finish and show the result before transitioning.
        Status = "准备下一组动作";
        Changed?.Invoke();
        yield return new WaitForSecondsRealtime(3f);
        string target = SceneFor(plan.actions[ActionIndex].action);
        if (SceneManager.GetActiveScene().name != target)
        {
            AsyncOperation load = SceneManager.LoadSceneAsync(target);
            if (load == null) { Abort("无法加载动作场景。"); yield break; }
            yield return load;
        }
        yield return null;
        if (scene == null || scene.gameObject.scene.name != target) { Abort("动作串场景入口缺失。"); yield break; }
        scene.PrepareMenu(plan.actions[ActionIndex].difficulty);
        if (!scene.PrepareDose(plan.actions[ActionIndex], out string doseError)) { Abort(doseError); yield break; }
        Status = scene.Scorer.IsCalibrationBypassed ? "流程预览：已临时跳过校准" : "请按提示完成躺姿校准";
        Changed?.Invoke();
        while (IsRunning && scene != null && !scene.Scorer.CanStartTraining) yield return null;
        if (!IsRunning) yield break;
        if (scene == null || scene.gameObject.scene.name != target) { Abort("动作场景已离开。"); yield break; }
        startingSet = true;
        scene.Menu.StartTraining();
        if (!awaitingCompletion) { Abort("当前动作未能开始。"); yield break; }
        transition = null;
    }

    private void Update()
    {
        if (IsRunning && awaitingCompletion && (scene == null || scene.Menu.IsTrainingMenuVisible))
            Cancel();
    }

    private void Abort(string error)
    {
        IsRunning = awaitingCompletion = startingSet = false;
        transition = null;
        Status = error;
        Changed?.Invoke();
        Failed?.Invoke(error);
    }
}
