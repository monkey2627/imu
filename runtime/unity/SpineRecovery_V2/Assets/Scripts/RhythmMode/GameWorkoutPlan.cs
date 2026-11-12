using System;
using UnityEngine;

[Serializable]
public sealed class GameWorkoutAction
{
    public string action;
    public string difficulty = "standard";
    public int sets = 1;
    // Exact-dose variants use prescription reps; other games use their standard scene dose.
    public int reps;
}

[Serializable]
public sealed class GameWorkoutPlan
{
    public GameWorkoutAction[] actions;

    public static string SceneFor(string action, bool recordedCoach = false)
    {
        if (action == "hipthrust_newscore") return "HipThrust_RingBridge_NewScoring";
        if (action == "hipthrust_hold30") return "HipThrust_RingBridge_Hold30";
        if (action == "sidebridge") return "SideBridge_RingBridge_Dynamic";
        if (action == "sidebridge_hold30") return "SideBridge_RingBridge_Hold30";
        if (recordedCoach)
        {
            switch (action)
            {
                case "deadbug": return "DeadBug_RingSkinC_Recorded";
                case "birddog": return "BirdDog_RingSkinC_Recorded";
                case "hipthrust": return "HipThrust_RingBridge_Recorded";
                default: return null;
            }
        }
        switch (action)
        {
            case "deadbug": return "DeadBug_RingSkinC_Recorded";
            case "birddog": return "BirdDog_RingSkinC_Recorded";
            case "hipthrust": return "HipThrust_RingBridge_Recorded";
            default: return null;
        }
    }

    public static string ActionName(string action)
    {
        switch (action)
        {
            case "deadbug": return "死虫式";
            case "birddog": return "鸟狗式";
            case "hipthrust": return "臀桥";
            case "hipthrust_newscore": return "臀桥 · 新评分";
            case "hipthrust_hold30": return "臀桥 · 保持30秒";
            case "sidebridge": return "左侧桥 · 动态";
            case "sidebridge_hold30": return "左侧桥 · 保持30秒";
            default: return action;
        }
    }

    public static string DifficultyName(string difficulty) => difficulty == "simple" ? "简单" : difficulty == "hard" ? "困难" : "标准";

    public GameWorkoutPlan ValidatedCopy()
    {
        if (actions == null || actions.Length == 0 || actions.Length > 64)
            throw new ArgumentException("动作串必须包含 1–64 个动作。");
        var copy = new GameWorkoutPlan { actions = new GameWorkoutAction[actions.Length] };
        int total = 0;
        for (int i = 0; i < actions.Length; i++)
        {
            var item = actions[i];
            if (item == null || SceneFor(item.action) == null) throw new ArgumentException($"第 {i + 1} 个动作无效。");
            if (item.difficulty != "simple" && item.difficulty != "standard" && item.difficulty != "hard")
                throw new ArgumentException($"第 {i + 1} 个动作难度无效。");
            if (item.sets < 1 || item.sets > 100) throw new ArgumentException("每个动作的组数必须为 1–100。");
            if (RequiresExactReps(item.action) && (item.reps < 1 || item.reps > 100))
                throw new ArgumentException("该动作必须提供处方 reps（1–100）。");
            total += item.sets;
            copy.actions[i] = new GameWorkoutAction { action = item.action, difficulty = item.difficulty, sets = item.sets, reps = item.reps };
        }
        if (total > 1000) throw new ArgumentException("动作串总组数不能超过 1000。");
        return copy;
    }
    public static bool RequiresExactReps(string action) => action=="hipthrust_hold30" || action=="sidebridge" || action=="sidebridge_hold30";
}
