using UnityEngine;

public class TestJsonLoader : MonoBehaviour
{
    public void LoadDeadBugJson()
    {
        // 把导出的 JSON 路径填在这里（用你的实际路径）
        string jsonPath = @"E:\ARp\dead_bug.motion.json";

        PracticeSceneInitializer.PendingMotionPackageUri = jsonPath;
        UnityEngine.SceneManagement.SceneManager.LoadScene("testPractice");
    }

    public void LoadBirdDogJson()
    {
        string jsonPath = @"E:\ARp\bird_dog.motion.json";

        PracticeSceneInitializer.PendingMotionPackageUri = jsonPath;
        UnityEngine.SceneManagement.SceneManager.LoadScene("testPractice");
    }
}
