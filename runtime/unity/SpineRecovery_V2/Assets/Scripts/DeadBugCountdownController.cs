using System.Collections;
using TMPro;
using UnityEngine;

public class DeadBugCountdownController : MonoBehaviour
{
    [SerializeField] private GameObject countdownRoot;
    [SerializeField] private TMP_Text countdownText;
    [SerializeField] private CoachActionController coachActionController;
    [SerializeField] private ActionPreviewCard deadBugPreviewCard;
    [SerializeField] private bool countdownEnabled = true;
    [SerializeField] private int countdownSeconds = 5;
    [SerializeField] private float countdownStepSeconds = 1f;
    [SerializeField] private bool autoStartOnPlay;
    [SerializeField] private string autoStartActionStateName = "animation_pose";

    private Coroutine countdownRoutine;

    public bool IsCountingDown => countdownRoutine != null;
    public int SecondsRemaining { get; private set; }

    private void Awake()
    {
        ResolveReferences();
        HideCountdown();
    }

    private void Start()
    {
        if (autoStartOnPlay)
        {
            BeginCountdown(autoStartActionStateName);
        }
    }

    private void OnDisable()
    {
        if (countdownRoutine != null)
        {
            StopCoroutine(countdownRoutine);
            countdownRoutine = null;
        }

        HideCountdown();
        SecondsRemaining = 0;
    }

    public void BeginCountdown(string actionStateName)
    {
        ResolveReferences();

        if (deadBugPreviewCard != null)
        {
            deadBugPreviewCard.PauseAndRewindPreview();
        }

        if (countdownRoutine != null)
        {
            StopCoroutine(countdownRoutine);
        }

        countdownRoutine = StartCoroutine(CountdownThenPlay(actionStateName));
    }

    public void CancelCountdown()
    {
        if (countdownRoutine != null) StopCoroutine(countdownRoutine);
        countdownRoutine = null;
        SecondsRemaining = 0;
        HideCountdown();
    }

    private IEnumerator CountdownThenPlay(string actionStateName)
    {
        int safeCountdownSeconds = countdownEnabled ? Mathf.Max(0, countdownSeconds) : 0;
        float safeStepSeconds = Mathf.Max(0f, countdownStepSeconds);

        if (safeCountdownSeconds > 0 && countdownRoot != null && countdownText != null)
        {
            countdownRoot.SetActive(true);

            for (int second = safeCountdownSeconds; second > 0; second--)
            {
                countdownText.text = second.ToString();
                SecondsRemaining = second;
                yield return new WaitForSeconds(safeStepSeconds);
            }
        }

        SecondsRemaining = 0;

        HideCountdown();

        if (deadBugPreviewCard != null)
        {
            deadBugPreviewCard.PlayPreview();
        }

        if (coachActionController != null)
        {
            coachActionController.PlayAction(actionStateName);
        }

        countdownRoutine = null;
    }

    private void ResolveReferences()
    {
        if (countdownRoot == null || countdownText == null)
        {
            Transform[] sceneTransforms = FindObjectsOfType<Transform>(true);
            foreach (Transform sceneTransform in sceneTransforms)
            {
                if (sceneTransform.name != "CountDown")
                {
                    continue;
                }

                if (countdownRoot == null)
                {
                    countdownRoot = sceneTransform.gameObject;
                }

                if (countdownText == null)
                {
                    countdownText = sceneTransform.GetComponent<TMP_Text>();
                }

                break;
            }
        }

        if (coachActionController == null)
        {
            coachActionController = FindObjectOfType<CoachActionController>();
        }

        if (deadBugPreviewCard == null)
        {
            deadBugPreviewCard = GetComponentInChildren<ActionPreviewCard>(true);
        }
    }

    private void HideCountdown()
    {
        if (countdownRoot != null)
        {
            countdownRoot.SetActive(false);
        }
    }
}
