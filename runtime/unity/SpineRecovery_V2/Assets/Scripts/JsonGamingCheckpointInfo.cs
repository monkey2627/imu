using System;

[Serializable]
public struct JsonGamingCheckpointInfo
{
    public string label;
    public int index;
    public int count;
    public int setIndex;
    public int setCount;
    public int segmentIndex;
    public int segmentCount;
    public int repeatIndex;
    public int repeatCount;
    public float holdSeconds;

    public JsonGamingCheckpointInfo(
        string label,
        int index,
        int count,
        int setIndex,
        int setCount,
        int segmentIndex,
        int segmentCount,
        int repeatIndex,
        int repeatCount,
        float holdSeconds)
    {
        this.label = label;
        this.index = index;
        this.count = count;
        this.setIndex = setIndex;
        this.setCount = setCount;
        this.segmentIndex = segmentIndex;
        this.segmentCount = segmentCount;
        this.repeatIndex = repeatIndex;
        this.repeatCount = repeatCount;
        this.holdSeconds = holdSeconds;
    }
}
