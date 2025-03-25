using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class PositionLogger : MonoBehaviour
{
    public Transform headTransform;
    public Transform leftControllerTransform;
    public Transform rightControllerTransform;
    private string filePath;

    // Start is called before the first frame update
    void Start()
    {
        AssignTransforms();
        // filePath = Path.Combine(Application.persistentDataPath, "position_log.txt");
        // filePath = @"C:\Users\chen2\GaussianSplattingVRViewerUnity-v1.1\trace\position_log.txt";
        /*filePath = @"D:\Viewer\GaussianSplattingVRViewerUnity-v1.1\trace\position_log.txt";*/
        filePath = @"D:\Viewer\GaussianSplattingVRViewerUnity-v1.1\GaussianSplattingVRViewer\collect\trace\position_log.txt";
        InvokeRepeating("LogPositionAndRotation", 0, 1.0f); // Log every second
    }

    // Update is called once per frame
    void Update()
    {
        LogPositionAndRotation();
    }

    void AssignTransforms()
    {
        // Find the XR Rig
        GameObject xrOrigin = GameObject.Find("XR Origin (XR Rig)");
        Debug.Log("XR Origin");
        if (xrOrigin != null)
        {
            Debug.Log("XR Origin found.");
            headTransform = xrOrigin.transform.Find("Camera Offset/Main Camera");
            leftControllerTransform = xrOrigin.transform.Find("Camera Offset/Left Controller");
            rightControllerTransform = xrOrigin.transform.Find("Camera Offset/Right Controller");
        }
        else
        {
            Debug.LogError("XR Origin not found in the scene.");
        }
    }

    void LogPositionAndRotation()
    {
        if (headTransform == null || leftControllerTransform == null || rightControllerTransform == null)
        {
            Debug.LogError("Transforms not assigned. Make sure the XR Origin is set up correctly.");
            return;
        }

        string logEntry = $"{System.DateTime.Now:yyyy-MM-dd HH:mm:ss}, " +
                          $"Head Position: {headTransform.position}, Head Rotation: {headTransform.rotation.eulerAngles}, " +
                          $"Left Controller Position: {leftControllerTransform.position}, Left Controller Rotation: {leftControllerTransform.rotation.eulerAngles}, " +
                          $"Right Controller Position: {rightControllerTransform.position}, Right Controller Rotation: {rightControllerTransform.rotation.eulerAngles}";

        Debug.Log(logEntry);
        WriteToFile(logEntry);
    }

    void WriteToFile(string logEntry)
    {
        using (StreamWriter writer = new StreamWriter(filePath, true))
        {
            writer.WriteLine(logEntry);
        }
    }
}
