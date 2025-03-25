using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class SimulateTrace : MonoBehaviour
{
    private Queue<List<float>> dataQueue;
    private string filePath = @"D:\Viewer\GaussianSplattingVRViewerUnity-v1.1\GaussianSplattingVRViewer\replay\trace\output.txt"; // Replace with your actual file path

    /*   public Transform xrOriginTransform; // Assign this via the Inspector*/
    private GameObject user;
 /*   private Vector3 currentPosition;
    private Quaternion currentRotation;*/

    // Start is called before the first frame update
    void Start()
    {
        // Initialize the queue
        dataQueue = new Queue<List<float>>();

        // Read the file and fill the queue
        ReadFileToQueue(filePath);

        // Ensure the xrOriginTransform is assigned
        /*if (xrOriginTransform == null)
        {
            Debug.LogError("XR Origin Transform is not assigned.");
        }*/

        user = GameObject.Find("Main Camera");
    }

    // Update is called once per frame
    void Update()
    {
        // Process the next line of data if available
        if (dataQueue.Count > 0)
        {
            List<float> data = dataQueue.Dequeue();
            ProcessData(data);
        }
    }

    void ReadFileToQueue(string filePath)
    {
        try
        {
            using (StreamReader reader = new StreamReader(filePath))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    string[] parts = line.Split(',');
                    List<float> floatList = new List<float>();

                    foreach (string part in parts)
                    {
                        if (float.TryParse(part, out float number))
                        {
                            floatList.Add(number);
                        }
                        else
                        {
                            Debug.LogWarning($"Unable to parse '{part}' as a float.");
                        }
                    }

                    // Check if the line has exactly 6 values (3 for position and 3 for rotation)
                    if (floatList.Count == 6)
                    {
                        dataQueue.Enqueue(floatList);
                    }
                    else
                    {
                        Debug.LogWarning("Line does not contain exactly 6 float values.");
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogError("An error occurred while reading the file:");
            Debug.LogError(ex.Message);
        }
    }

    void ProcessData(List<float> data)
    {
        /*if (xrOriginTransform == null)
        {
            Debug.LogError("XR Origin Transform is not assigned.");
            return;
        }*/

        // Extract position and rotation from the data
        Vector3 position = new Vector3(data[0], data[1], data[2]);
        Vector3 rotationEuler = new Vector3(data[3], data[4], data[5]);
        Quaternion rotation = Quaternion.Euler(rotationEuler);

        // Assign position and rotation to the XR Origin
        /*xrOriginTransform.position = -position;
        xrOriginTransform.rotation = Quaternion.Inverse(rotation);*/
        /*this.transform.position = position - user.transform.position;
        this.transform.rotation = Quaternion.Inverse(user.transform.rotation) * rotation;*/
        Debug.Log($"Position: {this.transform.position}, Rotation: {this.transform.rotation}");
        this.transform.position += position - user.transform.position;
        this.transform.rotation = rotation * Quaternion.Inverse(user.transform.rotation) * this.transform.rotation;

        Debug.Log($"Modified Position: {this.transform.position}, Modified Rotation: {this.transform.rotation}");
        // For debugging purposes
        /*Debug.Log($"Position: {position}, Rotation: {rotationEuler}");*/
    }
}
