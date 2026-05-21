using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using RosMessageTypes.Std;
using RosMessageTypes.Sensor;
using RosMessageTypes.BuiltinInterfaces;
using System;
using Unity.Robotics.Core;

public class CameraPub : MonoBehaviour
{
    ROSConnection ros;
    public string imgTopicName = "/cam";
    public string infoTopicName = "/cam/info";
    public float frequency = 30;
    float tau;
    float timeElapsed = 0;
    new private Camera camera;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>(imgTopicName, 10);
        ros.RegisterPublisher<CameraInfoMsg>(infoTopicName, 10);
        camera = GetComponent<Camera>();
        tau = 1 / frequency;
    }

    // Update is called once per frame
    void Update()
    {
        timeElapsed += Time.deltaTime;
        if (timeElapsed > tau)
        {
            var rt = RenderTexture.active;
            RenderTexture.active = camera.targetTexture;
            Texture2D texture = new(camera.targetTexture.width, camera.targetTexture.height, TextureFormat.RGB24, true);
            texture.ReadPixels(new Rect(0, 0, camera.targetTexture.width, camera.targetTexture.height), 0, 0);
            texture.Apply();
            RenderTexture.active = rt;
            // Graphics.Blit(null, rt);
            ImageMsg msg = texture.ToImageMsg(generateHeader("camera_link"));
            DestroyImmediate(texture);
            ros.Publish(imgTopicName, msg);
            
            // Publish camera info
            CameraInfoMsg infoMsg = CameraInfoGenerator.ConstructCameraInfoMessage(camera, generateHeader("meow"));
            ros.Publish(infoTopicName, infoMsg);;
            
            timeElapsed = 0;
        }
    }

    static HeaderMsg generateHeader(string frameID) {
        HeaderMsg headerMsg = new();
        headerMsg.frame_id = frameID;
        headerMsg.stamp = Clock.GetMsg();
        return headerMsg;
    }
}