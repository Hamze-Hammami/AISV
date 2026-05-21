using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using RosMessageTypes.Std;
using RosMessageTypes.Sensor;
using RosMessageTypes.BuiltinInterfaces;
using Unity.Robotics.Core;

public class DepthCam : MonoBehaviour
{
    ROSConnection ros;
    public string topicName = "depthCam";
    public float frequency = 30;
    float tau;
    float timeElapsed = 0;
    new private Camera depthCamera;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>(topicName, 10);
        depthCamera = GetComponent<Camera>();
        tau = 1 / frequency;
        depthCamera.depthTextureMode = DepthTextureMode.Depth;
    }

    void Update()
    {
        if (depthCamera)
        {
            // Create depth image message
            ImageMsg depthImage = new ImageMsg();
            depthImage.header = new HeaderMsg();
            depthImage.header.stamp = Clock.GetMsg();
            depthImage.height = (uint) depthCamera.pixelHeight;
            depthImage.width = (uint) depthCamera.pixelWidth;
            depthImage.encoding = "16UC1";  // Encoding for depth image (16-bit unsigned)
            depthImage.is_bigendian = 0;

            // Capture the depth image from the camera
            RenderTexture.active = depthCamera.targetTexture;
            Texture2D depthTexture = new Texture2D(depthCamera.pixelWidth, depthCamera.pixelHeight, TextureFormat.R16, false);
            depthTexture.ReadPixels(new Rect(0, 0, depthCamera.pixelWidth, depthCamera.pixelHeight), 0, 0);
            depthTexture.Apply();

            // Convert to raw depth data (this will vary depending on the encoding)
            depthImage.data = depthTexture.GetRawTextureData();

            // Publish the depth image to the ROS topic
            ros.Publish(topicName, depthImage);
        }
    }

}
