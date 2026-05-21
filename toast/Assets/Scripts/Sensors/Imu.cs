using System;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using Unity.Robotics.Core;

public class Imu : MonoBehaviour
{
    ROSConnection ros;
    public string topicName = "imu/raw";
    public string frameId = "imu_link";
    private Rigidbody imuBody;
    [Range(20.0f, 300.0f)]
    public float Hz = 50.0f;
    private ImuMsg msg;
    private float timeSincePublish;


    // Start is called before the first frame update
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImuMsg>(topicName, 60);
        timeSincePublish = 0.0f;
        imuBody = gameObject.GetComponent<Rigidbody>();
    }

    void Update()
    {
        timeSincePublish += Time.deltaTime;
        if (timeSincePublish < 1.0f/Hz){
            return;
        }

        msg = new ImuMsg(){
            header = new RosMessageTypes.Std.HeaderMsg(){
                frame_id = frameId
            },
            orientation = imuBody.transform.rotation.To<FLU>(),
            angular_velocity = imuBody.angularVelocity.To<FLU>()
        };
        msg.header.stamp = Clock.GetMsg();
        ros.Publish(topicName, msg);
        timeSincePublish = 0.0f;
    }
}