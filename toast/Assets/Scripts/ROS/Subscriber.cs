using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;

public class Subscriber : MonoBehaviour
{
    ROSConnection ros;
    public string topicName = "/turtle1/cmd_vel";
    public string poseTopic = "/robot/position";
    private TwistMsg moveCommand = new();
    public float thrustSpeed = 5;
    public float turningSpeed = 5;
    Rigidbody rb;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<TwistMsg>(topicName, moveCB);
        ros.RegisterPublisher<PoseStampedMsg>(poseTopic);
        rb = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        Quaternion rot = transform.rotation;
        Vector3 pos = transform.position;
        HeaderMsg header = new(new ((int)Time.fixedTime * 1000, (uint)Time.fixedTime * 1000000), "cam");
        PoseStampedMsg msg = new(header, new(new(pos.z, pos.x, pos.y), new(rot.z, rot.x, rot.y, rot.w)));
        ros.Publish(poseTopic, msg);
    }

    void moveCB(TwistMsg msg) {
        // float thrustCmd = thrustSpeed * ((float)moveCommand.linear.x);
        // float torqueCmd = turningSpeed * ((float)moveCommand.angular.z);
        // transform.position += thrustCmd * transform.forward;
        rb.AddForceAtPosition((float)msg.linear.x * transform.forward * thrustSpeed, transform.position - (transform.forward * 2), ForceMode.Acceleration);
        rb.AddTorque(-(float)msg.angular.z * transform.up, ForceMode.Acceleration);
    }
}