using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;

public class Thrusters : MonoBehaviour
{
    ROSConnection ros;

    public Transform[] thrusters = new Transform[2];
    public string thrustL = "/thruster_l";
    public string thrustR = "/thruster_r";

    public Vector3 thrustOffset = new();
    
    public float thrusterSpeed = 5;
    int thruster_l_data = 90;
    int thruster_r_data = 90;

    Rigidbody rb;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<Int32Msg>(thrustL, (msg) => {topicCB(msg, 'L');});
        ros.Subscribe<Int32Msg>(thrustR, (msg) => {topicCB(msg, 'R');});

        rb = GetComponent<Rigidbody>();
    }

    void FixedUpdate()
    {
        rb.AddForceAtPosition(((float) thruster_r_data / 90 - 1) * 10 * new Vector3(transform.forward.x, 0, transform.forward.z) * thrusterSpeed , thrusters[0].position, ForceMode.Acceleration);
        rb.AddForceAtPosition(((float) thruster_l_data / 90 - 1) * 10 * new Vector3(transform.forward.x, 0, transform.forward.z) * thrusterSpeed , thrusters[1].position, ForceMode.Acceleration);
    }

    void topicCB(Int32Msg msg, char side) {
        if (side == 'L') {
            thruster_l_data = msg.data;
        } else {
            thruster_r_data = msg.data;
        }
        
    }
}