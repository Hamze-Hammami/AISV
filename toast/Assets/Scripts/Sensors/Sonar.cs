using System;
using System.Linq;
using RosMessageTypes.Sensor;
using Unity.Collections;
using Unity.Jobs;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;
using Unity.Robotics.Core;

public class Sonar : MonoBehaviour
{
    public float minAngleDegrees = -30;
    public float maxAngleDegrees = 30;
    public float angleIncrementDegrees = 4f;
    public float minRange = 0.02f;
    public float maxRange = 5f;
    public float Hz = 100f;
    public string topicName = "scan";
    public string frameId = "lidar_link";
    public bool publishData = true;
    public bool drawRays = false;
    private Vector3 transformScale;
    private float timeSinceScan = 0.0f;
    private Vector3[] scanDirVectors;
    private RangeMsg msg = new RangeMsg();
    private ROSConnection ros;


    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<RangeMsg>(topicName);
        scanDirVectors = GenerateScanVectors();
    }

    void FixedUpdate(){
        timeSinceScan += Time.deltaTime;
        if (timeSinceScan < 1.0f/Hz){
            return;
        }

        transformScale = transform.lossyScale;
        scanDirVectors = GenerateScanVectors();
        float[] dists  =  PerformScan(scanDirVectors);
        if (publishData){
            msg = DistancesToLaserscan(dists);
            ros.Publish(topicName, msg);
        }
        timeSinceScan = 0.0f;
    }

    private Vector3[] GenerateScanVectors()
    {
        int numBeams = (int)((maxAngleDegrees-minAngleDegrees)/(angleIncrementDegrees));
        Debug.Assert(numBeams >= 0, "Number of beams is negative. Check min/max angle and angle increment.");
        Vector3[] scanVectors = new Vector3[numBeams];
        float minAngleRad = Mathf.Deg2Rad*minAngleDegrees;
        float angleIncrementRad = Mathf.Deg2Rad*angleIncrementDegrees;
        for (int i = 0; i < numBeams; i++)
        {
            float hRot = minAngleRad + angleIncrementRad*i;
            float x = -Mathf.Sin(hRot);
            float y = 0;
            float z = Mathf.Cos(hRot);
            scanVectors[i].x = x;
            scanVectors[i].y = y;
            scanVectors[i].z = z;
        }
        return scanVectors;
    }


    private float[] PerformScan(Vector3[] dirs)
    {
        int numPoints = dirs.Length;
        var commands = new NativeArray<RaycastCommand>(numPoints, Allocator.TempJob);
        var results = new NativeArray<RaycastHit>(numPoints, Allocator.TempJob);

        for (int i = 0; i < numPoints; i++)
        {
            Vector3 origin = transform.position;
            Vector3 direction = transform.rotation * dirs[i];
            commands[i] = new RaycastCommand(origin, direction, QueryParameters.Default, maxRange * 10);
        }

        int batchSize = 500;
        JobHandle handle = RaycastCommand.ScheduleBatch(commands, results, batchSize, 1);
        handle.Complete();

        float[] dists = new float[numPoints];
        for (int i = 0; i < numPoints; i++)
        {
            var hit = results[i];
            if (hit.collider != null && (transform.position - hit.point).sqrMagnitude > minRange * minRange * 100)
            {
                Vector3 beam = transform.InverseTransformPoint(hit.point);
                if (hit.distance != 0) {
                    // Debug.Log(hit.distance);
                    dists[i] = hit.distance * 0.1f;
                }
                if (drawRays)
                {
                    Debug.DrawLine(transform.position, transform.TransformPoint(beam), Color.red);
                }
            }
            else{
                dists[i] = float.PositiveInfinity;
            }
        }

        results.Dispose();
        commands.Dispose();
        return dists;
    }

    private RangeMsg DistancesToLaserscan(float[] dists){
        RangeMsg msg = new RangeMsg();
        msg.header.frame_id = frameId;

        msg.header.stamp = Clock.GetMsg();

        msg.field_of_view = maxAngleDegrees*Mathf.Deg2Rad - minAngleDegrees*Mathf.Deg2Rad;
        msg.min_range = minRange;
        msg.max_range = maxRange;
        float minDist = float.PositiveInfinity;
        foreach (float d in dists) {
            if (d != float.NaN) {
                // Debug.Log(d);
                minDist = Math.Min(minDist, d);
            }
        }
        // Debug.Log(minDist);
        msg.range = minDist;
        msg.radiation_type = 0;
        // Debug.Log(string.Join(", ", dists));
        return msg;
    }
}