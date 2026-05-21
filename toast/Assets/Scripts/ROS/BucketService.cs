using Unity.Robotics.ROSTCPConnector;
using UnityEngine;
using RosMessageTypes;
using RosMessageTypes.Std;
using System;
using Unity.Mathematics;

public class BucketService : MonoBehaviour
{
    SpawnerScript spawnerScript;

    public float xWindow = 4;
    public float zMin = 3;
    public float zMax = 6.5f;


    void Start()
    {
        ROSConnection.GetOrCreateInstance().ImplementService<TriggerRequest, TriggerResponse>("/collect_trash", ServiceCallback);
        spawnerScript = GameObject.Find("Spawner").GetComponent<SpawnerScript>();
        
    }

    private TriggerResponse ServiceCallback(TriggerRequest req) {
        print("service called!");
        
        Transform trashObjects = spawnerScript.GetScene().GetChild(0);
        foreach (Transform trash in trashObjects)
        {
            Vector3 pos = transform.position;
            Vector3 tPos = pos - trash.position;
            if (Math.Abs(tPos.x) <= xWindow && tPos.z >= zMin && tPos.z <= zMax) {
                Debug.Log(trash.gameObject.name + trash.position.ToString() + tPos.ToString());
                DestroyImmediate(trash.gameObject);
            } else {
                Debug.Log("Rejected: " + trash.gameObject.name + trash.position.ToString() + tPos.ToString() + pos);
            }
        }

        spawnerScript.UpdateBouyancy();
        return new TriggerResponse(true, ":D");
    }

}
