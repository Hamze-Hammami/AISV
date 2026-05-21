using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Publisher : MonoBehaviour
{
    // ROSConnection ros;
    public string topicName = "test";
    public float timeElapsed = 0;
    void Start()
    {
        // ros = ROSConnection.GetOrCreateInstance();
        // ros.RegisterPublisher<StringMsg>(topicName);
    }

    // Update is called once per frame
    void Update()
    {
        timeElapsed += Time.deltaTime;
        if (timeElapsed > 0.5f) {
            // StringMsg msg = new("lol " + Time.deltaTime.ToString());
            // ros.Publish(topicName, msg);
            timeElapsed = 0;
        }
    }
}
