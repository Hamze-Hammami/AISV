using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Rendering;

public class Waves : MonoBehaviour
{

    public float upthrust = 1.5f;
    Rigidbody parentRigidbody;
    public bool underwater;
    public float kF = 400;

    public float WaterDrag = 3;
    public float WaterDragAngular = 1;
    public float AirDrag = 0;
    public float AirDragAngular = 0.05f;
    // const float k = (float) Math.PI / 5;
    // const float c = 3.949327082f;
    // float timeAccum = 0;

    public Transform[] floaters;
    int floatersUnderwater;

    void Start()
    {
        parentRigidbody = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        parentRigidbody.AddForceAtPosition(new Vector3(0f, Physics.gravity.y, 0f), transform.position, ForceMode.Acceleration);
        // timeAccum += Time.deltaTime;
        floatersUnderwater = 0;
        foreach (Transform floater in floaters)
        {
            print(this.gameObject.name);
            float waterHeight = 0;
            float difference = floater.position.y - waterHeight;
            if (difference < 0)
            {
                parentRigidbody.AddForceAtPosition(Vector3.up * Math.Abs(difference) * kF, floater.position, ForceMode.Force);
                floatersUnderwater += 1;
                if (!underwater)
                {
                    underwater = true;
                    SwitchDrag();
                }
            }
        }
        if (underwater && floatersUnderwater == 0)
        {
            underwater = false;
            SwitchDrag();
        }
    }

    // float CalculateHeight() {
    //     Vector3[] verts = ocean.GetComponent<MeshFilter>().mesh.vertices;
    //     print(verts[0]);
    //     float f = k * (transform.position.x - c * timeAccum);
    //     return 0.795774715f * (float) Math.Cos(f);
    // }

    void SwitchDrag()
    {
        if (underwater)
        {
            parentRigidbody.drag = WaterDrag;
            parentRigidbody.angularDrag = WaterDragAngular;
        }
        else
        {
            parentRigidbody.drag = AirDrag;
            parentRigidbody.angularDrag = AirDragAngular;
        }
    }

    // void OnTriggerEnter(Collider obj)
    // {
    //     print(obj.name);

    //     if (obj.name == "Plane") {
    //         force = new Vector3(0 , upthrust, 0);
    //         thrustFramesElapsed = 0f;
    //     }
    // }
}
