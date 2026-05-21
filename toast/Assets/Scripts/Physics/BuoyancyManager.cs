using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

using Random=UnityEngine.Random;

public class BouyancyManager : MonoBehaviour
{
    // Public parameters
    public WaterSurface waterSurface = null;
    public int numFloaters;
    public GameObject robot = null;
    public List<GameObject> floaterObjects;
    public Vector4 robotParams = new(0.5f, 0.5f, 0.5f, 0.5f);
    public Vector4 trashParams = new(0.5f, 0.5f, 0.5f, 0.5f);

    // List of internal objects to float
    List<GameObject> floatingObjects = new();
    List<Rigidbody> objectBodies = new();

    // Input job parameters
    NativeArray<float3> targetPositionBuffer;

    // Output job parameters
    NativeArray<float> heightBuffer;
    NativeArray<float> errorBuffer;
    NativeArray<float3> candidatePositionBuffer;
    NativeArray<int> stepCountBuffer;

    // Make into a singleton
    public static BouyancyManager instance;
    private void Awake()
    {
        if (instance == null)
        {
            instance = this;
        }
        else
        {
            Destroy(this);
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        Init();
    }

    void OnEnable()
    {
        Init();
        Debug.Log("running OnEnable again");
    }

    void Init() {
        Debug.Log("numFloaters " + numFloaters.ToString());
        // Allocate the buffers
        targetPositionBuffer = new NativeArray<float3>(numFloaters + 1, Allocator.Persistent);
        heightBuffer = new NativeArray<float>(numFloaters + 1, Allocator.Persistent);
        errorBuffer = new NativeArray<float>(numFloaters + 1, Allocator.Persistent);
        candidatePositionBuffer = new NativeArray<float3>(numFloaters + 1, Allocator.Persistent);
        stepCountBuffer = new NativeArray<int>(numFloaters + 1, Allocator.Persistent);

        floatingObjects = new();
        objectBodies = new();
        // Add robot to floating objects
        floatingObjects.Add(robot);
        objectBodies.Add(robot.GetComponent<Rigidbody>());

        // Add 10 bottles
        foreach (GameObject obj in floaterObjects) {
            // Vector3 pos = Random.onUnitSphere * 3;
            // pos.Scale(new Vector3(8, 0.2f, 12));
            // Vector3 rot = new(Random.Range(0, 360), Random.Range(0, 360), Random.Range(80, 100));
            // GameObject randomPrefab = trashPrefabs[Random.Range(0, trashPrefabs.Count)];
            // GameObject bottle = Instantiate(randomPrefab, pos, Quaternion.Euler(rot));
            // bottle.transform.parent = transform;
            floatingObjects.Add(obj);
            objectBodies.Add(obj.GetComponent<Rigidbody>());
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (waterSurface == null)
            return;
        // Try to get the simulation data if available
        WaterSimSearchData simData = new WaterSimSearchData();
        if (!waterSurface.FillWaterSearchData(ref simData))
            return;

        // Fill the input positions
        int numElements = numFloaters;
        for (int i = 0; i < numElements; ++i)
            targetPositionBuffer[i] = floatingObjects[i].transform.position;

        // Prepare the first band
        WaterSimulationSearchJob searchJob = new WaterSimulationSearchJob();

        // Assign the simulation data
        searchJob.simSearchData = simData;

        // Fill the input data
        searchJob.targetPositionBuffer = targetPositionBuffer;
        searchJob.startPositionBuffer = targetPositionBuffer;
        searchJob.maxIterations = 8;
        searchJob.error = 0.01f;

        searchJob.heightBuffer = heightBuffer;
        searchJob.errorBuffer = errorBuffer;
        searchJob.candidateLocationBuffer = candidatePositionBuffer;
        searchJob.stepCountBuffer = stepCountBuffer;

        // Schedule the job with one Execute per index in the results array and only 1 item per processing batch
        JobHandle handle = searchJob.Schedule(numElements, 1);
        handle.Complete();

        // Fill the input positions
        for (int i = 0; i < numElements; ++i)
        {
            if (i == 0) {
                GameObject fb = floatingObjects[i];
                Rigidbody rb = objectBodies[i];
                Vector4 objParams = i == 0 ? robotParams : trashParams;
                
                // if (fb.transform.position.y - 1 < heightBuffer[i]){
                float displacementMulti = (heightBuffer[i] - fb.transform.position.y + objParams.x) * objParams.y;
                rb.AddForceAtPosition(new Vector3(0f, Physics.gravity.y + displacementMulti, 0f), fb.transform.position, ForceMode.Force);
                // rb.AddForce(displacementMulti * -rb.velocity * objParams.z * Time.fixedDeltaTime, ForceMode.VelocityChange);
                // rb.AddTorque(displacementMulti * -rb.angularVelocity * objParams.z * Time.fixedDeltaTime, ForceMode.VelocityChange);
                // print($"{heightBuffer[i]} {Physics.gravity.y} {displacementMulti}");   
		    } else {
                floatingObjects[i].transform.position = new Vector3(floatingObjects[i].transform.position.x, heightBuffer[i], floatingObjects[i].transform.position.z);
            }
        }
    }

    private void OnDestroy()
    {
        targetPositionBuffer.Dispose();
        heightBuffer.Dispose();
        errorBuffer.Dispose();
        candidatePositionBuffer.Dispose();
        stepCountBuffer.Dispose();
    }
}