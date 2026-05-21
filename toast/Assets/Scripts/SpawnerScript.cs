using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class SpawnerScript : MonoBehaviour
{
    public GameObject[] trashPrefabs;
    public GameObject[] obstaclePrefabs;

    // Make into a singleton
    public static SpawnerScript instance;
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

    public void UpdateBouyancy() {
        // Attach objects to allow bouyancy
        Debug.Log("Updating bouyancy");
        
        GameObject gameObj = GameObject.Find("SimManager");
        BouyancyManager manager = gameObj.GetComponent<BouyancyManager>();
        gameObj.SetActive(false);

        List<GameObject> allObjs = new();
        foreach (Transform child in transform.GetChild(0)) {
            allObjs.Add(child.gameObject);
        }
        Debug.Log("Objects: " + allObjs.Count.ToString());
        foreach (Transform child in transform.GetChild(1)) {
            allObjs.Add(child.gameObject);
        }
        manager.floaterObjects = allObjs;
        manager.numFloaters = allObjs.Count;
        
        gameObj.SetActive(true);
    }

    void Start() {
        UpdateBouyancy();
    }

    public Transform GetScene() {
        return transform;
    }

    public void ClearScene() {
        List<Transform> objs = new();
        foreach (Transform child in transform.GetChild(0)) {
            Debug.Log("Clearing: " + child.name);
            objs.Add(child);
        }
        foreach (Transform child in transform.GetChild(1)) {
            Debug.Log("Clearing: " + child.name);
            objs.Add(child);
        }
        foreach (Transform obj in objs) {
            DestroyImmediate(obj.gameObject);
        }
    }

    public void SpawnScene(SceneData data) {
        Debug.Log("Loading scene dated: " + data.sceneDate);
        foreach (SceneObject obj in data.trash) {
            Debug.Log(obj.id);
            Instantiate(
                trashPrefabs[obj.id], 
                obj.position, 
                Quaternion.Euler(new Vector3(90, 90, Random.Range(0, 360))), // Randomize the yaw rotation
                transform.GetChild(0));
        }
        foreach (SceneObject obj in data.obstacles) {
            Debug.Log(obj.id);
            Instantiate(obstaclePrefabs[obj.id], obj.position, obj.orientation, transform.GetChild(1));
        }
    }
}
