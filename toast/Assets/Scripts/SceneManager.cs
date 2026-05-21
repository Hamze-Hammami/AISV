using System;
using UnityEditor;
using UnityEngine.UIElements;
using UnityEngine;
using System.IO;
using System.Collections.Generic;
using UnityEditor.SearchService;
using System.Linq;
using Unity.VisualScripting;

[Serializable]
public class SceneData {
    public string sceneDate;
    public List<SceneObject> trash;
    public List<SceneObject> obstacles;

    public SceneData (string date) {
        sceneDate = date;
    }
}

[Serializable]
public class SceneObject {
    public int id;
    public Vector3 position;
    public Quaternion orientation;

    public SceneObject(int id_p, Transform t) {
        id = id_p;
        position = t.position;
        orientation = t.rotation;
    }
}

public class SceneManager : EditorWindow
{
    string scenesPath = Application.dataPath + "/Scenes/";

    [MenuItem("Window/Scene Manager")]
    public static void ShowWindow()
    {
        SceneManager wnd = GetWindow<SceneManager>();
        wnd.titleContent = new GUIContent("Scene Manager");
    }

    TextField fileNameField;
    string fileName;
    SpawnerScript spawnerScript;

    public void OnEnable()
    {
        spawnerScript = GameObject.Find("Spawner").GetComponent<SpawnerScript>();
    }

    public void CreateGUI()
    {
        // Instructions label
        VisualElement root = rootVisualElement;
        VisualElement label = new Label("Use this GUI to manage the scene's trash and obstacles.");
        root.Add(label);

        fileNameField = new()
        {
            label = "File name of JSON file (Assets/Scenes/)",
        };
        fileNameField.RegisterCallback<ChangeEvent<string>>(FileNameCB);
        root.Add(fileNameField);
        fileNameField.SetValueWithoutNotify("test.json");

        // Button to save scene in a file
        Button saveFileBtn = new()
        {
            name = "saveFileBtn",
            text = "Save scene in file",
        };
        saveFileBtn.RegisterCallback<ClickEvent>(SaveSceneCB);
        root.Add(saveFileBtn);

        // Button to load scene from file
        Button loadSceneBtn = new()
        {
            name = "loadSceneBtn",
            text = "Load new scene in file",
        };
        loadSceneBtn.RegisterCallback<ClickEvent>(LoadSceneCB);
        root.Add(loadSceneBtn);
        
        // Button to clear scene
        Button clearSceneBtn = new()
        {
            name = "clearSceneBtn",
            text = "Clear the scene",
        };
        clearSceneBtn.RegisterCallback<ClickEvent>(ClearSceneCB);
        root.Add(clearSceneBtn);
    }

    // Input handlers for the inputs
    private void FileNameCB(ChangeEvent<string> e)
    {
        fileName = e.newValue;
    }

    private void ClearSceneCB(ClickEvent e) {
        spawnerScript.ClearScene();
    }

    private void LoadSceneCB(ClickEvent e)
    {
        spawnerScript.ClearScene();
        SceneData data = JsonUtility.FromJson<SceneData>(File.ReadAllText(scenesPath + fileName));
        spawnerScript.SpawnScene(data);
    }

    private void SaveSceneCB(ClickEvent e)
    {
        SceneData newScene = new(DateTime.Now.ToString("H:mm dd-MM-yy"));
        Transform objects = spawnerScript.GetScene();
        List<SceneObject> trashObjs = new();
        foreach (Transform trash in objects.GetChild(0).transform) {
            // Debug.Log(trash.gameObject);
            SceneObject newObj = new(trash.GetComponent<PrefabMeta>().uniqueID, trash);
            Debug.Log(newObj);
            trashObjs.Add(newObj);
        }
        newScene.trash = trashObjs;
        List<SceneObject> obstacleObjs = new();
        foreach (Transform obstacle in objects.GetChild(1).transform) {
            SceneObject newObj = new(obstacle.GetComponent<PrefabMeta>().uniqueID, obstacle);
            Debug.Log(newObj);
            obstacleObjs.Add(newObj);
        }
        newScene.obstacles = obstacleObjs;
        File.WriteAllText(scenesPath + fileName, JsonUtility.ToJson(newScene, true));
        Debug.Log("done writing!");
    }
}
