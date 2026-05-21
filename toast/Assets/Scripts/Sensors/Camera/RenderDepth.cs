using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using RosMessageTypes.Std;
using RosMessageTypes.Sensor;
using RosMessageTypes.BuiltinInterfaces;

[ExecuteInEditMode]
public class RenderDepth : MonoBehaviour
{
	[Range(0f, 3f)]
	public float depthLevel = 0.5f;
    
    ROSConnection ros;
    public float frequency = 30;
    float tau;
    float timeElapsed = 0;

	Camera cam;
	
	private Shader _shader;
	private Shader shader
	{
		get { return _shader != null ? _shader : (_shader = Shader.Find("Custom/RenderDepth")); }
	}

	private Material _material;
	private Material material
	{
		get
		{
			if (_material == null)
			{
				_material = new Material(shader);
				_material.hideFlags = HideFlags.HideAndDontSave;
			}
			return _material;
		}
	}

	private void Start ()
	{
		// if (!SystemInfo.supportsImageEffects)
		// {
		// 	print("System doesn't support image effects");
		// 	enabled = false;
		// 	return;
		// }
		if (shader == null || !shader.isSupported)
		{
			enabled = false;
			print("Shader " + shader.name + " is not supported");
			return;
		}
        
        cam = GetComponent<Camera>();
		// turn on depth rendering for the camera so that the shader can access it via _CameraDepthTexture
		cam.depthTextureMode = DepthTextureMode.Depth;
        // ros = ROSConnection.GetOrCreateInstance();
        // ros.RegisterPublisher<ImageMsg>("/depthh", 10);
        // tau = 1 / frequency;
        
	}
	
    void Update()
    {
        timeElapsed += Time.deltaTime;
        if (timeElapsed > tau)
        {
            var rt = RenderTexture.active;
            RenderTexture.active = cam.targetTexture;
            Texture2D texture = new(256, 256);
            texture.ReadPixels(new Rect(0, 0, 256, 256), 0, 0);
            texture.Apply();
            RenderTexture.active = rt;
			cam.Render();
            Graphics.Blit(null, rt);
            // ImageMsg msg = texture.ToImageMsg(new HeaderMsg());
            DestroyImmediate(texture);
			print("meow");
            // ros.Publish("/depthh", msg);
            
            // // Publish camera info
            // CameraInfoMsg infoMsg = CameraInfoGenerator.ConstructCameraInfoMessage(camera, generateHeader("meow"));
            // ros.Publish(infoTopicName, infoMsg);;
            
            timeElapsed = 0;
        }
    }

	private void OnDisable()
	{
		if (_material != null)
			DestroyImmediate(_material);
	}
	
	private void OnRenderImage(RenderTexture src, RenderTexture dest)
	{
		if (shader != null)
		{
			Debug.Log("shader doesnt exist!");
			material.SetFloat("_DepthLevel", depthLevel);
			Graphics.Blit(src, dest, material);
			// Graphics.SetRenderTarget(dest);
		} else {
			Debug.Log("Shader does exist");
			Graphics.Blit(src, dest);
		}
	}
}