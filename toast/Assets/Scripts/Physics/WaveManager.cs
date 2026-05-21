using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WaveManager : MonoBehaviour
{
    public static WaveManager instance;
    public Shader waveShader;
    Material waveMaterial;

    public float waveSpeed = 0.63f;
    public float waveSteepness = 0.7f;
    public float waveLength = 10f;
    // public float timeOffset;

    private void Awake() {
        if (instance == null) {
            instance = this;
        } else {
            Destroy(this);
        }
    }

    void Start() {
        waveMaterial = GetComponent<Renderer>().material;
        waveMaterial.shader = waveShader;
        OnValidate();
    }

    void OnValidate() {
        if (waveMaterial != null) {
            waveMaterial.SetFloat("_Speed", waveSpeed);
            waveMaterial.SetFloat("_Steepness", waveSteepness);
            waveMaterial.SetFloat("_Wavelength", waveLength);
        }
    }

    public float GetWaveHeight(Vector3 pos) {
        double k = 2 * Math.PI / waveLength;
        double c = Math.Sqrt(9.8 / k);
        double a = waveSteepness / k;
        double f = k * (pos.x - a * 0.5 - c * Time.timeSinceLevelLoad);

        return (float) (a * Math.Sin(f));
    }
}
