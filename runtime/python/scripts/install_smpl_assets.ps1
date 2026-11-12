param(
    [string]$SmplPythonRoot = "D:\Download\SMPL_python_v.1.0.0",
    [string]$SmplUnityRoot = "D:\Download\SMPL_unity_v.1.0.0\SMPL_unity_v.1.0.0"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonSource = Join-Path $SmplPythonRoot "smpl\models\basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
$unitySource = Join-Path $SmplUnityRoot "smpl_mecanim\assets\SMPL\Models\SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx"
$pythonDestination = Join-Path $repoRoot "mobileposer\smpl\basicmodel_m.pkl"
$unityDestination = Join-Path $repoRoot "IMUTrack-for-Spine\Assets\Resources\SMPL\SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx"

foreach ($source in @($pythonSource, $unitySource)) {
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required SMPL asset not found: $source"
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $pythonDestination) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $unityDestination) | Out-Null
Copy-Item -LiteralPath $pythonSource -Destination $pythonDestination -Force
Copy-Item -LiteralPath $unitySource -Destination $unityDestination -Force

Write-Host "Installed SMPL Python model: $pythonDestination"
Write-Host "Installed SMPL Unity model:  $unityDestination"
Write-Host "These licensed model files are intentionally ignored by Git."
