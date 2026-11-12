[CmdletBinding()]
param(
    [switch]$ValidateOnly,
    [switch]$Archive
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$cmakeExe = if ($env:SLIMEVR_CMAKE_EXE) {
    $env:SLIMEVR_CMAKE_EXE
} else {
    "D:\Software\cmake\bin\cmake.exe"
}
$javaHome = if ($env:SLIMEVR_JAVA_HOME) {
    $env:SLIMEVR_JAVA_HOME
} else {
    "D:\Software\jdk-17"
}
$pnpmCmd = if ($env:SLIMEVR_PNPM_CMD) {
    $env:SLIMEVR_PNPM_CMD
} else {
    "D:\Nodejs\pnpm.cmd"
}
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"

function Assert-Path([string]$path, [string]$description) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "$description not found: $path"
    }
}

function Invoke-Native([scriptblock]$command, [string]$description) {
    & $command
    if ($LASTEXITCODE -ne 0) {
        throw "$description failed with exit code $LASTEXITCODE"
    }
}

Assert-Path $cmakeExe "CMake"
Assert-Path (Join-Path $javaHome "bin\java.exe") "JDK 17"
Assert-Path $pnpmCmd "pnpm"
Assert-Path $vswhere "vswhere"
Assert-Path (Join-Path $repoRoot "solarxr-protocol\lib\flatbuffers\include\flatbuffers\flatbuffers.h") "FlatBuffers headers"
Assert-Path (Join-Path $repoRoot "bindings-provider\openvr\lib\win64\openvr_api.lib") "OpenVR library"

$vsInstall = (& $vswhere `
    -latest `
    -products * `
    -version "[17.0,18.0)" `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath).Trim()

if (-not $vsInstall) {
    throw "A Visual Studio 2022 installation with the x64/x86 C++ tools is required."
}

$vsDevCmd = Join-Path $vsInstall "Common7\Tools\VsDevCmd.bat"
Assert-Path $vsDevCmd "Visual Studio 2022 developer environment"

Write-Host "CMake: $cmakeExe"
Write-Host "Java:  $javaHome"
Write-Host "pnpm:  $pnpmCmd"
Write-Host "VS:    $vsInstall"

if ($ValidateOnly) {
    Write-Host "Build prerequisites are available."
    exit 0
}

Push-Location $repoRoot
try {
    $previousJavaHome = $env:JAVA_HOME
    $env:JAVA_HOME = $javaHome

    Invoke-Native { .\gradlew.bat :server:desktop:shadowJar } "Server build"

    # The parent environment can contain both PATH and Path. MSBuild treats them
    # as duplicate keys, so create a clean child environment before VsDevCmd.
    $basePath = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\Wbem"
    $configure = 'set PATH=& set Path=& set "Path={0}"& call "{1}" -arch=x64 -host_arch=x64 && "{2}" --fresh -S bindings-provider -B bindings-provider/build/win-x64 -G "Visual Studio 17 2022" -A x64' -f $basePath, $vsDevCmd, $cmakeExe
    Invoke-Native { & $env:ComSpec /D /S /C $configure } "Bindings configuration"

    $compile = 'set PATH=& set Path=& set "Path={0}"& call "{1}" -arch=x64 -host_arch=x64 && "{2}" --build bindings-provider/build/win-x64 --config Release --parallel' -f $basePath, $vsDevCmd, $cmakeExe
    Invoke-Native { & $env:ComSpec /D /S /C $compile } "Bindings build"

    Invoke-Native { & $pnpmCmd run update-solarxr } "SolarXR TypeScript build"

    Push-Location (Join-Path $repoRoot "gui")
    try {
        Invoke-Native { & $pnpmCmd build } "GUI build"
        Invoke-Native { & .\node_modules\.bin\electron-builder.CMD --dir } "Windows app packaging"
    } finally {
        Pop-Location
    }

    $appDir = Join-Path $repoRoot "gui\dist\artifacts\win\win-unpacked"
    foreach ($name in @("SlimeVR.exe", "slimevr.jar", "SlimeVR-Bindings-Provider.exe", "openvr_api.dll")) {
        Assert-Path (Join-Path $appDir $name) $name
    }

    if ($Archive) {
        $archivePath = Join-Path $repoRoot "gui\dist\artifacts\win\SlimeVR-local-win-x64.zip"
        if (Test-Path -LiteralPath $archivePath) {
            Remove-Item -LiteralPath $archivePath -Force
        }
        Compress-Archive -Path (Join-Path $appDir "*") -DestinationPath $archivePath -CompressionLevel Optimal
        Write-Host "Archive: $archivePath"
    }

    Write-Host "Application: $(Join-Path $appDir 'SlimeVR.exe')"
} finally {
    $env:JAVA_HOME = $previousJavaHome
    Pop-Location
}
