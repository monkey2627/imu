$ErrorActionPreference = 'Stop'
$remote = 'duanyuhan@10.132.25.165'
$remoteRun = '/home/duanyuhan/dyh/motion/lines/imudino_sidebridge_v3_20261118'
$localRoot = 'E:\dyh\MotionRecover\realuse\MotionRecoverWorkspace'
$modelId = 'waist_imudino_stream_sidebridge_v3_rehabv5_20261118'
$localModel = Join-Path $localRoot "local_models\checkpoints\$modelId"
$log = Join-Path $localRoot 'sync\sidebridge_monitor.log'

function Log($msg) {
    Add-Content -LiteralPath $log -Value ("[{0}] {1}" -f (Get-Date -Format s), $msg)
}

New-Item -ItemType Directory -Force -Path (Split-Path $log), $localModel | Out-Null
Log "monitor started"
while ($true) {
    try {
        & ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 $remote "test -f $remoteRun/logs/COMPLETE"
        if ($LASTEXITCODE -eq 0) { break }
        & ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 $remote "test -f $remoteRun/logs/FAILED"
        if ($LASTEXITCODE -eq 0) { Log 'remote training FAILED'; exit 2 }
    } catch { Log ("poll error: " + $_.Exception.Message) }
    Start-Sleep -Seconds 60
}

Log 'training complete; pulling artifacts'
& scp -o ConnectTimeout=10 "$remote`:$remoteRun/runs/stream_sidebridge_v3_amass/best.pt" (Join-Path $localModel 'best.pt')
& scp -o ConnectTimeout=10 "$remote`:$remoteRun/runs/stream_sidebridge_v3_amass/metrics.jsonl" (Join-Path $localModel 'metrics.jsonl')
& scp -o ConnectTimeout=10 "$remote`:$remoteRun/logs/COMPLETE" (Join-Path $localModel 'COMPLETE')
$backbone = Join-Path $localRoot 'local_models\checkpoints\waist_imudino_stream_deadbug_rehabv5_std_20261118\imudino.pt'
Copy-Item -LiteralPath $backbone -Destination (Join-Path $localModel 'imudino.pt') -Force

$bestHash = (Get-FileHash (Join-Path $localModel 'best.pt') -Algorithm SHA256).Hash.ToLower()
$backboneHash = (Get-FileHash (Join-Path $localModel 'imudino.pt') -Algorithm SHA256).Hash.ToLower()
$manifest = [ordered]@{
    model_id = $modelId
    export_date = (Get-Date -Format yyyy-MM-dd)
    source = "$remote`:$remoteRun/runs/stream_sidebridge_v3_amass"
    architecture = 'frozen IMUDINO backbone + stateful StreamHead'
    contract = 'stream.common v1'
    sensor_order = @('lw','rw','ls','rs','waist')
    fps = 60
    online_lag_frames = 1
    dataset = 'side_bridge_v3 final_index 500 + validated topup supplement 47 + 18 AMASS datasets'
    files = [ordered]@{
        'best.pt' = [ordered]@{ source='best.pt'; bytes=(Get-Item (Join-Path $localModel 'best.pt')).Length; sha256=$bestHash }
        'imudino.pt' = [ordered]@{ source='shared frozen backbone'; bytes=(Get-Item (Join-Path $localModel 'imudino.pt')).Length; sha256=$backboneHash }
    }
    code = '../../model_code/waist_imudino_fiveimu'
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $localModel 'manifest.json') -Encoding UTF8

$backend = Join-Path $localRoot 'runtime\python\production\model_backends.py'
$bt = Get-Content -LiteralPath $backend -Raw
if ($bt -notmatch [regex]::Escape($modelId)) {
    $bt = $bt -replace '("waist_imudino_stream_bridge_rehabv5_std_20261118": \("imudino_stream", "waist_imudino_stream_bridge_rehabv5_std_20261118/best\.pt"\),)', ('$1' + "`r`n    `"$modelId`": (`"imudino_stream`", `"$modelId/best.pt`"),")
    $bt = $bt -replace '("hipthrust": "waist_imudino_stream_bridge_rehabv5_std_20261118",)', ('$1' + "`r`n    `"sidebridge`": `"$modelId`",")
    $bt = $bt -replace '("waist_imudino_stream_bridge_rehabv5_std_20261118": "[^"]+",)', ('$1' + "`r`n    `"$modelId`": `"IMUDINO Stream / SideBridge v3 RehabV5 2026-11-18`",")
    Set-Content -LiteralPath $backend -Value $bt -Encoding UTF8
}

foreach ($file in @('frame_chain.py','bridge_contact.py')) {
    $path = Join-Path $localRoot "runtime\python\production\$file"
    $t = Get-Content -LiteralPath $path -Raw
    if ($t -notmatch [regex]::Escape($modelId)) {
        $t = $t -replace '("waist_imudino_stream_bridge_rehabv5_std_20261118")', ('$1, "' + $modelId + '"')
        Set-Content -LiteralPath $path -Value $t -Encoding UTF8
    }
}

$unity = Join-Path $localRoot 'runtime\unity\SpineRecovery_V2\Assets\Scripts\MobilePoseWebSocketSource.cs'
$ut = Get-Content -LiteralPath $unity -Raw
if ($ut -notmatch [regex]::Escape($modelId)) {
    $ut = $ut -replace '("waist_imudino_stream_bridge_rehabv5_std_20261118",)', ('$1' + "`r`n        `"$modelId`",")
    $ut = $ut -replace '("IMUDINO Stream / Bridge RehabV5 standardized 2026-11-18",)', ('$1' + "`r`n        `"IMUDINO Stream / SideBridge v3 RehabV5 2026-11-18`",")
    Set-Content -LiteralPath $unity -Value $ut -Encoding UTF8
}
Log "registered $modelId"
