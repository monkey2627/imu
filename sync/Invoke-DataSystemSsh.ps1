param(
    [Parameter(Mandatory=$true)][string]$ScriptPath,
    [int]$TimeoutSeconds = 20
)
$ErrorActionPreference = 'Stop'
$start = New-Object System.Diagnostics.ProcessStartInfo
$start.FileName = 'C:\Windows\System32\OpenSSH\ssh.exe'
$start.Arguments = '-o BatchMode=yes -o ConnectTimeout=7 -o ConnectionAttempts=1 -o ServerAliveInterval=3 -o ServerAliveCountMax=2 10.132.25.165 "timeout 15s python3 -"'
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
$start.RedirectStandardInput = $true
$start.RedirectStandardOutput = $true
$start.RedirectStandardError = $true
$start.StandardOutputEncoding = [System.Text.Encoding]::UTF8
$start.StandardErrorEncoding = [System.Text.Encoding]::UTF8
$process = New-Object System.Diagnostics.Process
$process.StartInfo = $start
try {
    [void]$process.Start()
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    $process.StandardInput.Write((Get-Content -LiteralPath $ScriptPath -Raw -Encoding UTF8))
    $process.StandardInput.Close()
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        $process.Kill()
        Write-Output 'SSH_HARD_TIMEOUT: local SSH stopped after deadline; remote Python also has a 15s timeout.'
        exit 124
    }
    Write-Output $stdout.Result
    if ($stderr.Result) { Write-Output $stderr.Result }
    exit $process.ExitCode
} finally {
    if ($process.Id -and -not $process.HasExited) { $process.Kill() }
    $process.Dispose()
}
