param(
    [ValidateSet('launch', 'build', 'verify', 'verify-display', 'check')]
    [string]$Action = 'launch',
    [string]$Distro = 'Ubuntu-22.04',
    [string]$ModelDir = $env:RMUC_MODEL_DIR,
    [switch]$Headless,
    [switch]$Rebuild
)
$ErrorActionPreference = 'Stop'
$linuxRoot = (& wsl.exe -d $Distro --exec wslpath -a -u $PSScriptRoot)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$linuxRoot = $linuxRoot.Trim()
$prefixArgs = @('-d', $Distro, '--exec')
if ($ModelDir) {
    if ($ModelDir.StartsWith('/')) { $linuxModel = $ModelDir }
    else {
        $modelResolved = (Resolve-Path -LiteralPath $ModelDir).Path
        $linuxModel = (& wsl.exe -d $Distro --exec wslpath -a -u $modelResolved)
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        $linuxModel = $linuxModel.Trim()
    }
    $prefixArgs += @('env', "RMUC_MODEL_DIR=$linuxModel")
}
if ($Rebuild -and $Action -ne 'build') {
    & wsl.exe @prefixArgs bash "$linuxRoot/run.sh" build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$launchArgs = @()
if ($Headless -and $Action -eq 'launch') { $launchArgs += @('gui:=false', 'rviz:=false') }
& wsl.exe @prefixArgs bash "$linuxRoot/run.sh" $Action @launchArgs
exit $LASTEXITCODE
