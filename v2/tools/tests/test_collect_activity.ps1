#Requires -Version 7.0

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\collect_activity.ps1')

$script:Passed = 0

function Assert-Equal {
    param($Actual, $Expected, [string]$Name)
    if ($Actual -ne $Expected) {
        throw "$Name failed: expected '$Expected', got '$Actual'"
    }
    $script:Passed++
}

Assert-Equal $BaseUrl 'http://127.0.0.1:3000' 'default BaseUrl'
Assert-Equal $PrepSeconds 0 'default PrepSeconds'
Assert-Equal (Get-ActivityPrefix -ActivityLabel 'SITTING') 'sitting' 'SITTING prefix'
Assert-Equal (Get-ActivityPrefix -ActivityLabel 'MOVING') 'moving' 'MOVING prefix'
Assert-Equal (Get-ActivityPrefix -ActivityLabel 'LYING') 'lying' 'LYING prefix'
Assert-Equal (Get-ActivitySessionId -ActivityLabel 'SITTING' -Number 3) 'sitting_003' 'zero padding'
Assert-Equal (Get-PreparationMessage -ActivityLabel 'LYING') '누운 자세를 준비하세요.' 'preparation message'
Assert-Equal `
    (Test-ActivityApiSuccess -Response ([pscustomobject]@{
        TransportSuccess = $true
        HttpSuccess = $true
        Body = [pscustomobject]@{ success = $true }
    })) `
    $true `
    'API success acceptance'
Assert-Equal `
    (Test-ActivityApiSuccess -Response ([pscustomobject]@{
        TransportSuccess = $true
        HttpSuccess = $false
        Body = [pscustomobject]@{ success = $false; error = 'no_recent_esp32_csi' }
    })) `
    $false `
    'HTTP failure rejection'

$script:CountdownSleeps = 0
Invoke-ActivityPreparationCountdown `
    -PrepSeconds 0 `
    -SleepAction { param($Seconds) $script:CountdownSleeps += $Seconds }
Assert-Equal $script:CountdownSleeps 0 'zero-second countdown'

$script:CountdownSleeps = 0
Invoke-ActivityPreparationCountdown `
    -PrepSeconds 3 `
    -SleepAction { param($Seconds) $script:CountdownSleeps += $Seconds }
Assert-Equal $script:CountdownSleeps 3 'countdown waits before start'

$helperSource = Get-Content (Join-Path $PSScriptRoot '..\collect_activity.ps1') -Raw
$countdownCall = $helperSource.IndexOf('Invoke-ActivityPreparationCountdown -PrepSeconds $PreparationSeconds')
$startApiCall = $helperSource.IndexOf('$startResponse = Invoke-ActivityJsonPost')
Assert-Equal ($countdownCall -ge 0 -and $countdownCall -lt $startApiCall) $true 'countdown precedes start API'

$testRoot = Join-Path ([IO.Path]::GetTempPath()) ("ruview-activity-collect-test-" + [guid]::NewGuid())
try {
    foreach ($labelName in @('SITTING', 'MOVING', 'LYING')) {
        [void](New-Item -ItemType Directory -Path (Join-Path $testRoot $labelName) -Force)
    }
    $existing = Join-Path $testRoot 'SITTING\sitting_003.jsonl'
    Set-Content -LiteralPath $existing -Value '{}' -Encoding utf8NoBOM
    Assert-Equal `
        (Test-ActivitySessionExists -ActivityRoot $testRoot -SessionId 'sitting_003') `
        $true `
        'existing session detection'

    $retryOne = Join-Path $testRoot 'SITTING\sitting_003_retry1.jsonl'
    Set-Content -LiteralPath $retryOne -Value '{}' -Encoding utf8NoBOM
    Assert-Equal `
        (Get-NextRetrySessionId -ActivityRoot $testRoot -BaseSessionId 'sitting_003') `
        'sitting_003_retry2' `
        'retry never overwrites'

    $expectedPath = Join-Path $testRoot 'MOVING\moving_008.jsonl'
    Assert-Equal `
        (Get-ActivityRecordingPath -ActivityRoot $testRoot -ActivityLabel 'MOVING' -SessionId 'moving_008') `
        $expectedPath `
        'recording path'
}
finally {
    $resolvedTemp = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).Path.TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $resolvedTarget = (Resolve-Path -LiteralPath $testRoot).Path
    if (-not $resolvedTarget.StartsWith($resolvedTemp + [IO.Path]::DirectorySeparatorChar)) {
        throw "Refusing cleanup outside temp: $resolvedTarget"
    }
    Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
}

Write-Host "$script:Passed PowerShell helper assertions passed."
