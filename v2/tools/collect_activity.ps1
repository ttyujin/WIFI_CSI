#Requires -Version 7.0

[CmdletBinding()]
param(
    [ValidateSet('SITTING', 'MOVING', 'LYING')]
    [string]$Label,

    [ValidateRange(1, 999)]
    [int]$Start = 1,

    [ValidateRange(1, 999)]
    [int]$End = 8,

    [ValidateRange(0, 300)]
    [int]$PrepSeconds = 0,

    [string]$BaseUrl = 'http://127.0.0.1:3000',

    [string]$PythonPath
)

Set-StrictMode -Version Latest

function Get-ActivityPrefix {
    param([Parameter(Mandatory)][string]$ActivityLabel)

    switch ($ActivityLabel) {
        'SITTING' { return 'sitting' }
        'MOVING' { return 'moving' }
        'LYING' { return 'lying' }
        default { throw "지원하지 않는 label입니다: $ActivityLabel" }
    }
}

function Get-PreparationMessage {
    param([Parameter(Mandatory)][string]$ActivityLabel)

    switch ($ActivityLabel) {
        'SITTING' { return '앉은 자세를 준비하세요.' }
        'MOVING' { return '걸을 준비를 하세요.' }
        'LYING' { return '누운 자세를 준비하세요.' }
        default { throw "지원하지 않는 label입니다: $ActivityLabel" }
    }
}

function Get-ActivitySessionId {
    param(
        [Parameter(Mandatory)][string]$ActivityLabel,
        [Parameter(Mandatory)][ValidateRange(1, 999)][int]$Number
    )

    $prefix = Get-ActivityPrefix -ActivityLabel $ActivityLabel
    return '{0}_{1:D3}' -f $prefix, $Number
}

function Get-ActivityRecordingPath {
    param(
        [Parameter(Mandatory)][string]$ActivityRoot,
        [Parameter(Mandatory)][string]$ActivityLabel,
        [Parameter(Mandatory)][string]$SessionId
    )

    return Join-Path $ActivityRoot $ActivityLabel "$SessionId.jsonl"
}

function Test-ActivitySessionExists {
    param(
        [Parameter(Mandatory)][string]$ActivityRoot,
        [Parameter(Mandatory)][string]$SessionId
    )

    foreach ($candidateLabel in @('SITTING', 'MOVING', 'LYING')) {
        $candidate = Get-ActivityRecordingPath `
            -ActivityRoot $ActivityRoot `
            -ActivityLabel $candidateLabel `
            -SessionId $SessionId
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $true
        }
    }
    return $false
}

function Get-NextRetrySessionId {
    param(
        [Parameter(Mandatory)][string]$ActivityRoot,
        [Parameter(Mandatory)][string]$BaseSessionId
    )

    for ($retryNumber = 1; $retryNumber -le 999; $retryNumber++) {
        $candidate = '{0}_retry{1}' -f $BaseSessionId, $retryNumber
        if (-not (Test-ActivitySessionExists -ActivityRoot $ActivityRoot -SessionId $candidate)) {
            return $candidate
        }
    }
    throw "사용 가능한 retry session id를 찾지 못했습니다: $BaseSessionId"
}

function Resolve-ActivityPython {
    param(
        [string]$RequestedPath,
        [Parameter(Mandatory)][string]$V2Root
    )

    if ($RequestedPath) {
        if (Test-Path -LiteralPath $RequestedPath -PathType Leaf) {
            return [pscustomobject]@{
                Executable = (Resolve-Path -LiteralPath $RequestedPath).Path
                PrefixArguments = @()
            }
        }
        $requestedCommand = Get-Command $RequestedPath -ErrorAction SilentlyContinue
        if ($null -ne $requestedCommand) {
            return [pscustomobject]@{
                Executable = $requestedCommand.Source
                PrefixArguments = @()
            }
        }
        throw "Python 실행 파일을 찾을 수 없습니다: $RequestedPath"
    }

    $pathCandidates = @(
        (Join-Path $V2Root '.venv\Scripts\python.exe'),
        (Join-Path $V2Root '..\.venv\Scripts\python.exe'),
        (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
    )
    foreach ($candidate in $pathCandidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return [pscustomobject]@{
                Executable = (Resolve-Path -LiteralPath $candidate).Path
                PrefixArguments = @()
            }
        }
    }

    foreach ($commandName in @('python', 'python3', 'py')) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            return [pscustomobject]@{
                Executable = $command.Source
                PrefixArguments = if ($commandName -eq 'py') { @('-3') } else { @() }
            }
        }
    }
    throw 'Python 3을 찾을 수 없습니다. -PythonPath로 python.exe 경로를 지정하세요.'
}

function Invoke-ActivityJsonPost {
    param(
        [Parameter(Mandatory)][string]$Uri,
        [Parameter(Mandatory)][hashtable]$Body
    )

    try {
        $response = Invoke-WebRequest `
            -Uri $Uri `
            -Method Post `
            -ContentType 'application/json' `
            -Body ($Body | ConvertTo-Json -Compress) `
            -SkipHttpErrorCheck `
            -TimeoutSec 15 `
            -ErrorAction Stop
        $rawBody = [string]$response.Content
        $jsonBody = $null
        if (-not [string]::IsNullOrWhiteSpace($rawBody)) {
            try {
                $jsonBody = $rawBody | ConvertFrom-Json -ErrorAction Stop
            }
            catch {
                $jsonBody = $null
            }
        }
        $statusCode = [int]$response.StatusCode
        return [pscustomobject]@{
            TransportSuccess = $true
            HttpSuccess = $statusCode -ge 200 -and $statusCode -lt 300
            StatusCode = $statusCode
            Body = $jsonBody
            RawBody = $rawBody
            ErrorMessage = $null
        }
    }
    catch {
        return [pscustomobject]@{
            TransportSuccess = $false
            HttpSuccess = $false
            StatusCode = $null
            Body = $null
            RawBody = $null
            ErrorMessage = $_.Exception.Message
        }
    }
}

function Write-ActivityApiFailure {
    param(
        [Parameter(Mandatory)][string]$Operation,
        [Parameter(Mandatory)]$Response
    )

    Write-Host "$Operation 실패" -ForegroundColor Red
    if ($null -ne $Response.StatusCode) {
        Write-Host "HTTP status : $($Response.StatusCode)" -ForegroundColor Red
    }
    if ($null -ne $Response.Body) {
        foreach ($field in @('success', 'error', 'reason', 'last_seen_ms')) {
            $property = $Response.Body.PSObject.Properties[$field]
            if ($null -ne $property) {
                Write-Host ('{0,-12}: {1}' -f $field, $property.Value) -ForegroundColor Red
            }
        }
    }
    elseif (-not [string]::IsNullOrWhiteSpace([string]$Response.RawBody)) {
        Write-Host "response    : $($Response.RawBody)" -ForegroundColor Red
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$Response.ErrorMessage)) {
        Write-Host "transport   : $($Response.ErrorMessage)" -ForegroundColor Red
    }
}

function Test-ActivityApiSuccess {
    param([Parameter(Mandatory)]$Response)

    if (-not $Response.TransportSuccess -or -not $Response.HttpSuccess -or $null -eq $Response.Body) {
        return $false
    }
    $successProperty = $Response.Body.PSObject.Properties['success']
    return $null -ne $successProperty -and $successProperty.Value -eq $true
}

function Invoke-ActivityQc {
    param(
        [Parameter(Mandatory)]$Python,
        [Parameter(Mandatory)][string]$QcScript,
        [Parameter(Mandatory)][string]$RecordingPath
    )

    $arguments = @($Python.PrefixArguments) + @($QcScript, $RecordingPath)
    $output = @(& $Python.Executable @arguments 2>&1)
    $exitCode = $LASTEXITCODE
    foreach ($line in $output) {
        Write-Host $line
    }
    $text = $output -join [Environment]::NewLine
    if ($text -match '(?m)^Result\s*:\s*(PASS|WARN|FAIL)\s*$') {
        return $Matches[1]
    }
    throw "QC 결과를 판독하지 못했습니다(exit=$exitCode)."
}

function Read-ActivityYesNo {
    param([Parameter(Mandatory)][string]$Prompt)

    while ($true) {
        $answer = (Read-Host $Prompt).Trim().ToUpperInvariant()
        if ($answer -eq 'Y') { return $true }
        if ($answer -eq 'N') { return $false }
        Write-Host 'Y 또는 N을 입력하세요.' -ForegroundColor Yellow
    }
}

function Invoke-ActivityPreparationCountdown {
    param(
        [Parameter(Mandatory)][ValidateRange(0, 300)][int]$PrepSeconds,
        [scriptblock]$SleepAction = { param([int]$Seconds) Start-Sleep -Seconds $Seconds }
    )

    if ($PrepSeconds -eq 0) {
        return
    }
    Write-Host "${PrepSeconds}초 후 recording을 시작합니다." -ForegroundColor Yellow
    for ($remaining = $PrepSeconds; $remaining -ge 1; $remaining--) {
        Write-Host $remaining -ForegroundColor Yellow
        & $SleepAction 1
    }
}

function Invoke-ActivitySession {
    param(
        [Parameter(Mandatory)][string]$SessionId,
        [Parameter(Mandatory)][string]$ActivityLabel,
        [Parameter(Mandatory)][string]$PreparationMessage,
        [Parameter(Mandatory)][string]$ApiBaseUrl,
        [Parameter(Mandatory)][string]$ActivityRoot,
        [Parameter(Mandatory)]$Python,
        [Parameter(Mandatory)][string]$QcScript,
        [Parameter(Mandatory)][bool]$IsFirstSession,
        [Parameter(Mandatory)][ValidateRange(0, 300)][int]$PreparationSeconds
    )

    $recordingPath = Get-ActivityRecordingPath `
        -ActivityRoot $ActivityRoot `
        -ActivityLabel $ActivityLabel `
        -SessionId $SessionId
    if (Test-ActivitySessionExists -ActivityRoot $ActivityRoot -SessionId $SessionId) {
        Write-Host "이미 존재하는 session입니다. 덮어쓰지 않습니다: $SessionId" -ForegroundColor Red
        return [pscustomobject]@{ Outcome = 'FATAL'; SessionId = $SessionId }
    }

    Write-Host ''
    Write-Host "============================================================"
    Write-Host "현재 session : $SessionId" -ForegroundColor Cyan
    Write-Host $PreparationMessage -ForegroundColor Yellow
    if ($IsFirstSession) {
        [void](Read-Host '준비가 완료되면 Enter를 누르세요')
    }
    else {
        [void](Read-Host '다음 session 준비 후 Enter를 누르세요')
    }
    Invoke-ActivityPreparationCountdown -PrepSeconds $PreparationSeconds

    $startResponse = Invoke-ActivityJsonPost `
        -Uri "$ApiBaseUrl/api/v1/activity/recording/start" `
        -Body @{ session_id = $SessionId; label = $ActivityLabel }
    $startAccepted = Test-ActivityApiSuccess -Response $startResponse
    if (-not $startAccepted) {
        Write-ActivityApiFailure -Operation 'recording start' -Response $startResponse
        return [pscustomobject]@{ Outcome = 'FATAL'; SessionId = $SessionId }
    }

    Write-Host 'Recording 시작 성공. 30초 동안 기록합니다.' -ForegroundColor Green
    $stopAttempted = $false
    try {
        Start-Sleep -Seconds 30
        Write-Host '30초 기록 완료. Recording stop을 요청합니다.' -ForegroundColor Cyan
        $stopResponse = Invoke-ActivityJsonPost `
            -Uri "$ApiBaseUrl/api/v1/activity/recording/stop" `
            -Body @{}
        $stopAttempted = $true
    }
    finally {
        if (-not $stopAttempted) {
            Write-Host '기록이 중단되어 안전한 stop을 한 번 요청합니다.' -ForegroundColor Yellow
            [void](Invoke-ActivityJsonPost `
                -Uri "$ApiBaseUrl/api/v1/activity/recording/stop" `
                -Body @{})
        }
    }

    $stopAccepted = Test-ActivityApiSuccess -Response $stopResponse
    if (-not $stopAccepted) {
        Write-ActivityApiFailure -Operation 'recording stop' -Response $stopResponse
        return [pscustomobject]@{ Outcome = 'FATAL'; SessionId = $SessionId }
    }
    Write-Host 'Recording stop 성공.' -ForegroundColor Green

    if (-not (Test-Path -LiteralPath $recordingPath -PathType Leaf)) {
        Write-Host "생성된 JSONL을 찾지 못했습니다: $recordingPath" -ForegroundColor Red
        return [pscustomobject]@{ Outcome = 'FATAL'; SessionId = $SessionId }
    }

    Write-Host ''
    Write-Host "QC 실행: $recordingPath" -ForegroundColor Cyan
    try {
        $qcStatus = Invoke-ActivityQc `
            -Python $Python `
            -QcScript $QcScript `
            -RecordingPath $recordingPath
    }
    catch {
        Write-Host "QC 실행 실패: $($_.Exception.Message)" -ForegroundColor Red
        return [pscustomobject]@{ Outcome = 'FATAL'; SessionId = $SessionId }
    }
    return [pscustomobject]@{ Outcome = $qcStatus; SessionId = $SessionId }
}

function Invoke-ActivityCollection {
    param(
        [string]$ActivityLabel,
        [int]$StartNumber,
        [int]$EndNumber,
        [string]$ApiBaseUrl,
        [string]$RequestedPythonPath,
        [ValidateRange(0, 300)][int]$PreparationSeconds
    )

    if ([string]::IsNullOrWhiteSpace($ActivityLabel)) {
        Write-Host 'Label이 필요합니다: SITTING, MOVING, LYING' -ForegroundColor Red
        return 2
    }
    if ($ActivityLabel -notin @('SITTING', 'MOVING', 'LYING')) {
        Write-Host "허용되지 않은 label입니다: $ActivityLabel" -ForegroundColor Red
        return 2
    }
    if ($StartNumber -gt $EndNumber) {
        Write-Host "Start는 End보다 클 수 없습니다: $StartNumber > $EndNumber" -ForegroundColor Red
        return 2
    }

    try {
        $baseUri = [uri]$ApiBaseUrl
        if (-not $baseUri.IsAbsoluteUri -or $baseUri.Scheme -notin @('http', 'https')) {
            throw 'HTTP 또는 HTTPS absolute URL이 아닙니다.'
        }
    }
    catch {
        Write-Host "잘못된 BaseUrl입니다: $ApiBaseUrl" -ForegroundColor Red
        return 2
    }
    $normalizedBaseUrl = $ApiBaseUrl.TrimEnd('/')
    $v2Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
    $activityRoot = Join-Path $v2Root 'data\recordings\activity'
    $qcScript = Join-Path $v2Root 'tools\activity_qc.py'

    if (-not (Test-Path -LiteralPath $qcScript -PathType Leaf)) {
        Write-Host "QC 도구를 찾을 수 없습니다: $qcScript" -ForegroundColor Red
        return 1
    }
    try {
        $python = Resolve-ActivityPython `
            -RequestedPath $RequestedPythonPath `
            -V2Root $v2Root
    }
    catch {
        Write-Host $_.Exception.Message -ForegroundColor Red
        return 1
    }

    $existingSessions = @()
    foreach ($number in $StartNumber..$EndNumber) {
        $plannedId = Get-ActivitySessionId -ActivityLabel $ActivityLabel -Number $number
        if (Test-ActivitySessionExists -ActivityRoot $activityRoot -SessionId $plannedId) {
            $existingSessions += $plannedId
        }
    }
    if ($existingSessions.Count -gt 0) {
        Write-Host '이미 존재하는 formal session이 있어 수집을 시작하지 않습니다.' -ForegroundColor Red
        foreach ($existingSession in $existingSessions) {
            Write-Host "  - $existingSession" -ForegroundColor Red
        }
        return 1
    }

    Write-Host "Activity collection: $ActivityLabel $StartNumber..$EndNumber"
    Write-Host "API: $normalizedBaseUrl"
    Write-Host '각 recording은 사용자 Enter 확인 후 시작하며 30초간 진행됩니다.'
    if ($PreparationSeconds -gt 0) {
        Write-Host "Enter 확인 후 ${PreparationSeconds}초 준비시간을 적용합니다."
    }
    $preparationMessage = Get-PreparationMessage -ActivityLabel $ActivityLabel
    $isFirstSession = $true

    foreach ($number in $StartNumber..$EndNumber) {
        $baseSessionId = Get-ActivitySessionId -ActivityLabel $ActivityLabel -Number $number
        $currentSessionId = $baseSessionId
        while ($true) {
            $result = Invoke-ActivitySession `
                -SessionId $currentSessionId `
                -ActivityLabel $ActivityLabel `
                -PreparationMessage $preparationMessage `
                -ApiBaseUrl $normalizedBaseUrl `
                -ActivityRoot $activityRoot `
                -Python $python `
                -QcScript $qcScript `
                -IsFirstSession $isFirstSession `
                -PreparationSeconds $PreparationSeconds
            $isFirstSession = $false

            if ($result.Outcome -eq 'FATAL') {
                Write-Host "수집을 중단합니다. session: $($result.SessionId)" -ForegroundColor Red
                return 1
            }
            if ($result.Outcome -ne 'FAIL') {
                break
            }

            $retry = Read-ActivityYesNo -Prompt '이 session을 다시 측정할까요? Y/N'
            if (-not $retry) {
                break
            }
            $currentSessionId = Get-NextRetrySessionId `
                -ActivityRoot $activityRoot `
                -BaseSessionId $baseSessionId
            Write-Host "기존 파일을 보존하고 새 retry session을 사용합니다: $currentSessionId" -ForegroundColor Yellow
        }
    }

    Write-Host ''
    Write-Host '요청한 activity collection이 완료되었습니다.' -ForegroundColor Green
    return 0
}

if ($MyInvocation.InvocationName -ne '.') {
    $exitCode = Invoke-ActivityCollection `
        -ActivityLabel $Label `
        -StartNumber $Start `
        -EndNumber $End `
        -ApiBaseUrl $BaseUrl `
        -RequestedPythonPath $PythonPath `
        -PreparationSeconds $PrepSeconds
    exit $exitCode
}
