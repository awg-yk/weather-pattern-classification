# 新しいターミナルを開くたびに必要な準備をまとめて行う。
#
#   .\setup-session.ps1
#
# やること:
#   1. venvを有効にする(有効でなければ)
#   2. $processed_dir を天気図の置き場所に設定する
#   3. 現在のブランチとラベル件数を表示する
#
# 実行が拒否される場合は、いちどだけ次を実行してください:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

if (-not $env:VIRTUAL_ENV) {
    $activate = Join-Path $repo "venv\Scripts\Activate.ps1"
    if (Test-Path $activate) {
        & $activate
        Write-Host "venvを有効にしました" -ForegroundColor Green
    } else {
        Write-Host "venvが見つかりません: $activate" -ForegroundColor Yellow
    }
}

# 天気図の画像はデータ用リポジトリ側にある
$candidates = @(
    (Join-Path (Split-Path -Parent $repo) "weather-pattern-classification-data\processed"),
    (Join-Path $repo "data\processed")
)
$global:processed_dir = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($global:processed_dir) {
    $count = (Get-ChildItem $global:processed_dir -Filter *.png -ErrorAction SilentlyContinue).Count
    Write-Host "`$processed_dir = $global:processed_dir ($count 枚)" -ForegroundColor Green
} else {
    Write-Host "天気図の置き場所が見つかりません。`$processed_dir を手で設定してください" -ForegroundColor Yellow
}

Write-Host ("ブランチ: " + (git rev-parse --abbrev-ref HEAD)) -ForegroundColor Cyan
foreach ($csv in @("data\labels.csv", "data\labels_v2.csv")) {
    if (Test-Path $csv) {
        $rows = (Import-Csv $csv).Count
        Write-Host "  $csv : $rows 件"
    }
}
