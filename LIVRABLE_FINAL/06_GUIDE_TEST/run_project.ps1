Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Test rapide du projet Deep Learning EMSI" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $projectRoot
Set-Location $projectRoot

Write-Host "[1/4] Verification de l'aide du script principal" -ForegroundColor Yellow
python main.py --help
if ($LASTEXITCODE -ne 0) {
    Write-Host "Echec pendant l'affichage de l'aide." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/4] Verification rapide des datasets locaux" -ForegroundColor Yellow
$tabularPath = Join-Path $projectRoot "data\tabular\online_shoppers_intention.csv"
$textPath = Join-Path $projectRoot "data\text\amazon_food_reviews.csv"

if (Test-Path $tabularPath) {
    Write-Host "OK - Dataset tabulaire detecte : $tabularPath" -ForegroundColor Green
} else {
    Write-Host "ATTENTION - Dataset tabulaire introuvable : $tabularPath" -ForegroundColor DarkYellow
}

if (Test-Path $textPath) {
    Write-Host "OK - Dataset textuel detecte : $textPath" -ForegroundColor Green
} else {
    Write-Host "ATTENTION - Dataset textuel introuvable : $textPath" -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "[3/4] Lancement du quick-run complet" -ForegroundColor Yellow
python main.py --task all --quick-run
if ($LASTEXITCODE -ne 0) {
    Write-Host "Le quick-run a echoue. Verifie les dependances Python." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[4/4] Verification des artefacts" -ForegroundColor Yellow
$artifactDirs = @(
    "artifacts\partie_1_mlp",
    "artifacts\partie_2_cnn",
    "artifacts\partie_3_sequences"
)

foreach ($dir in $artifactDirs) {
    $fullPath = Join-Path $projectRoot $dir
    if (Test-Path $fullPath) {
        Write-Host "OK - dossier present : $fullPath" -ForegroundColor Green
    } else {
        Write-Host "ATTENTION - dossier absent : $fullPath" -ForegroundColor DarkYellow
    }
}

Write-Host ""
Write-Host "Test termine." -ForegroundColor Cyan
