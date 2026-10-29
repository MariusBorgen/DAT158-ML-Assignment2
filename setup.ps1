# Kjor fra PowerShell: .\setup.ps1
# Skriptet bruker prosjektets Python direkte, sa aktivering er ikke nodvendig.
$ErrorActionPreference = 'Stop'
$venvDirectory = Join-Path $PSScriptRoot '.venv'
$venvPython = Join-Path $venvDirectory 'Scripts\python.exe'
$requirements = Join-Path $PSScriptRoot 'requirements.txt'
$modelPath = Join-Path $PSScriptRoot 'models\face_recognition_custom.joblib'

try {
    # 1. Opprett miljoet hvis det ikke finnes fra for.
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        if (Test-Path -LiteralPath $venvDirectory) {
            throw 'Mappen .venv finnes, men mangler Scripts\python.exe. Sjekk miljoet for du prover igjen.'
        }

        Write-Host '1/3 Oppretter Python-miljoet .venv ...'
        if (Get-Command py -CommandType Application -ErrorAction SilentlyContinue) {
            & py -3 -m venv $venvDirectory
        } elseif (Get-Command python -CommandType Application -ErrorAction SilentlyContinue) {
            & python -m venv $venvDirectory
        } else {
            throw 'Python ble ikke funnet. Installer Python fra python.org (prosjektet er testet med 3.13), og apne terminalen pa nytt.'
        }
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
            throw 'Kunne ikke opprette .venv. Sjekk at Python er installert. Prosjektet er testet med Python 3.13.'
        }
    } else {
        Write-Host '1/3 Bruker eksisterende .venv.'
    }

    # 2. Installer i riktig miljo, uansett hvilken Python terminalen ellers bruker.
    Write-Host '2/3 Installerer avhengigheter. Dette kan ta noen minutter ...'
    & $venvPython -m pip install --disable-pip-version-check -r $requirements
    if ($LASTEXITCODE -ne 0) {
        throw 'Installeringen feilet. Se pip-feilen over. Sjekk internett og Python-versjon; prosjektet er testet med Python 3.13.'
    }

    # 3. Sjekk importene og at OpenCV finner ansiktsdetektoren.
    Write-Host '3/3 Kontrollerer installasjonen ...'
    $check = @'
import cv2, gradio, joblib, matplotlib, numpy, scipy, sklearn, threadpoolctl
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
assert not detector.empty(), 'OpenCV fant ikke ansiktsdetektoren.'
print('OpenCV:', cv2.__version__)
print('scikit-learn:', sklearn.__version__)
'@
    & $venvPython -c $check
    if ($LASTEXITCODE -ne 0) {
        throw 'Kontrollen feilet. Se feilmeldingen over; installasjonen er ikke klar.'
    }

    Write-Host ''
    Write-Host 'Python-miljoet er klart.' -ForegroundColor Green
    if (Test-Path -LiteralPath $modelPath -PathType Leaf) {
        Write-Host 'Modellfilen finnes. Start gjenkjenning fra prosjektmappen med:'
        Write-Host '  .\.venv\Scripts\python.exe src/recognize_face.py'
    } else {
        Write-Host 'Modellfilen mangler: models/face_recognition_custom.joblib' -ForegroundColor Yellow
        Write-Host 'Den folger ikke med fra GitHub. Fa den fra gruppen via privat deling,'
        Write-Host 'eller legg inn egne bilder i data/faces/<person>/ og tren med:'
        Write-Host '  .\.venv\Scripts\python.exe src/train.py'
        Write-Host 'Treningen krever minst 10 unike bilder per person i face_utils.py.'
    }
} catch {
    Write-Host ''
    Write-Host ('Oppsettet stoppet: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
