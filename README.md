# Installasjon og bruk (Windows)

## 1. Last ned prosjektet fra GitHub
Åpne Visual Studio Code og åpne terminalen via **Terminal → New Terminal**.

Gå til mappen der du ønsker å lagre prosjektet, og kjør:

```powershell
git clone https://github.com/MariusBorgen/DAT158-ML-Assignment2.git
```

Gå deretter inn i prosjektmappen:

```powershell
cd .\DAT158-ML-Assignment2\
```

## 2. Installer nødvendige biblioteker
Kjør:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

Dette oppretter Python-miljøet `.venv` og installerer alle nødvendige biblioteker automatisk.

## 3. Registrer ansikt
Kjør:
```powershell
.\.venv\Scripts\python.exe .\src\capture_faces.py
```
Skriv inn navnet ditt når programmet spør.
Programmet registrerer personen automatisk og lagrer ansiktsbildene.

## 4. Tren modellen
Kjør:
```powershell
.\.venv\Scripts\python.exe .\src\train.py
```
Programmet trener PCA + SVM ved hjelp av Olivetti Faces-datasettet og de registrerte ansiktsbildene.

## 5. Start ansiktsgjenkjenning
Kjør:
```powershell
.\.venv\Scripts\python.exe .\src\recognize_face.py
```
Kameraet åpnes og forsøker å gjenkjenne personen.
Når en person blir gjenkjent, vises:
**Welcome home, [Navn]!**


## 6. Vis treningsresultatene
Kjør:
```powershell
.\.venv\Scripts\python.exe .\src\plot_results.py
```
Dette viser grafer fra treningen.

## Viktig
- Alle kommandoene skal kjøres fra prosjektmappen `DAT158-ML-Assignment2`.
- Installasjonen trenger bare å kjøres første gang.
- Minst 10 unike ansiktsbilder kreves for å trene på en person.
- Når en ny person registreres, må modellen trenes på nytt.
- Private ansiktsbilder og den ferdigtrente modellen følger ikke med fra GitHub.