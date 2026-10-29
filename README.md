# DAT158-ML-Assignment2

Kjør kommandoene fra prosjektmappen med prosjektets Python-miljø.

## Kom i gang på Windows

Installer Python (prosjektet er testet med 3.13). Åpne PowerShell i prosjektmappen og kjør:

```powershell
.\setup.ps1
```

Dette oppretter `.venv`, installerer avhengighetene (inkludert OpenCV) og sjekker om modellen finnes.
OpenCV tillater versjoner fra 4.10 til før 5 (`opencv-python>=4.10,<5`).
Miljøet trenger ikke aktiveres. Med modellfilen på plass starter du slik:

```powershell
.\.venv\Scripts\python.exe src/recognize_face.py
```

Private bilder og modeller følger ikke med fra GitHub; skriptet forklarer hva du gjør hvis modellen mangler.
Hvis PowerShell blokkerer skriptet, kjør `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1`.
Dette gjelder bare den prosessen og endrer ikke maskinens lagrede policy.

## Slik leser du koden

1. `src/capture_faces.py`: åpner kameraet og lagrer ansiktsbilder.
2. `src/train.py`: leser bildene, deler dem 80/20, trener og lagrer modellen.
3. `src/recognize_face.py`: bruker den lagrede modellen på nye kamerabilder.
4. `src/face_utils.py`: felles funksjoner for bildeformat, personnavn og 50 %-grensen.
5. `src/plot_results.py`: viser og lagrer grafene fra siste trening når du kjører filen.

Start med én fil om gangen og følg kommentarene ovenfra og ned.
`X` betyr bildene som tall, og `y` betyr riktig personnummer for bildene.
`fit` trener modellen. `predict_proba` beregner sannsynligheter.
`decide` velger høyeste sannsynlighet og sjekker om navnet kan vises.
PCA reduserer antall tall per bilde; SVM lærer å skille personene.

## Opptak

Start opptaket og skriv navnet på personen foran kameraet:

```powershell
.\.venv\Scripts\python.exe src/capture_faces.py
```

S starter, P pauser, Q avslutter. R sletter bare bilder fra den aktive
opptaksøkten. Programmet lagrer bare når nøyaktig ett ansikt er synlig.

Alle PNG-bilder ligger direkte i `data/faces/<person>/`.
Nye bilder får tidsstempel i filnavnet for å unngå overskriving.
Det finnes ingen train/test-mapper. Fjern eventuelle utdaterte bilder fra
personmappen og tren på nytt for å utelate dem fra modellen.

80 % trekkes tilfeldig til trening, og 20 % holdes av til evaluering.
Fast seed (30) gir samme oppdeling så lenge bildesettet er uendret.
Identiske bilder telles én gang. Det kreves minst 10 unike bilder per person.
Denne tilfeldige oppdelingen er en enklere test enn en separat kameraøkt.

## Trening og gjenkjenning

```powershell
.\.venv\Scripts\python.exe src/train.py
.\.venv\Scripts\python.exe src/plot_results.py
.\.venv\Scripts\python.exe src/recognize_face.py
```

Tren på nytt etter endringer i bildesettet, og start gjenkjenningen på nytt.
`train.py` lagrer resultatene i `reports/training_results.npz` uten å åpne grafer.
`plot_results.py` leser resultatene og åpner grafene uten ny trening.
Evaluering og kamera bruker samme beslutning: høyeste sannsynlighet må være
over 50 %, og klassen må finnes i personregisteret.
Evalueringen viser riktige navn, feil navn og avvisninger på de 20 % avholdte
bildene. Sannsynligheten er et modellestimat, ingen garanti for korrekt navn.
Olivetti-testen måler toppklassen før avvisning; den måler ikke ukjente personer.


## Welcome Home og registrerte personer

Dette er en robotdemonstrasjon som ønsker personer velkommen, uten dører eller adgangskontroll.
Nye installasjoner starter uten registrerte personer. Skriv et navn og trykk
**Registrer person** i appen. Registeret lagres automatisk i `data/people.json`,
med unike ID-er fra 40. Ta deretter bilder og tren modellen. Navn og ID-er
hentes fra registeret av opptak, trening og gjenkjenning.

Fra terminalen kan du registrere og starte opptak med:

```powershell
.\.venv\Scripts\python.exe src/capture_faces.py --register "Navn"
```

Eksisterende lokale personer er overført til registeret med sine opprinnelige
ID-er; bilder og modeller er beholdt. Registeret er privat og følger ikke med
fra GitHub. Behold registeret sammen med bilder og modeller ved flytting.
Trening hopper over personer med færre enn 10 unike bilder og stopper uten
å erstatte modellen hvis ingen registrerte personer har nok bilder.

Bruk `--person "Navn"` ved opptak for å unngå terminalspørsmål. Opptaket tar
fire runder à 25 bilder med pause for endret belysning og fem sekunders
nedtelling ved hver start. Maksimum er 100 bilder per opptaksøkt.
Trening og analyse bruker de første 100 unike bildene per person sortert etter
filnavn, deretter 80/20-deling. Flere opptak erstatter ikke automatisk gamle bilder.
Ved 100 unike bilder blir det 80 treningsbilder og 20 testbilder per person.

Kameraet viser modellens umiddelbare estimat og en separat stabil velkomst.
Tre påfølgende godkjente prediksjoner kreves (omtrent 0,6 sekunder fra første
prediksjon ved 0,3 sekunders intervall). Hendelsen skrives én gang i terminalen
via `on_welcome`. Minst tre sekunder uten synlig ansikt åpner for ny velkomst.
50 %-grensen er uendret og garanterer ikke at ukjente personer avvises.

## Analyse og uavhengig kameraøkt

```powershell
.\.venv\Scripts\python.exe src/compare_models.py
.\.venv\Scripts\python.exe src/check_analysis.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Analysen bruker bare treningsdelen og tilpasser PCA separat i hver fold.
Den sammenligner PCA med 25–150 komponenter, whitening av/på og fire algoritmer.
Hovedmodellen har 100 komponenter, uten whitening, og SVM RBF med `C=10`
(allerede en lokal endring ved denne gjennomgangen; Git-versjonen hadde `C=20`).
Analysen bruker også `C=10`, men PCA har eksplisitt `svd_solver="full"`, mens
trening bruker standard `auto`. Analysen bruker `predict()` uten sannsynligheter;
appen bruker høyeste `predict_proba()` med avvisning. Disse kan velge ulike klasser.
Tidligere lagrede modeller/resultater er ikke trent eller beregnet på nytt.

Bilder tatt med 0,3 sekunders mellomrom kan være nesten identiske. Tilfeldig
80/20-deling og kryssvalidering kan derfor gi optimistiske resultater.
Samle en **ny, separat kameraøkt**, gjerne en annen dag, med ulike lys og vinkler:

```powershell
.\.venv\Scripts\python.exe src/capture_faces.py --person "Navn" --evaluation
.\.venv\Scripts\python.exe src/evaluate_session.py
```

Gjenta for hver kjent person. Disse bildene ligger i `data/evaluation/<navn>/`
og brukes aldri av trenings- eller analyseskriptet. For ukjente personer legges
64×64 gråtoneansiktsutsnitt i `data/evaluation/unknown/`. Ikke kopier treningsbilder.
Evaluatoren avviser eksakte kopier fra eksisterende personmapper, men kan ikke
bevise at bildene kommer fra en ny økt. Hold disse dataene utenfor modellvalg.
Den skriver JSON i terminalen med accuracy før avvisning (bare kjente), riktige
navn, feil navn, avvisninger og feil navn på ukjente. Dette er beslutninger per
bilde, ikke tidsstabiliserte velkomsthendelser. Målt tid gjelder batchprediksjon,
uten kamera, deteksjon og bildebehandling. Ingen nye målinger følger med.

## Lokalt grensesnitt og avhengigheter

```powershell
.\.venv\Scripts\python.exe src/app.py
```

Velg person og start opptak. Q lukker kameravinduet; tren deretter eksplisitt.
En kjøring av gangen hindrer konkurrerende bruk av kamera/modell. Trening er
ferdig først når ferdigmeldingen vises. Appen kjører med `share=False`.
`requirements.txt` inneholder direkte avhengigheter; pip installerer deres
underavhengigheter. Versjonskravene for de beholdte pakkene er uendret.

Private bilder, evalueringer og lokale modeller ignoreres av Git. Git sporer
fortsatt et eldre `models/face_recognition_model.joblib`, Olivetti-arkivet og et
offentlig `.gradio/certificate.pem`; ignoreringsregler fjerner ikke sporede filer.
Sertifikatet inneholder ingen privat nøkkel. Last bare modellfiler fra betrodde
kilder. Den tilpassede modellen må mottas privat eller trenes fra egne bilder.
