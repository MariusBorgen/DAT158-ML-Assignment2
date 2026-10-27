# DAT158-ML-Assignment2

Kjør kommandoene fra prosjektmappen med prosjektets Python-miljø.

# VIKTIG, KJØR "pip install "opencv-python<5" I TERMINAL FOR Å KUNNE BRUKE PROGRAMMET

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

Velg person (`Marius`, `Endre` eller `David`):

```powershell
python src/capture_faces.py --person Marius
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
python src/train.py
python src/plot_results.py
python src/recognize_face.py
```

Tren på nytt etter endringer i bildesettet, og start gjenkjenningen på nytt.
`train.py` lagrer resultatene i `reports/training_results.npz` uten å åpne grafer.
`plot_results.py` leser resultatene og åpner grafene uten ny trening.
Evaluering og kamera bruker samme beslutning: høyeste sannsynlighet må være
over 50 %, og klassen må være en av de tre navngitte personene.
Evalueringen viser riktige navn, feil navn og avvisninger på de 20 % avholdte
bildene. Sannsynligheten er et modellestimat, ingen garanti for korrekt navn.
Olivetti-testen måler toppklassen før avvisning; den måler ikke ukjente personer.
