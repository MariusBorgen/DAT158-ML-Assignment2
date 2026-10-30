"""Ta ansiktsbilder og lagre dem i mappen til valgt person."""
import cv2
import time
from datetime import datetime
import argparse
from face_utils import ROOT, PERSON_LABELS, prepare_face, detect_faces, register_person

# Innstillinger du kan endre.
MAX_IMAGES = 100
IMAGES_PER_ROUND = 25
SAVE_INTERVAL = 0.3
START_DELAY = 5  # Tid til å gå tilbake til døren etter at du trykker S.

def main():
    # --person lar deg velge navn fra terminalen uten å endre denne filen.
    parser = argparse.ArgumentParser() #Oppretter en argument parser 
    parser.add_argument("--person", choices=PERSON_LABELS, default=None) #Legger til et argument for å velge en person fra PERSON_LABELS
    parser.add_argument("--register", metavar="NAVN", help="Registrer en ny person før opptak") #Legger til et argument for å registrere en ny person
    parser.add_argument("--evaluation", action="store_true", help="Lagre separat fra treningsbildene")# Legger til et argument for å lagre bildene separat fra treningsbildene
    args = parser.parse_args() # lagrer argumentene fra parser i args
    
    # Hvis --register er spesifisert, prøv å registrere personen. Hvis det mislykkes, vis en feilmelding.
    if args.register is not None:
        try:
            args.person = register_person(args.register)
        except (ValueError, OSError) as error:
            parser.error(str(error))

    # Hvis ingen person er spesifisert, be brukeren om å velge en fra listen over registrerte personer, eller registrere en ny person.
    if args.person is None:
        print("Registrerte personer: " + ", ".join(PERSON_LABELS))
        args.person = input("Velg person: ").strip()
        existing_name = next(
            (name for name in PERSON_LABELS if name.casefold() == args.person.casefold()),
            None,
        )

        # Hvis brukeren skrev inn et navn som allerede er registrert, bruk det eksisterende navnet. Hvis ikke, prøv å registrere det nye navnet.
        if existing_name is not None:
            args.person = existing_name
        else:
            try:
                args.person = register_person(args.person)
            except (ValueError, OSError) as error:
                parser.error(str(error))

    # Lagre bildene i en mappe som er spesifikk for personen, og evaluering.
    save_dir = ROOT / "data" / ("evaluation" if args.evaluation else "faces") / args.person
    save_dir.mkdir(parents=True, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") #Genererer en unik ID for denne økten basert på dato og klokkeslett
    session_files = []  # Bare filene som lagres i denne kjøringen.

    # OpenCV har en ferdig detektor som finner ansiktsrektangelet.
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    camera = cv2.VideoCapture(0)  # 0 er standardkameraet.

    # Kaster en feil hvis kameraet ikke kan åpnes
    if not camera.isOpened():
        raise RuntimeError("Kunne ikke åpne kameraet.")

    #Standardverdier for opptak, venting og tidsstempling
    last_capture = 0
    capturing = False
    waiting_for_light_change = False
    capture_starts_at = 0

    # Gi brukeren instruksjoner om hvordan de skal bruke programmet.
    print("S = Start | P = Pause | R = Reset | Q = Quit")
    print(f"Saving to {save_dir}. R resets only this session.")
    print("La kameraet stå på samme sted. Kom inn gjennom samme dør i hver runde.")
    print(f"Etter {IMAGES_PER_ROUND} bilder pauses opptaket: endre lyset og trykk S for neste runde.")

    # Hovedløkken for å lese fra kameraet, finne ansikter og lagre bilder.
    try:
        while True:
            ret, frame = camera.read() # Leser et bilde fra kameraet. ret er True hvis det lykkes, frame er bildet.
            if not ret: # Hvis det ikke lykkes å lese fra kameraet, avslutt løkken.
                break

            # frame er ett kamerabilde. gray er det samme bildet i gråtoner.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(detector, gray)
            now = time.monotonic() # Nåværende tid i sekunder siden programstart.

            # Hvis det er nøyaktig ett ansikt i bildet, lagre ansiktsutsnittet.
            if len(faces) == 1:
                x, y, w, h = faces[0]  # Venstre kant, topp, bredde og høyde.

                # Tegn et grønt rektangel rundt ansiktet i bildet.
                cv2.rectangle(
                    frame, (x, y), (x+w, y+h),
                    (0, 255, 0), 2
                )

                enough_time = now - last_capture >= SAVE_INTERVAL # Sjekker om nok tid har gått siden forrige opptak.

                # Hvis vi er i opptaksmodus, og det har gått nok tid siden forrige opptak, og vi har ikke nådd maks antall bilder, lagre ansiktsutsnittet.
                if capturing and now >= capture_starts_at and len(session_files) < MAX_IMAGES and enough_time:
                    face = prepare_face(gray, (x, y, w, h))

                    # Tidsstempel beskytter gamle opptak mot overskriving.
                    filename = save_dir / f"{session_id}_face_{len(session_files):03d}.png"
                    if cv2.imwrite(str(filename), face):
                        session_files.append(filename)
                        last_capture = now # Oppdaterer tidspunktet for siste opptak.

                        # Sett capturng til False om vi har nådd maks antall bilder
                        if len(session_files) >= MAX_IMAGES:
                            capturing = False
                            print(f"Ferdig: {MAX_IMAGES} bilder lagret.")
                        # Hvis bilder per runde er nådd, pause opptaket og be brukeren om å endre lyset i rommet.
                        elif len(session_files) % IMAGES_PER_ROUND == 0:
                            capturing = False
                            waiting_for_light_change = True
                            print(f"{len(session_files)}/{MAX_IMAGES} bilder lagret. Endre lyset i rommet.")
                            print("Trykk S når du er klar for neste runde.")

            #Dersom vi har nådd maks antall bilder, stopp opptaket.
            if len(session_files) >= MAX_IMAGES:
                capturing = False

            status = "READY / PAUSED" #Vise status på skjermen. Klar til opptak eller pauset.

            # Sett status til recording dersom capturing er true 
            if capturing:
                status = "RECORDING"
            
            # Sett status til completed dersom maks antall bilder er nådd
            if len(session_files) >= MAX_IMAGES:
                status = "COMPLETED"
            
            # Sett status til paused dersom vi venter på at brukeren skal endre lyset i rommet
            elif waiting_for_light_change:
                status = "PAUSED - CHANGE ROOM LIGHTING"

            # Dersom vi venter på at opptaket skal starte, vis en nedtelling til opptaket starter
            elif capturing and now < capture_starts_at:
                seconds_left = int(capture_starts_at - now) + 1
                status = f"START IN {seconds_left}s"

            # Sett status til "Need exactly one face" dersom vi er i opptaksmodus, men det ikke er nøyaktig ett ansikt i bildet
            elif capturing and len(faces) != 1:
                status = "Need exactly one face"

            # Vis antall lagrede bilder, og status på skjermen
            cv2.putText(
                frame, f"{status} - {len(session_files)}/{MAX_IMAGES}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2 
            )

            # Vis instruksjoner på skjermen dersom vi venter på at brukeren skal endre lyset i rommet
            if waiting_for_light_change:
                cv2.putText(
                    frame, "Change the light.",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2,
                )
                cv2.putText(
                    frame, f"Press S for next round ({START_DELAY}s countdown).",
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2,
                )

            # Vis instruksjoner på skjermen for hvordan man starter, pauser, tilbakestiller og avslutter programmet
            cv2.putText(
                frame, "S: Start  P: Pause  R: Reset  Q: Quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2
            )

            cv2.imshow("Capture Faces - DAT158", frame) #Viser bildet med ansiktsdeteksjon og status på skjermen.

            key = cv2.waitKey(1) & 0xFF  # Les en tast uten å stoppe kameravisningen.

            # Håndter tastetrykk for å starte programmet.
            if key == ord("s"):
                if not capturing and len(session_files) < MAX_IMAGES:
                    capturing = True
                    waiting_for_light_change = False
                    capture_starts_at = time.monotonic() + START_DELAY
                    last_capture = 0

            # Håndter tastetrykk for å pause programmet.
            elif key == ord("p"):
                capturing = False

            # Håndter tastetrykk for å tilbakestille programmet. Dette sletter bare bildene som er tatt i denne økten, ikke eldre bilder.
            elif key == ord("r"):
                capturing = False
                waiting_for_light_change = False
                capture_starts_at = 0
                # R fjerner bare denne øktens bilder, aldri eldre bilder.
                for file in session_files:
                    file.unlink()
                session_files.clear()
                last_capture = 0

            # Håndter tastetrykk for å avslutte programmet.
            elif key == ord("q"):
                break

    finally:
        # Frigjør kameraet også hvis programmet stopper med en feil.
        camera.release()
        cv2.destroyAllWindows()

    print(f"Saved {len(session_files)} images in {save_dir}") # Skriver ut hvor mange bilder som ble lagret og i hvilken mappe.


if __name__ == "__main__":
    main()
