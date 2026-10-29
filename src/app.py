"""Lokalt grensesnitt. Kameraet vises i egne OpenCV-vinduer."""
import os
import subprocess
import sys
import threading
import gradio as gr
from face_utils import ROOT, PERSON_LABELS, refresh_people, register_person

process_lock = threading.Lock()


def run_script(script, *arguments):
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / script), *arguments],
        cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or "Ukjent feil")[-6000:])
    return result.stdout[-6000:]


def execute(script, *arguments):
    if not process_lock.acquire(blocking=False):
        yield "Et program kjører allerede. Avslutt kameravinduet med Q."
        return
    try:
        yield "Kjører. Kamera: S starter, Q avslutter. Trening er ikke ferdig før ferdigmeldingen vises."
        output = run_script(script, *arguments)
        yield "Programmet er avsluttet.\n" + output
    except (RuntimeError, OSError) as error:
        yield f"Programmet feilet:\n{error}"
    finally:
        process_lock.release()


def capture(person):
    refresh_people()
    if person not in PERSON_LABELS:
        yield "Velg en registrert person."
        return
    yield from execute("capture_faces.py", "--person", person)


def train():
    yield from execute("train.py")


def recognize():
    yield from execute("recognize_face.py")


def register(name):
    try:
        name = register_person(name)
        return gr.Dropdown(choices=list(PERSON_LABELS), value=name), "", f"Registrert {name}. Ta bilder og tren modellen."
    except (ValueError, OSError) as error:
        return gr.skip(), gr.skip(), str(error)


def main():
    refresh_people()
    with gr.Blocks() as demo:
        gr.Markdown("# Welcome Home\nKameraet åpnes lokalt. Avslutt med Q. Trening startes separat etter opptak.")
        person = gr.Dropdown(choices=list(PERSON_LABELS), value=next(iter(PERSON_LABELS), None), label="Person")
        new_name = gr.Textbox(label="Navn på ny person")
        register_button = gr.Button("Registrer person")
        capture_button = gr.Button("Ta bilder")
        train_button = gr.Button("Tren modell (erstatter lokal modell ved fullføring)")
        recognition_button = gr.Button("Gjenkjenn ansikt")
        status = gr.Textbox(label="Status")
        register_button.click(register, inputs=new_name, outputs=[person, new_name, status], concurrency_id="local_process")
        capture_button.click(capture, inputs=person, outputs=status, concurrency_id="local_process")
        train_button.click(train, outputs=status, concurrency_id="local_process")
        recognition_button.click(recognize, outputs=status, concurrency_id="local_process")
    demo.queue().launch(share=False)


if __name__ == "__main__":
    main()
