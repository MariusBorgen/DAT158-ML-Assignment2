
import gradio as gr
import subprocess
import sys
import os

# Get the folder containing this file
folder = os.path.dirname(os.path.abspath(__file__))

def run_capture_faces():
    result = subprocess.run([
        sys.executable,
        os.path.join(folder, "capture_faces.py")
    ], check=True)

    return "Face captured successfully!"

def run_train():
    result = subprocess.run([
        sys.executable,
        os.path.join(folder, "train.py")
    ], check=True)

    return "Training completed!"

def learn_face():
    try:
        capture_message = run_capture_faces()
        train_message = run_train()
        return f"{capture_message}\n{train_message}"
    except subprocess.CalledProcessError as e:
        return f"An error occurred while running a script: {e}"

def recognize_face():
    try:
        result = subprocess.run([
            sys.executable,
            os.path.join(folder, "recognize_face.py")
        ], check=True)
        return "Face recognition completed!"
    except subprocess.CalledProcessError as e:
        return f"An error occurred while running the recognition script: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# Facial Recognition")

    learn_button = gr.Button("Learn Face")
    recognition_button = gr.Button("Recognize Face")
    status = gr.Textbox(label="Status")

    learn_button.click(
        fn=learn_face,
        inputs=[],
        outputs=status
    )

    recognition_button.click(
        fn=recognize_face,
        inputs=[],
        outputs=status
    )

demo.launch(share=True)
