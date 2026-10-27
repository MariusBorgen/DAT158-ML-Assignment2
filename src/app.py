import gradio as gr
import subprocess
import sys
import os

def capture_face():
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "capture_faces.py"
    )

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        return "Face capture script finished!"
    else:
        return f"Error:\n{result.stderr}"

demo = gr.Interface(
    fn=capture_face,
    inputs=[],
    outputs=gr.Textbox(label="Status"),
    title="Face Capture",
    description="Click the button to capture and remember your face."
)

demo.launch(share=True)