import gradio as gr
import whisper

# Load a larger Whisper model for better accuracy
model = whisper.load_model("medium", device="cpu")

def transcribe_live(audio):
    if audio is None:
        return ""
    # Force Burmese language
    result = model.transcribe(audio, language="my")
    return result["text"]

def main():
    # Gradio interface with live transcription (no button)
    iface = gr.Interface(
        fn=transcribe_live,
        inputs=gr.Audio(sources=["microphone"], type="filepath", label="Speak here"),
        outputs=gr.Textbox(label="Transcription", placeholder="Your speech will appear here..."),
        title="Burmese Live Speech-to-Text",
        description="Speak into your microphone in Burmese. Transcription updates automatically.",
        live=True  # Enables automatic transcription as you speak
    )
    
    iface.launch()

if __name__ == "__main__":
    main()
