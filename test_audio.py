import gradio as gr
import whisper

# Load the medium model on CPU
model = whisper.load_model("medium", device="cpu")

def transcribe_live(audio):
    if audio is None:
        return ""
    # Use beam search for better accuracy
    result = model.transcribe(audio, beam_size=5, temperature=0)
    return result["text"]

def main():
    # Gradio interface
    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak Now")
    output_text = gr.Textbox(label="Transcription")

    iface = gr.Interface(
        fn=transcribe_live,
        inputs=audio_input,
        outputs=output_text,
        title="Live Speech-to-Text",
        description="Speak into your microphone and get higher-quality transcription (medium model, CPU)."
    )
    
    iface.launch()

if __name__ == "__main__":
    main()