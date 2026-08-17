import gradio as gr
from pipeline import MultimodalMedicalPipeline

pipeline = MultimodalMedicalPipeline()

def analyze_consultation(audio_input, image_input):
    if not audio_input and not image_input:
        return "Please provide a voice recording or upload an image.", "", None

    # Step 1: Speech-to-Text via Whisper on Groq
    if audio_input:
        transcribed_text = pipeline.transcribe_audio(audio_input)
    else:
        transcribed_text = "Visual clinical assessment requested."

    # Step 2: Multimodal VLM Inference + RAG Grounding + Pydantic Schema
    analysis = pipeline.run_inference(user_query=transcribed_text, image_path=image_input)

    # Step 3: Format Structured Medical Markdown Report
    report_md = f"""
### 🩺 Clinical Assessment Report

**Primary Visual Observation:**
{analysis.primary_observation}

**Differential Diagnoses:**
- """ + "\n- ".join(analysis.differential_diagnosis) + f"""

**Triage Severity:** `{analysis.severity_level}`

**Evidence Grounding (RAG Citations):**
- """ + "\n- ".join(analysis.evidence_grounding) + f"""

**Recommended Actions:**
- """ + "\n- ".join(analysis.recommended_actions) + f"""

**⚠️ Emergency Red Flags:**
- """ + "\n- ".join(analysis.red_flags) + """
"""

    # Step 4: TTS Generation using Free gTTS
    audio_path = pipeline.generate_tts_audio(analysis.spoken_summary)

    return transcribed_text, report_md, audio_path

demo = gr.Interface(
    fn=analyze_consultation,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎙️ Patient Voice Query"),
        gr.Image(type="filepath", label="📷 Clinical/Skin Image")
    ],
    outputs=[
        gr.Textbox(label="📝 Transcribed Query"),
        gr.Markdown(label="📋 Structured Clinical Report"),
        gr.Audio(label="🔊 Doctor Voice Response", autoplay=True)
    ],
    title="🩺 Multimodal Medical AI Assistant (Vision + Voice RAG)",
    description="Production-grade AI medical consultation system with clinical RAG retrieval, Pydantic guardrails, and real-time voice synthesis.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
