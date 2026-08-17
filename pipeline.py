import os
import base64
import json
from typing import Optional
from dotenv import load_dotenv
from groq import Groq
from schema import MedicalAnalysisSchema
from rag_engine import MedicalRAGEngine
from gtts import gTTS

load_dotenv()

class MultimodalMedicalPipeline:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing in .env")
        self.groq_client = Groq(api_key=api_key)
        self.rag = MedicalRAGEngine()
        self.vision_model = "llama-3.2-90b-vision-preview"
        self.whisper_model = "whisper-large-v3-turbo"

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def transcribe_audio(self, audio_file_path: str) -> str:
        with open(audio_file_path, "rb") as file:
            transcription = self.groq_client.audio.transcriptions.create(
                file=(audio_file_path, file.read()),
                model=self.whisper_model,
                language="en",
                response_format="text"
            )
        return str(transcription).strip()

    def run_inference(self, user_query: str, image_path: Optional[str] = None) -> MedicalAnalysisSchema:
        retrieved_context = self.rag.retrieve_context(user_query)

        system_instruction = f"""
You are an expert Clinical AI Diagnostic Assistant.
Evaluate visual presentations alongside patient queries and ground all deductions in verified clinical evidence.

Verified Clinical Context:
{retrieved_context}

You must return a strictly valid JSON response adhering to this schema:
{json.dumps(MedicalAnalysisSchema.model_json_schema(), indent=2)}

Do NOT include conversational preambles, markdown blocks outside JSON, or assumptions without grounding.
"""

        content_payload = [{"type": "text", "text": f"Patient Query: {user_query}"}]

        if image_path:
            base64_image = self.encode_image(image_path)
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        response = self.groq_client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": content_payload}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        raw_json = response.choices[0].message.content
        return MedicalAnalysisSchema.model_validate_json(raw_json)

    def generate_tts_audio(self, text_to_speak: str, output_path: str = "doctor_response.mp3") -> str:
        tts = gTTS(text=text_to_speak, lang="en", slow=False)
        tts.save(output_path)
        return output_path
