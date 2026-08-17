# 🩺 Multimodal Medical AI Assistant — Vision + Voice RAG

A **multimodal clinical decision-support prototype** that combines **Automatic Speech Recognition (ASR), Vision-Language Models (VLMs), Retrieval-Augmented Generation (RAG), Pydantic validation, and Text-to-Speech (TTS)** into a modular AI pipeline.

The system demonstrates how modern AI components can be orchestrated to process **patient voice queries and medical images**, retrieve relevant clinical knowledge, generate structured assessments, identify potential red flags, and return both **structured and voice-based responses**.

> ⚠️ **Disclaimer:** This project is an AI research/portfolio prototype and is **not intended to provide medical diagnosis, treatment, or replace qualified healthcare professionals.** Outputs should not be used for clinical decision-making.

---

## 🌟 Key Features

### 🎙️ Speech-to-Text — ASR

Uses **OpenAI Whisper Large-v3-Turbo through Groq** to transcribe patient voice recordings into text with low-latency inference.

- Voice-based symptom input
- Fast speech transcription
- Supports natural-language patient queries
- Forms the first stage of the multimodal pipeline

### 👁️ Multimodal Clinical Vision

Uses a **Vision-Language Model (VLM)** to analyze uploaded medical imagery alongside the patient's textual description.

Example inputs include:

- Skin rashes
- Dermatological lesions
- Visible skin abnormalities
- Other supported medical imagery

The vision component combines the image with the retrieved clinical context rather than analyzing the image in isolation.

### 📚 Retrieval-Augmented Generation — RAG

Uses **ChromaDB** as a local vector database to retrieve relevant information from a curated clinical knowledge base.

The retrieval pipeline uses:

- **ChromaDB**
- **Sentence Transformers**
- **all-MiniLM-L6-v2 embeddings**
- Curated clinical reference material

This provides contextual grounding for the language model and reduces reliance on unsupported model-generated information.

### 🛡️ Pydantic Guardrails

The generated response is validated using **Pydantic schemas** before being returned to the user.

The schema layer is responsible for enforcing structured output such as:

- Differential considerations
- Severity / triage classification
- Potential red flags
- Supporting evidence
- Recommended next steps

This helps prevent malformed or inconsistent model outputs from reaching the application layer.

### 🔊 Text-to-Speech

Uses **gTTS (Google Text-to-Speech)** to convert the generated response into an audio file.

The application can therefore provide both:

**Structured text → Audio response**

without requiring a separate paid TTS API.

### 🖥️ Interactive Gradio Interface

A Gradio-based interface provides a simple way to interact with the complete multimodal pipeline.

Users can:

- Record a voice query
- Upload an image
- Submit symptoms
- View structured results
- Listen to the generated response

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────────┐
                         │        User / UI         │
                         │        app.py            │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │   Multimodal Pipeline    │
                         │      pipeline.py         │
                         └────────────┬─────────────┘
                                      │
                  ┌───────────────────┼───────────────────┐
                  │                   │                   │
                  ▼                   ▼                   ▼
           ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
           │ Whisper ASR │    │ ChromaDB RAG │    │ Vision VLM  │
           │   Groq      │    │   Retrieval  │    │   Analysis  │
           └──────┬──────┘    └──────┬───────┘    └──────┬──────┘
                  │                  │                   │
                  └──────────────────┼───────────────────┘
                                     │
                                     ▼
                         ┌──────────────────────────┐
                         │   Pydantic Validation    │
                         │       schema.py          │
                         └────────────┬─────────────┘
                                      │
                           ┌──────────┴──────────┐
                           │                     │
                           ▼                     ▼
                  ┌─────────────────┐   ┌─────────────────┐
                  │ Structured      │   │ gTTS            │
                  │ Clinical Output │   │ Voice Response  │
                  └─────────────────┘   └─────────────────┘
```

---

# 🔄 End-to-End Workflow

```text
Patient Voice / Image
        │
        ▼
   Whisper ASR
        │
        ▼
 Transcribed Query
        │
        ├──────────────────────┐
        │                      │
        ▼                      ▼
 Clinical RAG             Medical Image
 Retrieval                    │
        │                      │
        └──────────┬───────────┘
                   ▼
             Vision-Language
                 Model
                   │
                   ▼
          Structured Generation
                   │
                   ▼
          Pydantic Validation
                   │
             ┌─────┴─────┐
             ▼           ▼
      Clinical Output   gTTS
                         │
                         ▼
                  Audio Response
```

### 1. 🎙️ Patient Voice Input

The user records a description of their symptoms.

For example:

> "I have a red, itchy and scaly patch on my hand."

Whisper converts the audio into text.

### 2. 📚 Knowledge Retrieval

The transcribed query is passed to the RAG layer.

ChromaDB retrieves semantically relevant information from the curated clinical knowledge base.

### 3. 👁️ Vision Analysis

If an image is provided, the Vision-Language Model receives:

- The uploaded image
- Patient's transcribed query
- Retrieved contextual information

The model then generates a structured assessment.

### 4. 🛡️ Output Validation

The generated response is passed through the Pydantic schema layer.

The system validates the structure and extracts fields such as:

- Possible conditions
- Severity
- Red flags
- Supporting evidence
- Suggested next steps

### 5. 🔊 Voice Response

The validated response is converted into speech using gTTS and presented through the Gradio interface.

---

# 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| UI | Gradio |
| ASR | Whisper Large-v3-Turbo |
| Inference | Groq |
| Vision | Llama Vision |
| RAG | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 |
| Validation | Pydantic |
| TTS | gTTS |
| Environment Management | python-dotenv |

---

# 📁 Project Structure

```text
medical-voice-rag-assistant/
│
├── app.py                  # Gradio application / UI
├── pipeline.py             # End-to-end multimodal pipeline
├── rag_engine.py           # ChromaDB retrieval and embeddings
├── schema.py               # Pydantic output schemas
│
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore              # Git exclusions
└── README.md               # Project documentation
```

The project follows a **modular architecture**, separating:

```text
UI
 │
 ├── Pipeline
 │
 ├── ASR
 │
 ├── RAG
 │
 ├── Vision
 │
 ├── Validation
 │
 └── TTS
```

This makes individual components easier to test, replace, and extend.

---

# 🚀 Getting Started

## Prerequisites

Make sure you have:

- Python 3.10+
- Internet connection
- Git
- A Groq API key

---

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/medical-voice-rag-assistant.git

cd medical-voice-rag-assistant
```

Alternatively, download the repository as a ZIP and extract it locally.

---

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY="your_groq_api_key_here"
```

Never commit API keys or `.env` files to GitHub.

---

## 5. Run the Application

```bash
python app.py
```

The Gradio interface will be available locally at:

```text
http://localhost:7860
```

---

# 💡 Example Use Cases

The system can process multimodal queries such as:

```text
"I developed a red, scaly rash on my hand that is painful and itchy."
```

```text
"Can you analyze this skin image and explain what conditions could potentially cause these symptoms?"
```

```text
"I have circular, scaly patches on my arm with central clearing."
```

The system combines the patient's description, optional image input, and retrieved knowledge to produce a **structured, evidence-grounded response**.

---

# 🧠 Why This Architecture?

Instead of building a single large LLM prompt, the application separates the workflow into specialized components:

```text
Speech
   ↓
ASR
   ↓
Knowledge Retrieval
   ↓
Vision + Language Reasoning
   ↓
Schema Validation
   ↓
Structured Response
   ↓
Text-to-Speech
```

This architecture provides several advantages:

### Modularity

Each component can be independently replaced or upgraded.

### Grounding

RAG provides external context instead of relying entirely on the model's parametric knowledge.

### Structured Outputs

Pydantic ensures that downstream components receive predictable data.

### Multimodal Reasoning

The system can combine text, speech-derived information, and visual inputs.

### Extensibility

Additional agents, knowledge sources, models, and interfaces can be added without rewriting the entire application.

---

# 🔮 Future Improvements

The current implementation can be extended into a more robust research platform with:

### 🧠 Multi-Turn Medical Memory

Add conversation state so the system can maintain context across multiple interactions.

### 🏥 FHIR / EHR Integration

Integrate standardized healthcare data formats and simulated EHR workflows for research purposes.

### 🩺 Specialist Routing

Build a routing layer capable of selecting specialized knowledge bases or models for:

- Dermatology
- Cardiology
- Pediatrics
- Radiology
- General medicine

### 🧪 Evaluation Framework

Create automated evaluation datasets for measuring:

- Retrieval accuracy
- Groundedness
- Schema compliance
- Hallucination rate
- Triage classification accuracy
- Vision-language consistency

### 🐳 Containerization

Package the application using Docker for reproducible deployment.

### ☁️ Cloud Deployment

Deploy the application to platforms such as Hugging Face Spaces or cloud infrastructure.

### 🔐 Production Security

Add:

- Authentication
- Rate limiting
- Secure secret management
- Input validation
- Audit logging
- Data privacy controls

---

# 📊 Engineering Highlights

This project demonstrates practical implementation of:

- **Multimodal AI pipelines**
- **Vision-Language Models**
- **Retrieval-Augmented Generation**
- **Vector databases**
- **Speech recognition**
- **Structured LLM outputs**
- **Pydantic validation**
- **LLM orchestration**
- **Modular Python architecture**
- **Interactive AI interfaces**
- **AI system evaluation concepts**

Rather than treating an LLM as a single end-to-end black box, the project demonstrates how multiple specialized AI components can be composed into a **structured multimodal system**.

---

# 👨‍💻 Author

**Saivamsi Kanithi**

Built as an advanced portfolio project exploring:

**Multimodal AI • RAG • Vision-Language Models • Voice AI • LLM Guardrails • Vector Search • AI System Design**

---

## ⚠️ Medical Disclaimer

This application is an **educational and research-oriented AI prototype**.

It is not a medical device and should not be used to diagnose diseases, prescribe medication, determine treatment, or make emergency medical decisions. Always consult a qualified healthcare professional for medical advice.                 ▼
