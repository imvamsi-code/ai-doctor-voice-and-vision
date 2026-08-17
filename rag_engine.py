import chromadb
from chromadb.utils import embedding_functions

class MedicalRAGEngine:
    def __init__(self, collection_name: str = "clinical_guidelines"):
        self.client = chromadb.Client()
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )
        self._seed_clinical_knowledge_base()

    def _seed_clinical_knowledge_base(self):
        """Seeds vector store with standard clinical dermatological & triage protocols."""
        if self.collection.count() == 0:
            guidelines = [
                "Acne Vulgaris: Characterized by inflammatory papules, pustules, and comedones. First-line treatments include topical retinoids, benzoyl peroxide, and salicylic acid. Severe cystic acne requires dermatologist evaluation for oral isotretinoin.",
                "Contact Dermatitis & Eczema: Erythematous pruritic plaques, scaling, and skin barrier disruption. Standard protocol includes identifying allergens, applying topical hydrocortisone/corticosteroids, and maintaining ceramide-based moisturization.",
                "Seborrheic Dermatitis / Dandruff: Malassezia yeast proliferation causing flaking, erythema on scalp/face. Managed with antifungal ketoconazole 2% shampoo, zinc pyrithione, and selenium sulfide.",
                "Erythema Multiforme / Urticaria Red Flags: Rapidly spreading rash with fever, blistering, mucosal involvement (lips/eyes), or difficulty breathing indicates potential severe drug reaction or anaphylaxis requiring immediate ER triage.",
                "Tinea Infections (Ringworm): Annular scaly plaques with central clearing. Managed with topical antifungals (terbinafine, clotrimazole). Avoid topical corticosteroids which worsen fungal proliferation."
            ]
            doc_ids = [f"guideline_{i}" for i in range(len(guidelines))]
            self.collection.add(
                documents=guidelines,
                ids=doc_ids,
                metadatas=[{"source": "Clinical Practice Guidelines 2026"} for _ in guidelines]
            )

    def retrieve_context(self, query: str, n_results: int = 2) -> str:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        docs = results.get("documents", [[]])[0]
        return "\n---\n".join(docs) if docs else "No specific guideline matched."
