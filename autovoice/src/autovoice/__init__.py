"""AutoVoice: Automotive Complaint Intelligence on GCP.

Ingests public NHTSA vehicle complaints, extracts components/symptoms with a
fine-tuned Transformer (NER), classifies severity, and answers questions with a
LangGraph RAG agent -- all served from a FastAPI app within the GCP Always Free tier.
"""

__version__ = "0.1.0"
