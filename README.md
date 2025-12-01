One of the most challenging problems I solved recently was building a Chat-with-PDF application that allows users to upload documents and ask questions interactively. 
The main difficulty was implementing an efficient semantic search pipeline using FAISS and a transformer-based embedding model to retrieve the most relevant chunks from large PDFs.
I had to carefully handle text extraction, chunking, and embedding generation to maintain accuracy without slowing down the system. 
Another challenge was integrating the pipeline with Google’s Gemini API so the model could generate context-aware responses based on the retrieved chunks.
Ensuring that irrelevant or low-quality chunks didn’t confuse the model required tuning the embedding similarity thresholds. After several iterations, I achieved a smooth and accurate conversational experience where the model consistently provides reliable answers grounded in the PDF content.
