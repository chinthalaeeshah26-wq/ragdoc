import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
st.set_page_config(
    page_title="Mini RAG Q&A",
    page_icon="📚",
    layout="wide"
)
st.title("📚 Mini RAG Q&A")

st.write(
    "Upload a PDF or paste a document, store it in "
    "ChromaDB, and ask questions using Ollama."
)
@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()

client = chromadb.EphemeralClient()

collection = client.get_or_create_collection(
    name="documents"
)


with st.sidebar:

    st.header("ℹ️ About This App")

    st.write(
        "This is a Mini Retrieval-Augmented "
        "Generation (RAG) application."
    )

    st.divider()

    st.subheader("🛠 Technologies")

    st.write(
        "• Python\n"
        "• Streamlit\n"
        "• PyPDF\n"
        "• Sentence Transformers\n"
        "• ChromaDB\n"
        "• Ollama\n"
        "• Llama 3.2"
    )

    st.divider()

    st.subheader("📖 How It Works")

    st.write(
        "1. Upload a PDF or paste text\n\n"
        "2. Extract the document text\n\n"
        "3. Split the text into chunks\n\n"
        "4. Create embeddings\n\n"
        "5. Store embeddings in ChromaDB\n\n"
        "6. Search relevant chunks\n\n"
        "7. Send context to Ollama\n\n"
        "8. Generate the answer"
    )

    st.divider()

    st.write(
        f"📦 Stored chunks: "
        f"{collection.count()}"
    )

st.subheader("📄 Add Document")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

document = st.text_area(
    "Or paste your document here",
    height=250,
    placeholder=(
        "Paste your notes, article, syllabus, "
        "assignment, etc..."
    )
)
pdf_text = ""

if uploaded_file is not None:

    try:

        reader = PdfReader(uploaded_file)

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        pdf_text = "\n\n".join(pages)

        st.success(
            f"PDF loaded successfully. "
            f"{len(reader.pages)} page(s) found."
        )

    except Exception as e:

        st.error(
            "Could not read the PDF."
        )

        st.code(str(e))



if st.button("➕ Add Document"):

    if pdf_text.strip():

        document_text = pdf_text

    elif document.strip():

        document_text = document

    else:

        document_text = ""


    if not document_text:

        st.warning(
            "Please upload a PDF or paste some text."
        )

    else:


        if collection.count() > 0:

            collection.delete(
                where={}
            )
        chunk_size = 500

        chunks = [
            document_text[i:i + chunk_size]
            for i in range(
                0,
                len(document_text),
                chunk_size
            )
        ]


        with st.spinner(
            "Creating document embeddings..."
        ):

            embeddings = embedding_model.encode(
                chunks
            )


        ids = [
            f"chunk_{i}"
            for i in range(
                len(chunks)
            )
        ]


        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist()
        )


        st.success(
            f"Successfully added "
            f"{len(chunks)} chunk(s) to ChromaDB."
        )


st.subheader("❓ Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What is a DBMS?"
)
if st.button("🔍 Ask AI"):
    if not question.strip():

        st.warning(
            "Please enter a question."
        )
    elif collection.count() == 0:

        st.warning(
            "Please add a document first."
        )


    else:

        try:
            with st.spinner(
                "Searching the document..."
            ):

                question_embedding = (
                    embedding_model.encode(
                        [question]
                    )[0]
                )

            number_of_results = min(
                3,
                collection.count()
            )

            results = collection.query(
                query_embeddings=[
                    question_embedding.tolist()
                ],
                n_results=number_of_results
            )

            retrieved_chunks = (
                results["documents"][0]
            )

            context = "\n\n".join(
                retrieved_chunks
            )

            prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY using the context
provided below.

Do not use outside knowledge.

If the answer is not present in the context,
say exactly:

"I don't know based on the provided document."

Keep the answer clear and concise.

---------------- CONTEXT ----------------

{context}

-------------- END CONTEXT --------------

Question:

{question}
"""
            with st.spinner(
                "Generating answer with Ollama..."
            ):

                response = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

            st.subheader("🤖 Answer")

            st.write(
                response["message"]["content"]
            )
            with st.expander(
                "🔍 Retrieved Context"
            ):

                for i, chunk in enumerate(
                    retrieved_chunks
                ):

                    st.write(
                        f"**Chunk {i + 1}:**"
                    )

                    st.write(chunk)

        except Exception as e:

            st.error(
                "Could not connect to Ollama."
            )

            st.write(
                "Make sure Ollama is installed, "
                "running, and the llama3.2 model "
                "has been downloaded."
            )

            st.code(
                str(e)
            )
st.divider()

st.caption(
    "Python + PyPDF + Sentence Transformers + "
    "ChromaDB + Ollama + Streamlit"
)
