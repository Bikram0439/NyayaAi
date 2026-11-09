
import streamlit as st

from Retrieval import (
    vectorstore,
    llm,
    rag_prompt,
    general_prompt,
)

st.set_page_config(
    page_title="NyayaAI - Legal Assistant",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ NyayaAI")
st.caption("Indian Legal Research Assistant")
st.info(
    "Ask questions about legal documents in the database "
    "or ask general knowledge questions."
)

# Sidebar
with st.sidebar:
    st.header("About NyayaAI")
    st.write("AI-powered legal document search and question answering.")
    st.divider()
    st.write("**Knowledge base:** Your existing legal PDFs")
    st.write("**Retrieval:** FAISS")
    st.write("**Language model:** Llama 3.2 via Ollama")
    st.warning(
        "General answers are not verified against the legal database."
    )

# Keep conversation history during the session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("mode"):
            st.caption(f"Mode: {message['mode']}")

        if message.get("sources"):
            with st.expander("View source documents"):
                for source in message["sources"]:
                    st.write(f"**File:** {source['file']}")
                    st.write(f"**Page:** {source['page']}")
                    st.write(f"**Distance:** {source['distance']:.4f}")
                    st.write("---")


def generate_answer(question):
    # Retrieve documents using your existing FAISS index
    results = vectorstore.similarity_search_with_score(
        question,
        k=5,
    )

    # Assumes the existing index uses L2 distance.
    # Calibrate this threshold on a labelled test set.
    threshold = 1.0

    relevant_docs = [
        (doc, float(distance))
        for doc, distance in results
        if float(distance) <= threshold
    ]

    if relevant_docs:
        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Page: {doc.metadata.get('page', 'Unknown')}\n"
            f"Content: {doc.page_content}"
            for doc, distance in relevant_docs
        )

        response = (rag_prompt | llm).invoke({
            "question": question,
            "context": context,
        })

        mode = "RAG - Retrieved legal documents"

        sources = [
            {
                "file": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "distance": distance,
            }
            for doc, distance in relevant_docs
        ]

    else:
        response = (general_prompt | llm).invoke({
            "question": question,
        })

        mode = "General LLM - No relevant documents selected"
        sources = []

    return response.content, mode, sources


# Chat input
question = st.chat_input("Ask your question...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                answer, mode, sources = generate_answer(question)

                st.markdown(answer)
                st.caption(f"Mode: {mode}")

                if sources:
                    with st.expander("View source documents"):
                        for source in sources:
                            st.write(f"**File:** {source['file']}")
                            st.write(f"**Page:** {source['page']}")
                            st.write(
                                f"**Distance:** "
                                f"{source['distance']:.4f}"
                            )
                            st.write("---")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "mode": mode,
                    "sources": sources,
                })

            except Exception as error:
                st.error(f"Could not generate an answer: {error}")

# Clear conversation
if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()
