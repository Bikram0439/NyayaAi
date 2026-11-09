
import matplotlib.pyplot as plt

from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy


# 1. Load the existing FAISS database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "faiss_db",
    embeddings,
    distance_strategy=DistanceStrategy.COSINE,
    allow_dangerous_deserialization=True
)


# 2. Test questions and manually verified relevant pages
# Page numbers are zero-based.
test_cases = [
    {
        "question": "What is the Code of Civil Procedure, 1908?",
        "relevant_pages": [
            ("Data/Bihar District Courts/2025010231.pdf", 0)
        ]
    }
]


# 3. Values of k
k_values = [1, 3, 5, 10]

precision_scores = []
recall_scores = []


# 4. Normalize paths consistently
def normalize_path(path):
    return Path(path).as_posix().lstrip("./").lower()


# 5. Evaluate retrieval
for k in k_values:
    precision_sum = 0.0
    recall_sum = 0.0
    valid_questions = 0

    for test in test_cases:
        query = test["question"]

        relevant_pages = {
            (normalize_path(source), page)
            for source, page in test["relevant_pages"]
        }

        results = vectorstore.similarity_search(query, k=k)

        print(f"\nQuestion: {query}")
        print(f"Retrieved chunks for k={k}:")

        retrieved_relevant = 0

        for result in results:
            source = normalize_path(
                result.metadata.get("source", "")
            )
            page = result.metadata.get("page")

            print("Source:", source)
            print("Page:", page)
            print("Text:", result.page_content[:150])
            print("-" * 40)

            if (source, page) in relevant_pages:
                retrieved_relevant += 1

        if not relevant_pages:
            print("Warning: No relevant pages defined.")
            continue

        # Precision = relevant retrieved chunks / retrieved chunks
        precision = (
            retrieved_relevant / len(results)
            if results else 0.0
        )

        # Recall = relevant retrieved pages found / known relevant pages
        recall = min(
            retrieved_relevant / len(relevant_pages),
            1.0
        )

        precision_sum += precision
        recall_sum += recall
        valid_questions += 1

    if valid_questions:
        precision_scores.append(
            precision_sum / valid_questions * 100
        )
        recall_scores.append(
            recall_sum / valid_questions * 100
        )
    else:
        precision_scores.append(0.0)
        recall_scores.append(0.0)


# 6. Print evaluation results
print("\nk | Precision@k (%) | Recall@k (%)")
print("-" * 38)

for k, precision, recall in zip(
    k_values, precision_scores, recall_scores
):
    print(f"{k:<2}| {precision:>14.2f} | {recall:>11.2f}")


# 7. Draw and save graph
plt.figure(figsize=(9, 5))

plt.plot(
    k_values,
    precision_scores,
    marker="o",
    label="Precision@k"
)

plt.plot(
    k_values,
    recall_scores,
    marker="s",
    label="Recall@k"
)

plt.xlabel("Number of Retrieved Chunks (k)")
plt.ylabel("Score (%)")
plt.title("FAISS Retrieval Quality Evaluation")

plt.xticks(k_values)
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("retrieval_quality.png", dpi=300)
plt.show()
