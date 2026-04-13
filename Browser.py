import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
)

from flask import Flask, request, render_template
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from google import genai

app = Flask(__name__)


def _fallback_answer_from_context(context):
    if not context:
        return "I could not find a relevant answer in the document."
    page_text = context[0].page_content.strip().replace("\n", " ")
    snippet = page_text[:900]
    return f"Gemini quota is exhausted right now. Best matching text from your document: {snippet}"

# Route for chat UI (GET for page, POST for question)
@app.route("/", methods=["GET", "POST"])
def chat():
    question = None
    answer = None
    if request.method == "POST":
        question = request.form.get("question")
        if question:
            try:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                new_vector_store = FAISS.load_local(
                    "tcs_doc_index",
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                context = new_vector_store.similarity_search(question, k=1)
                our_prompt = f"""I am going to ask a question based on the context provided.
                \nPlease answer the question based on the context.\nIf you don't know or don't 
                get answer from the context say 'I didn't find the answer in the context.'
                \n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"""

                api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyBq-ZTkWoe82NF51IQkOI6HYToyDfHXTEg")
                client = genai.Client(api_key=api_key)
                try:
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=our_prompt,
                    )
                    answer = response.text
                except Exception as llm_exc:
                    if "RESOURCE_EXHAUSTED" in str(llm_exc) or "429" in str(llm_exc):
                        answer = _fallback_answer_from_context(context)
                    else:
                        answer = f"Error: {llm_exc}"
            except Exception as exc:
                answer = f"Error: {exc}"
    return render_template("index.html", question=question, answer=answer)


if __name__ == "__main__":
    app.run(debug=False, port=5600)
