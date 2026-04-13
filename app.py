'''
Step1: read qn 
step2: convert qn to vector using embedding model
step3: load faiss db, get revelent chunk
step4: make prompt with qn and contex
step5: pass prompt to any free LLM
step6: once we get answer from LLM, return answer to POSTMAN

'''
import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
)

from flask import Flask, request, jsonify, redirect
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


@app.route("/")
def home():
    return redirect("http://127.0.0.1:5600")


@app.route("/tcs", methods = ["POST"])
def TcsChatbotApi():
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("tcs_question")
        if not question:
            return jsonify({"error": "Please provide 'tcs_question' in JSON body."}), 400

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        new_vector_store = FAISS.load_local(
            "tcs_doc_index",
            embeddings,
            allow_dangerous_deserialization=True,
        )
        context = new_vector_store.similarity_search(question, k=1)

        our_prompt = f'''I am going to ask a question based on the context provided.
Please answer the question based on the context.
If you don't know or don't get an answer from the context, say: I didn't find the answer in the context.

Context: {context}

Question: {question}

Answer:'''

        api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyBq-ZTkWoe82NF51IQkOI6HYToyDfHXTEg")
        client = genai.Client(api_key=api_key)
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=our_prompt,
            )
            return jsonify({"response": response.text})
        except Exception as llm_exc:
            if "RESOURCE_EXHAUSTED" in str(llm_exc) or "429" in str(llm_exc):
                return jsonify({"response": _fallback_answer_from_context(context)})
            return jsonify({"response": f"Unable to process request: {llm_exc}"})
    except Exception as exc:
        return jsonify({"response": f"Unable to process request: {exc}"})



# print(response.output_text)
# def f1():
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#     question_vector = embeddings.embed_query("What is the revenue of TCS")

#     # Load the Faiss database and get relevant chunks
#     new_vector_store = FAISS.load_local("tcs_doc_index", embeddings, allow_dangerous_deserialization=True)
#     # Ask quetion to the vector store
#     context = new_vector_store.similarity_search_by_vector(question_vector, k=1)
#     print(context)

# f1()
if __name__ == "__main__":
    app.run(debug=False, port=5601)
