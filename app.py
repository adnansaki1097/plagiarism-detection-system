# MD Adnan Saki
# https://www.linkedin.com/in/adnan-saki/
# https://github.com/adnansaki1097

from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def calculate_similarity(documents):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(documents)
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        files = request.files.getlist("files")
        documents = [file.read().decode("utf-8") for file in files]
        similarity_matrix = calculate_similarity(documents)
        results = similarity_matrix
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
