
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from agents.resume_graph import create_resume_analysis_graph

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_desc = request.form.get("job_desc")
        resume_file = request.files.get("resume")

        if not resume_file or not job_desc:
            return render_template("index.html", error="⚠ Please upload both resume and job description.")

        if resume_file.filename.endswith(".pdf"):
            reader = PdfReader(resume_file)
            resume_text = "".join([page.extract_text() for page in reader.pages])
        else:
            resume_text = resume_file.read().decode("utf-8", errors="ignore")

        graph = create_resume_analysis_graph()
        result_state = graph.invoke({"resume": resume_text, "job_desc": job_desc})

        analysis = result_state["analysis"]
        letter = result_state["letter"]

        return render_template("result.html", analysis=analysis, letter=letter)

    return render_template("index.html")

if __name__ == "__main__":
    print("🚀 Flask server starting...")
    app.run(debug=True, host="127.0.0.1", port=5000)
