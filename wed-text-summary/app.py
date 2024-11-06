from flask import Flask, request, render_template, redirect, session, url_for
import os
from summarizer import url, pdf, txt, summarize_text
from test import (
    coisne_similarity,
    extract,
    idf,
    textrank_list_not_sort,
    textrank,
    textrank_list_sorted,
    tf,
    vocab_unique,
    tf_idf
)
from PyPDF2 import PdfReader

app = Flask(__name__)
app.secret_key = 'your_secret_key'
os.makedirs("uploads", exist_ok=True)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/home", methods=["GET", "POST"])
def home():
    summary = ""
    error_message = ""
    
    if request.method == "POST":
        num_sentences = int(request.form.get("num_sentences", 5))
        summarization_method = request.form.get("summarization_method")

        url_input = request.form.get("url")
        if url_input:
            text = url(url_input)
            summary = process_summary(text, summarization_method, num_sentences)

        file = request.files.get("file")
        if file and file.filename != '':
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)

            if file.filename.endswith('.pdf'):
                start_page = int(request.form.get("start_page", 1)) - 1
                end_page = int(request.form.get("end_page", start_page + 1)) - 1

                # Đọc nội dung từ trang PDF theo khoảng trang chỉ định
                text = extract_text_from_pdf(file_path, start_page, end_page)
                
            elif file.filename.endswith('.txt'):
                text = txt(file_path)
            else:
                text = ""
                error_message = "Loại file không được hỗ trợ. Chỉ hỗ trợ PDF và TXT."

            if text:
                summary = process_summary(text, summarization_method, num_sentences)

            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Có lỗi khi xóa tệp: {e}")

        text_input = request.form.get("text")
        if text_input:
            summary = process_summary(text_input, summarization_method, num_sentences)

        if summary:
            session['summary'] = summary
            return redirect(url_for('show_summary'))
        else:
            error_message = "Không có tóm tắt nào được tạo ra."

    return render_template('index.html', error=error_message)

def extract_text_from_pdf(file_path, start_page, end_page):
    """Hàm để trích xuất văn bản từ các trang PDF trong khoảng chỉ định."""
    text = ""
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        num_pages = len(reader.pages)
        
        # Đảm bảo trang cuối không vượt quá số trang thực tế
        end_page = min(end_page, num_pages - 1)
        
        for page_num in range(start_page, end_page + 1):
            text += reader.pages[page_num].extract_text()
    return text

def process_summary(text, method, num_sentences):
    """Xử lý tóm tắt dựa trên phương pháp và số câu"""
    if method == 'textrank':
        vocab = vocab_unique(text)
        tf_matrix = tf(text, vocab)
        idf_matrix = idf(text, vocab)
        tf_idf_matrix = tf_idf(tf_matrix, idf_matrix)
        cosine_matrix = coisne_similarity(tf_idf_matrix, vocab)
        textrank_matrix = textrank(cosine_matrix, esp=1e-6, d=0.85)
        textrank_list = textrank_list_not_sort(textrank_matrix)
        textrank_list_sort = textrank_list_sorted(textrank_list, num_sentences)
        return extract(textrank_list_sort, text)
    else:
        return summarize_text(text, num_sentences)

@app.route('/summary')
def show_summary():
    summary = session.get('summary', '')
    return render_template('summary.html', summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
