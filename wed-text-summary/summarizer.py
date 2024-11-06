import math
import nltk
import re
import requests
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Khởi tạo các biến và đối tượng cần thiết
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


# Định nghĩa hàm tf, idf, tf_idf
def tf_z(term, doc):
    return doc.count(term) / len(doc)


def idf_z(term, docs):
    num_docs_with_term = sum(1 for doc in docs if term in doc)
    return math.log(len(docs) / (1 + num_docs_with_term))


def tf_idf_z(term, doc, docs):
    return tf_z(term, doc) * idf_z(term, docs)


# Hàm phân loại từ loại cho lemmatizer
def word_classification(word):
    if word[1].startswith('N'):
        return 'n'
    elif word[1].startswith('V'):
        return 'v'
    elif word[1].startswith('J'):
        return 'a'
    elif word[1].startswith('R'):
        return 'r'
    else:
        return 'n'


# Hàm tiền xử lý văn bản
def text_preprocessing(sentences):
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r"\W", " ", sentence)
        sentence = re.sub(r"\s+", " ", sentence)
        tokens = [word for word in word_tokenize(sentence) if word not in stop_words]
        pos_tags = pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(word[0], word_classification(word)) for word in pos_tags]
        processed_sentences.append(" ".join(lemmatized_tokens))
    return processed_sentences


# Xử lý văn bản từ URL
def url(url_path):
    file_html = requests.get(url_path)
    file_raw = BeautifulSoup(file_html.content, "html.parser")
    get_p_tags = file_raw.find_all("p")
    text = ""
    for p_tag in get_p_tags:
        text += p_tag.get_text()
    return text


# Xử lý văn bản từ file TXT
def txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


# Xử lý văn bản từ file PDF
def pdf(pdf_path):
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    text = ""
    start_page = 1
    end_page = num_pages
    yc = input("Do you want to read the entire PDF? [Y]/N: ")
    if yc == "N" or yc == "n":
        start_page = int(input("Enter start page (Indexing from 1): "))
        end_page = int(input(f"Enter end page (Less than {num_pages + 1}): "))
        if (start_page < 1 or start_page > num_pages or end_page < 1 or end_page > num_pages or start_page > end_page):
            print("Invalid page range")
            return ""
    for i in range(start_page - 1, end_page):
        page = reader.pages[i]
        page_text = page.extract_text()
        if page_text:  # Kiểm tra nếu trang có chứa văn bản
            text += page_text + " "
    return text


# Hàm tóm tắt văn bản
def summarize_text(text, num_sentences):
    if not text.strip():
        print("No text found to summarize.")
        return ""

    # Tách văn bản thành các câu
    sentences = [sentence.strip() for sentence in sent_tokenize(text)]

    # Áp dụng tiền xử lý cho từng câu và giữ cấu trúc câu
    processed_sentences = text_preprocessing(sentences)

    # Tính điểm cho từng câu dựa trên TF-IDF và lưu kèm vị trí câu ban đầu
    sentence_scores = []
    docs = [word_tokenize(sentence) for sentence in processed_sentences]
    for i, doc in enumerate(docs):
        score = 0
        for term in set(doc):  # Dùng set để tránh tính lặp từ
            score += tf_idf_z(term, doc, docs)
        sentence_scores.append((i, sentences[i], score))

    # Sắp xếp các câu dựa trên điểm số từ cao đến thấp
    sentence_scores.sort(key=lambda x: x[2], reverse=True)

    # Lấy số câu do người dùng nhập hoặc toàn bộ câu nếu ít hơn yêu cầu
    top_sentences = sorted(sentence_scores[:num_sentences], key=lambda x: x[0])

    # Ghép các câu theo thứ tự gốc
    summary = ' '.join(sentence for _, sentence, _ in top_sentences)

    return summary
