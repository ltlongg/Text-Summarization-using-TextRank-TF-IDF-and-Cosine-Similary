import numpy as np
import sys
import pandas as pd
import nltk
import re
from nltk import sent_tokenize, word_tokenize, pos_tag
import string
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import math as m
from collections import Counter
from bs4 import BeautifulSoup
import requests
from pypdf import PdfReader
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def url (url_path):
    #yêu cầu lấy file html
    file_html = requests.get(url_path)
    #chuyển thành file raw
    file_raw = BeautifulSoup(file_html.content,"html.parser")
    #tìm tất cả những p tag
    get_p_tags = file_raw.find_all("p")
    text = ""
    for p_tag in get_p_tags:
      #truy cập vào mỗi p tag và lấy text
      text += p_tag.get_text()
    return text
#####Xử lí txt
def txt (txt_path):
   with open(txt_path,'r') as f:
      text = f.read()
   return text
####Xử lí pdf
def pdf (pdf_path):
   reader = PdfReader(pdf_path)
   num_pages = len(reader.pages)
   text = ""
   start_page = 1
   end_page = num_pages
   yc = input("Do you want to read entire pdf ?[Y]/N  :  ")
   if yc == "N" or yc == "n":
     start_page = int(input("Enter start page (Indexing from 1): "))
     end_page = int(input(f"Enter end pages (Less than {num_pages + 1}): "))
     if (start_page < 1 or start_page > num_pages):
        print("\nInvalid start page given")
        sys.exit()
     if (end_page < 1 or end_page > num_pages):
        print("\nInvalid end page given")
        sys.exit()
     if (start_page > end_page):
        print("\nStart page cannot be greater than end page")
        sys.exit()
   for i in range(start_page - 1,end_page):
       page = reader.pages[i]
       text += page.extract_text() + " "
   return text
####Tiền xử lí văn bản
stop_words = stopwords.words("english")
print(stop_words)
#dùng để lemma từ bằng cách tìm loại từ của từ đó trong 1 câu và áp dụng lemma khi biết từ và loại từ
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
def text_preprocessing(text):
  sentence = text.lower()
  sentence = re.sub(r"\W"," ",sentence)
  sentence = re.sub(r"\s+"," ",sentence)
  sentence = [word for word in word_tokenize(sentence) if word not in stop_words]
  new_sentence = pos_tag(sentence)
  sentence_lemma = [WordNetLemmatizer().lemmatize(word[0],word_classification(word)) for word in new_sentence]
  # sentence_stem = [PorterStemmer().stem(word) for word in sentence_lemma]
  sentence = " ".join(sentence_lemma)
  return sentence
#####Vocab
def vocab_unique(text):
  punct = list(string.punctuation)
  vocab = []
  for sentence in sent_tokenize(text):
    for word in word_tokenize(sentence):
      if word not in vocab and word not in punct:
        vocab.append(word)
  return vocab
#####TF-IDF
def tf(text, vocab):
  sentences = sent_tokenize(text)
  #tách đoạn text thành từng câu
  tf_matrix = np.zeros((len(sentences), len(vocab)))
  #tạo ma trận tf_matrix có số hàng là số câu của đoạn text, số cột là số từ vựng của đoạn text đó
  for i, sentence in enumerate(sentences):
    #i là chỉ số, sentence là giá trị (tức là từng câu trong sentences)
    word_counts = Counter(word_tokenize(sentence))
    #tạo 1 dictionary có key là word và value là số lần xuất hiện của từ đó trong câu
    for j, word in enumerate(vocab):
      #tương tự trên
      tf_matrix[i, j] = word_counts[word] / len(word_tokenize(sentence))
      #tf_matrix[i,j] được tính bằng số lần xuất hiện của 1 word trong 1 câu cụ thể
      #ta tính lần lượt từng hàng, từ mỗi hàng tính giá trị từng cột (tức từng word)
  return tf_matrix

def idf(text, vocab):
  sentences = sent_tokenize(text)
  #tách đoạn text thành list các câu
  idf_matrix = np.zeros((len(vocab), 1))
  for i, word in enumerate(vocab):
    idf_matrix[i, 0] = m.log(len(sentences) / sum(1 for sentence in sentences if word in word_tokenize(sentence)))
    #idf_matrix[i,0] được tính là log(số câu của đoạn text chia cho số câu mà chứa word đó)
  return idf_matrix

def tf_idf(tf_matrix, idf_matrix):
  count_row = tf_matrix.shape[0]
  count_col = tf_matrix.shape[1]
  tf_idf_matrix = np.ones((count_row,count_col))
  for i in range (count_row):
    for j in range (count_col):
      tf_idf_matrix[i,j] = tf_matrix[i,j]*idf_matrix[j,0]
  return tf_idf_matrix
#####Coisine Similarity
def norm(tf_idf_matrix):
  norm_vector = np.ones((tf_idf_matrix.shape[0],1),dtype=float)
  for i in range (tf_idf_matrix.shape[0]):
    norm_vector[i] = m.sqrt(sum(pow(tf_idf_matrix[i,j],2) for j in range(tf_idf_matrix.shape[1])))
    #norm_vector[i] là norm của từng câu, được tính là căn của tổng bình phương tất cả các word trong 1 câu (các word là các word trong hàng i của ma trận)
  """
  norm_vector = np.sqrt(np.sum(tf_idf_matrix ** 2, axis=1, keepdims=True))
  """
  return norm_vector # Fixed: Corrected the indentation of the return statement

def coisne_similarity(tf_idf_matrix,vocab):
  norm_vector = norm(tf_idf_matrix)
  cosine_matrix = np.zeros((tf_idf_matrix.shape[0],tf_idf_matrix.shape[0]))
  for i in range (tf_idf_matrix.shape[0]):
    for j in range (i,tf_idf_matrix.shape[0]):
      # gán j như vậy để tránh việc tính [i,j] rồi lại phải quay lại tính [j,i]
      # ví dụ tính [1,2] thì sẽ k phải tính lại [2,1] nữa
      if (i==j):
        cosine_matrix[i,j] = 1
        #nếu 2 câu trùng nhau thì gán luôn bằng 1 đỡ phải tính
      else:
        cosine_matrix[i,j] = (tf_idf_matrix[i] @ tf_idf_matrix[j])/(norm_vector[i,0]*norm_vector[j,0])
          # được tính bằng tích vô hướng của câu i và câu j chia cho tích 2 norm của i và j
        cosine_matrix[j,i] = cosine_matrix[i,j]
        #trong 1 ma trận thì [i,j] và [j,i] đều được tính bởi 1 ct và các giá trị giống nhau nên gán luôn cho nhanh, đỡ phải tính lại
  return cosine_matrix
  # cosine matrix là ma trận biểu diễn sự tương đồng giữa các câu, nếu giá trị càng gần 1 thì càng giống nhau và ngược lại
  #####Text Rank
def textrank(cosine_matrix,esp,d):
  textrank_matrix = np.ones((cosine_matrix.shape[0],1))/cosine_matrix.shape[0]
  count = 100
  while (count > 0):
    new_textrank_matrix = (np.ones((cosine_matrix.shape[0],1))*(1-d))/(cosine_matrix.shape[0]) + d*(cosine_matrix @ textrank_matrix)
    #được tính bằng (1-d) + d*(tích vô hướng giữa cosine_matrix và textrank_matrix)
    # xem video này để hiểu hơn: https://www.youtube.com/watch?v=qtLk2x59Va8&t=616s
    delta = np.linalg.norm(new_textrank_matrix - textrank_matrix)
    # tính norm độ chênh lệch giữa new_textrank_matrix và textrank_matrix
    if delta <= esp:
      #nếu nhỏ hơn ngưỡng mình đề ra thì trả về ma trận vừa tính (tức là sự khác biệt giữa 2 thằng quá nhỏ, nếu tính thì cũng k chênh nhau nhiều nữa)
      return new_textrank_matrix
    textrank_matrix = new_textrank_matrix
    #gán để tiếp tục vòng lặp
    count -=1
  return new_textrank_matrix
  #nếu chưa tìm ra thì chạy hêt 100 vòng tự out và trả về new_textrank_matrix
def textrank_list_not_sort(new_textrank_matrix):
  textrank_list= []
  for i in range (new_textrank_matrix.shape[0]):
     textrank_list.append((i,new_textrank_matrix[i][0]))
  return textrank_list

def textrank_list_sorted(textrank_list,num_sentences):
    textrank_list.sort(key = lambda x:x[1], reverse=True)
    textrank_list_sort = sorted(textrank_list[:num_sentences],key = lambda x:x[0])
    return textrank_list_sort

def extract (textrank_list_sorted,text):
   sentences = []
   sentence = sent_tokenize(text)
   for i in range(len(textrank_list_sorted)):
      sentences.append(sentence[textrank_list_sorted[i][0]])
   return " ".join(sentences)
