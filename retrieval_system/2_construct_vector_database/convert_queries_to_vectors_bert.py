import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from tqdm import tqdm

# 嵌入模型名稱
# model_name = "microsoft/codebert-base"
# model_name = "cssupport/mobilebert-sql-injection-detect"
model_name = "jackaduma/SecBERT"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 將模型名稱轉換為文件友好的格式（替換 - 和 / 為 _）
model_file_name = model_name.replace('-', '_').replace('/', '_')

# 指定主輸出目錄和模型專屬子目錄
base_output_dir = "D:/RAG/SQL_legality/dataset/vector"
model_output_dir = os.path.join(base_output_dir, model_file_name)
os.makedirs(model_output_dir, exist_ok=True)  # 如果目錄不存在則創建

# 讀取處理後的語句和標籤
with open("D:/RAG/SQL_legality/dataset/processed_queries.txt", "r", encoding="utf-8") as f:
    queries = [line.strip() for line in f.readlines()]

with open("D:/RAG/SQL_legality/dataset/processed_labels.txt", "r", encoding="utf-8") as f:
    labels = [int(line.strip()) for line in f.readlines()]

# 確保數據數量一致
assert len(queries) == len(labels), "語句和標籤數量不一致！"

# 定義嵌入函數
def get_codebert_embedding(query):
    """
    使用 CodeBERT 提取語句的嵌入向量。
    Args:
        query (str): 要嵌入的語句
    Returns:
        np.ndarray: 語句的嵌入向量
    """
    inputs = tokenizer(query, return_tensors="pt", padding=True, max_length = 512, truncation=True)
    with torch.no_grad():  # 禁用梯度計算
        outputs = model(**inputs)
    # 提取最後一層隱層輸出，取平均值作為嵌入
    hidden_states = outputs.last_hidden_state  # [batch_size, seq_length, hidden_dim]
    sentence_embedding = hidden_states.mean(dim=1).squeeze().numpy()  # [hidden_dim]
    return sentence_embedding

# 嵌入第一條語句以獲取模型的向量維度
print(f"使用模型 {model_name} 嵌入語句...")
sample_embedding = get_codebert_embedding(queries[0])
embedding_dimension = sample_embedding.shape[0]
print(f"模型 {model_name} 的嵌入向量維度為：{embedding_dimension}")

# 初始化進度條，將所有語句轉換為嵌入向量
embeddings = []
for query in tqdm(queries, desc="嵌入語句進度", unit="句"):
    embeddings.append(get_codebert_embedding(query))

# 將嵌入結果轉換為 NumPy 數組
embeddings = np.array(embeddings)
print(f"向量化完成，共生成 {len(embeddings)} 條語句的向量，維度為 {embeddings.shape[1]}")

# 正規化嵌入向量，準備使用 Cosine Similarity
normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
print("向量正規化完成，適用於餘弦相似度檢索。")

# 將向量存入 FAISS 檢索庫
faiss_index = faiss.IndexFlatIP(embedding_dimension)  # 使用內積索引，適配 Cosine Similarity
faiss_index.add(np.array(normalized_embeddings, dtype='float32'))
print(f"向量庫中已存入 {faiss_index.ntotal} 條向量。")

# 保存向量索引和標籤到模型專屬資料夾
faiss_index_file = os.path.join(model_output_dir, f"vector_index_{model_file_name}.faiss")
faiss.write_index(faiss_index, faiss_index_file)
print(f"索引已保存為 '{faiss_index_file}'。")

# 保存標籤和語句到文件
labels_file = os.path.join(model_output_dir, f"vector_labels_{model_file_name}.npy")
queries_file = os.path.join(model_output_dir, f"queries_{model_file_name}.npy")

np.save(labels_file, labels)  # 保存標籤
np.save(queries_file, queries)  # 保存原始語句

# 提示保存完成
print(f"標籤已保存為 '{labels_file}'。")
print(f"語句已保存為 '{queries_file}'。")
print(f"2.1 向量化語句資料庫完成，使用模型: {model_name}！")
