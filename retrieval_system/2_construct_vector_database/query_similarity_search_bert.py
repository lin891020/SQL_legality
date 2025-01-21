import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# 嵌入模型名稱
# model_name = "microsoft/codebert-base"
# model_name = "cssupport/mobilebert-sql-injection-detect"
model_name = "jackaduma/SecBERT"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 文件名轉換（替換 - 為 _）
model_file_name = model_name.replace('-', '_').replace('/', '_')

# 動態設置文件路徑
base_dir = "D:/RAG/SQL_legality/dataset/vector"
model_dir = os.path.join(base_dir, model_file_name)

index_file = os.path.join(model_dir, f"vector_index_{model_file_name}.faiss")
labels_file = os.path.join(model_dir, f"vector_labels_{model_file_name}.npy")
queries_file = os.path.join(model_dir, f"queries_{model_file_name}.npy")

# 加載向量索引和標籤
print(f"加載模型 {model_name} 的向量資料...")
index = faiss.read_index(index_file)
labels = np.load(labels_file)
queries = np.load(queries_file, allow_pickle=True)

print(f"向量索引中包含 {index.ntotal} 條語句。")

# 定義 CodeBERT 嵌入函數
def get_codebert_embedding(query):
    """
    使用 CodeBERT 提取語句的嵌入向量。
    Args:
        query (str): 輸入的 SQL 語句。
    Returns:
        np.ndarray: 語句的嵌入向量。
    """
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # 提取最後一層的隱層輸出，並取平均值
    hidden_states = outputs.last_hidden_state  # [batch_size, seq_length, hidden_dim]
    sentence_embedding = hidden_states.mean(dim=1).squeeze().numpy()  # [hidden_dim]
    return sentence_embedding

# 定義檢索函數
def retrieve_sql_legality(user_query, k=3):
    """
    檢索 SQL 語句的合法性，並返回詳細結果。
    
    Args:
        user_query (str): 用戶輸入的 SQL 語句。
        k (int): 返回的最相似語句數量。
    
    Returns:
        list: 包含檢索到的索引、標籤、距離和語句內容。
    """
    print(f"輸入語句: {user_query}")
    
    # 嵌入用戶輸入語句
    query_embedding = get_codebert_embedding(user_query)
    
    # 檢索向量索引
    distances, indices = index.search(np.array([query_embedding], dtype="float32"), k)
    
    # 打印檢索結果
    print(f"前 {k} 個最相似的語句：")
    results = []
    for i, idx in enumerate(indices[0]):
        result = {
            "index": int(idx),
            "label": int(labels[idx]),
            "distance": float(distances[0][i]),
            "query": queries[idx]  # 加入原始語句
        }
        results.append(result)
        print(f"- 相似語句: {result['query']}, 標籤: {result['label']}, 距離: {result['distance']}")
    
    return results

# 測試檢索功能
user_query = "SELECT * FROM users WHERE id = 1;"
result = retrieve_sql_legality(user_query, k=3)

# 打印檢索結果
print("\n詳細信息：")
for i, res in enumerate(result, start=1):
    print(f"第 {i} 筆：")
    print(f"  - 索引: {res['index']}")
    print(f"  - 標籤: {res['label']}")
    print(f"  - 距離: {res['distance']}")
    print(f"  - 原始語句: {res['query']}")

# 提示檢索完成
print(f"2.2 SQL 語句檢索完成，使用模型: {model_name}！")
