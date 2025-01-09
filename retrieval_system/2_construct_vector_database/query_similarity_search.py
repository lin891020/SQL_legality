import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 配置模型名稱
model_name = 'paraphrase-MiniLM-L6-v2'
# model_name = 'paraphrase-mpnet-base-v2'  # 替換為其他嵌入模型名稱進行測試
model = SentenceTransformer(model_name)
print(f"正在使用 {model_name} 模型進行檢索...")

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
    query_embedding = model.encode([user_query])

    # 查詢向量正規化
    normalized_query = query_embedding / np.linalg.norm(query_embedding)
    
    # 檢索向量索引
    distances, indices = index.search(np.array(normalized_query, dtype="float32"), k)
    
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

        # 檢測距離是否符合餘弦相似度範圍
        if not -1 <= result["distance"] <= 1:
            print(f"警告：距離 {result['distance']} 超出餘弦相似度範圍（-1 到 1）。")
    
    return results

# 測試檢索功能
user_query = "SELECT * FROM users WHERE id = 1;"
result = retrieve_sql_legality(user_query, k=3)

print("\n詳細信息：")
for i, res in enumerate(result, start=1):
    print(f"第 {i} 筆：")
    print(f"  - 索引: {res['index']}")
    print(f"  - 標籤: {res['label']}")
    print(f"  - 距離: {res['distance']}")
    print(f"  - 原始語句: {res['query']}")

# 提示檢索完成
print(f"2.2 SQL 語句檢索完成，使用模型: {model_name}！")
