import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")

# 配置模型名稱
model_name = 'paraphrase-MiniLM-L6-v2'
# model_name = 'paraphrase-mpnet-base-v2'  # 替換為其他嵌入模型名稱進行測試

model = SentenceTransformer(model_name)
print(f"正在使用 {model_name} 模型進行分類...")
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

def classify_sql_legality(user_query, k, epsilon=1e-6):
    """
    判斷 SQL 語句的合法性，不受距離閾值限制。
    Args:
        user_query (str): 輸入的 SQL 語句。
        k (int): 返回的最相似語句數量。
        epsilon (float): 防止分母為 0 的小常數。
    Returns:
        dict: 包含判斷結果和詳細信息的字典。
    """
    print(f"輸入語句: {user_query}")
    
    # 嵌入用戶輸入語句
    query_embedding = model.encode([user_query])
    
    # 查詢向量正規化
    normalized_query = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    # 檢索向量索引
    distances, indices = index.search(np.array(normalized_query, dtype="float32"), k)

    # 計算分數
    scores = {0: 0, 1: 0}
    valid_results = []
    for idx, dist in zip(indices[0], distances[0]):
        scores[labels[idx]] += dist
        valid_results.append({
            "index": int(idx),
            "label": int(labels[idx]),
            "distance": round(float(dist), 4),
            "query": queries[idx]
        })
    
    # 判斷語句合法性
    legality = "legal" if scores[0] > scores[1] else "illegal"

    return {
        "input_query": user_query,
        "legality": legality,
        "reason": f"Scores: {{'legal': {scores[0]:.4f}, 'illegal': {scores[1]:.4f}}}",
        "details": valid_results
    }


# 測試 SQL 判斷功能
# user_query = "SELECT * FROM users WHERE id = 1;" # 合法語句    
user_query = "select * from users where id = 1 %!<1 or 1 = 1 -- 1" # 非法語句
# user_query = "SELECT AVG ( Price ) FROM sail;" # 合法語句
# user_query = "SELECT hall, origin, becomingFROM wear WHERE hat IS NOT NULL;" # 非法語句

# 設置 k 值
k_value = 5
result = classify_sql_legality(user_query, k = k_value)

# 輸出結果
print("\n判斷結果：")
print(f"輸入語句: {result['input_query']}")
print(f"k_value: {k_value}")
print(f"判斷結果: {result['legality']}")
print(f"分類概率: {result['reason']}")

# 詳細匹配資訊
print("\n詳細檢索資訊：")
for i, detail in enumerate(result["details"], start=1):
    print(f"第 {i} 筆：")
    print(f"  - 索引: {detail['index']}")
    print(f"  - 標籤: {detail['label']}")
    print(f"  - 距離: {detail['distance']:.4f}")
    print(f"  - 原始語句: {detail['query']}")
print(f"3.1 單筆SQL語句檢索完成，使用模型: {model_name}！")
