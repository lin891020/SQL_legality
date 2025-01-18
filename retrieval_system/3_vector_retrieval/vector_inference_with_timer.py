import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time

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
    start_time = time.perf_counter()
    print(f"\n輸入語句: {user_query}\n")
    
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
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        "input_query": user_query,
        "legality": legality,
        "reason": f"Scores: {{'legal': {scores[0]:.4f}, 'illegal': {scores[1]:.4f}}}",
        "details": valid_results,
        "inference_time_ms": inference_time_ms
    }

# 設置 k 值
k_value = 1

# 循環輸入查詢語句
while True:
    user_query = input("請輸入SQL語句 (或輸入 'exit' 結束): ")
    if user_query.lower() == 'exit':
        print("結束程序。")
        break

    result = classify_sql_legality(user_query, k = k_value)

    # 輸出結果
    print("\n判斷結果：")
    print(f"輸入語句: {user_query}")
    print(f"語句合法性：{result['legality']}")
    print(f"原因：{result['reason']}")
    print(f"推論時間: {result['inference_time_ms']:.4f} ms")
    print(f"3.2 SQL 語句合法性判斷完成，使用模型: {model_name}！")
