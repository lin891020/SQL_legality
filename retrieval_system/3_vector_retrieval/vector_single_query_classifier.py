import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

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

def classify_sql_legality(user_query, k=3, distance_threshold=0.8, epsilon=1e-6):
    print(f"輸入語句: {user_query}")
    query_embedding = model.encode([user_query])

    # 查詢向量正規化
    normalized_query = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    # 檢索向量索引
    distances, indices = index.search(np.array(normalized_query, dtype="float32"), k)
    print(f"尋找符合threshold > {distance_threshold:.2f}，最近的 {k} 個語句。")

    # 確保距離值符合餘弦相似度的範圍 [-1, 1]
    if not np.all((distances >= -1) & (distances <= 1)):
        print("檢測到距離值超出餘弦相似度範圍！")
        return None

    valid_results = [
        {"index": int(idx), "label": int(labels[idx]), "distance": float(dist),
         "weight": 1 / (float(dist) + epsilon), "query": queries[idx]}
        for idx, dist in zip(indices[0], distances[0]) if dist >= distance_threshold
    ]

    # 動態調整距離閾值直到找到至少一筆語句
    while not valid_results:
        distance_threshold -= 0.1
        print(f"未找到符合閾值的結果，降低相似度閾值到 {distance_threshold + 0.1:.2f} ~ {distance_threshold:.2f} ...")
        valid_results = [
            {"index": int(idx), "label": int(labels[idx]), "distance": float(dist),
             "weight": 1 / (float(dist) + epsilon), "query": queries[idx]}
            for idx, dist in zip(indices[0], distances[0]) if dist >= distance_threshold
        ]
        if distance_threshold < -1.0:
            print("達到最小閾值，停止調整。")
            break

    # 如果仍沒有找到符合條件的結果，強制使用最近的 K 個語句
    if not valid_results:
        print(f"未找到符合閾值的結果，返回距離最近的 {len(indices[0])} 個語句。")
        valid_results = [
            {"index": int(idx), "label": int(labels[idx]), "distance": float(dist),
             "weight": 1 / (float(dist) + epsilon), "query": queries[idx]}
            for idx, dist in zip(indices[0], distances[0])
        ]

    # 計算加權分數
    weighted_scores = {0: 0, 1: 0}
    for res in valid_results:
        weighted_scores[res["label"]] += res["weight"]

    legality = "legal合法語句" if weighted_scores[0] > weighted_scores[1] else "illegal非法語句"
    result = {
        "input_query": user_query,
        "legality": legality,
        "reason": f"{{legal: {weighted_scores[0]:.4f}, illegal: {weighted_scores[1]:.4f}}}",
        "details": valid_results
    }

    return result

# 測試 SQL 判斷功能
# user_query = "SELECT * FROM users WHERE id = 1;" # 合法語句    
# user_query = "select * from users where id = 1 %!<1 or 1 = 1 -- 1" # 非法語句
# user_query = "SELECT AVG ( Price ) FROM sail;" # 合法語句
# user_query = "SELECT hall, origin, becomingFROM wear WHERE hat IS NOT NULL;" # 非法語句
user_query = "SELECT * FROM earnings;" # 合法語句

result = classify_sql_legality(user_query, k=5, distance_threshold=0.8)


# 輸出結果
print("\n判斷結果：")
print(f"輸入語句: {user_query}")
print(f"判斷結果：{result['legality']}")
print(f"加權分數: {result['reason']}")
print("\n詳細信息：")
for i, res in enumerate(result['details'], start=1):
    print(f"第 {i} 筆：")
    print(f"  - 索引: {res['index']}")
    print(f"  - 標籤: {res['label']}")
    print(f"  - 距離: {res['distance']:.4f}")
    print(f"  - 原始語句: {res['query']}")
print(f"3.1 單筆SQL語句檢索完成，使用模型: {model_name}！")
