import os
import csv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

# 配置模型名稱
# model_name = 'paraphrase-MiniLM-L6-v2'
model_name = 'paraphrase-mpnet-base-v2'  # 替換為其他嵌入模型名稱進行測試
model = SentenceTransformer(model_name)
print(f"正在使用 {model_name} 模型進行分類...")

# 文件名轉換（替換 - 為 _）
model_file_name = model_name.replace('-', '_').replace('/', '_')

# 動態設置主資料夾路徑
base_output_dir = "D:/RAG/SQL_legality/result"
model_output_dir = os.path.join(base_output_dir, model_file_name)

# 確保模型對應的輸出資料夾存在
os.makedirs(model_output_dir, exist_ok=True)

# 動態設置輸出檔案路徑
output_file = os.path.join(model_output_dir, f"testing_results_{model_file_name}.csv")
confusion_matrix_file = os.path.join(model_output_dir, f"confusion_matrix_{model_file_name}.png")

# 加載向量索引和標籤
base_vector_dir = "D:/RAG/SQL_legality/dataset/vector"
model_vector_dir = os.path.join(base_vector_dir, model_file_name)
index_file = os.path.join(model_vector_dir, f"vector_index_{model_file_name}.faiss")
labels_file = os.path.join(model_vector_dir, f"vector_labels_{model_file_name}.npy")
queries_file = os.path.join(model_vector_dir, f"queries_{model_file_name}.npy")

print(f"加載模型 {model_name} 的向量資料...")
index = faiss.read_index(index_file)
labels = np.load(labels_file)
queries = np.load(queries_file, allow_pickle=True)
print(f"向量索引中包含 {index.ntotal} 條語句。")

# 定義 SQL 合法性分類函數
def classify_sql_legality(user_query, k=3, distance_threshold=0.8, epsilon=1e-6):
    query_embedding = model.encode([user_query])

    # 查詢向量正規化
    normalized_query = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    # 檢索向量索引
    distances, indices = index.search(np.array(normalized_query, dtype="float32"), k)

    # 檢測有效的檢索結果
    valid_results = [
        {"index": int(idx), "label": int(labels[idx]), "distance": float(dist),
         "weight": 1 / (float(dist) + epsilon), "query": queries[idx]}
        for idx, dist in zip(indices[0], distances[0]) if dist >= distance_threshold
    ]

    # 如果無結果，強制返回最近的 K 個語句
    if not valid_results:
        valid_results = [
            {"index": int(idx), "label": int(labels[idx]), "distance": float(dist),
             "weight": 1 / (float(dist) + epsilon), "query": queries[idx]}
            for idx, dist in zip(indices[0], distances[0])
        ]

    # 計算加權分數
    weighted_scores = {0: 0, 1: 0}
    for res in valid_results:
        weighted_scores[res["label"]] += res["weight"]

    legality = "legal" if weighted_scores[0] > weighted_scores[1] else "illegal"
    return {
        "input_query": user_query,
        "legality": legality,
        "reason": f"Weighted scores: {weighted_scores}",
        "details": valid_results
    }

# 讀取測試數據
input_file = "D:/RAG/SQL_legality/dataset/testingdata.csv"
print(f"正在從 {input_file} 讀取測試數據...")
results = []
true_labels = []
predicted_labels = []

data_count = 0
with open(input_file, "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    data = list(reader)
    data_count = len(data)
    print(f"共讀取到 {data_count} 筆測試數據。")

# 處理每筆數據
for row in tqdm(data, desc="處理測試數據進度", unit="筆"):
    user_query = row["Query"]
    true_label = row["Label"]

    # 判斷語句合法性
    result = classify_sql_legality(user_query, k=5, distance_threshold=0.8)

    # 保存結果
    results.append({
        "query": user_query,
        "true_label": true_label,
        "predicted_label": result["legality"],
        "reason": result["reason"]
    })

    true_labels.append(int(true_label))
    predicted_labels.append(0 if result["legality"] == "legal" else 1)

# 寫入結果到 CSV
print(f"正在將結果寫入到 {output_file}...")
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["query", "true_label", "predicted_label", "reason"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

print(f"結果已保存到 {output_file}！")

# 繪製混淆矩陣
print("繪製混淆矩陣...")
cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["legal", "illegal"])
disp.plot(cmap=plt.cm.Blues, colorbar=True, values_format='.0f')

# 設置標題與標籤
plt.title(f"Confusion Matrix_{model_file_name}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# 保存混淆矩陣圖像
plt.savefig(confusion_matrix_file)
plt.show()

print(f"混淆矩陣已保存為：{confusion_matrix_file}")
