import os
import csv
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from tqdm import tqdm

# 配置模型名稱
model_name = 'microsoft/codebert-base'  # 使用 Hugging Face 的 CodeBERT 模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print(f"正在使用 {model_name} 模型進行分類...")

# 文件名轉換（替換 - 為 _）
model_file_name = model_name.replace('-', '_').replace('/', '_')

# 動態設置主資料夾路徑
base_output_dir = "D:/RAG/SQL_legality/result/retrieval"
model_output_dir = os.path.join(base_output_dir, model_file_name)

# 確保模型對應的輸出資料夾存在
os.makedirs(model_output_dir, exist_ok=True)

# 動態設置輸出檔案路徑
output_file = os.path.join(model_output_dir, f"testing_results_{model_file_name}.csv")
wrong_output_file = os.path.join(model_output_dir, f"testing_results_wrong_{model_file_name}.csv")
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

def classify_sql_legality(user_query, k=3, distance_threshold=0.8, epsilon=1e-6):
    query_embedding = get_codebert_embedding(user_query)
    
    # 查詢向量正規化
    normalized_query = query_embedding / np.linalg.norm(query_embedding, keepdims=True)
    
    # 檢索向量索引
    distances, indices = index.search(np.array([normalized_query], dtype="float32"), k)

    valid_results = [
        {
            "index": int(idx), 
            "label": int(labels[idx]), 
            "distance": round(float(dist), 4),
            "weight": round(1 / (float(dist) + epsilon), 4), 
            "query": queries[idx]
        }
        for idx, dist in zip(indices[0], distances[0]) if dist >= distance_threshold
    ]

    # 如果仍沒有找到符合條件的結果，強制使用最近的 K 個語句
    if not valid_results:
        valid_results = [
            {
                "index": int(idx),
                "label": int(labels[idx]),
                "distance": round(float(dist), 4),  # 將距離限制為4位小數
                "weight": round(1 / (float(dist) + epsilon), 4),  # 將權重限制為4位小數
                "query": queries[idx]
            }
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
        "reason": f"Weighted scores: {{0: {weighted_scores[0]:.4f}, 1: {weighted_scores[1]:.4f}}}",
        "details": valid_results
    }

    return result

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

    # 定義映射
    mapped_label = {"legal合法語句": 0, "illegal非法語句": 1}

  

    results.append({
        "query": user_query,
        "true_label": int(true_label),  # 確保 true_label 為數字
        "predicted_label": mapped_label[result["legality"]],  # 轉換 predicted_label
        "reason": result["reason"]  # 已經處理小數位數
    })
    true_labels.append(int(true_label))
    predicted_labels.append(mapped_label[result["legality"]])

# 過濾錯誤預測
wrong_predictions = [
    result for result in results if result["true_label"] != result["predicted_label"]
]

# 寫入結果到 CSV
print(f"正在將結果寫入到 {output_file}...")
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["query", "true_label", "predicted_label", "reason"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

print(f"結果已保存到 {output_file}！")

# 寫入錯誤預測結果到 CSV
print(f"正在將錯誤預測結果寫入到 {wrong_output_file}...")
with open(wrong_output_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["query", "true_label", "predicted_label", "reason"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(wrong_predictions)

print(f"錯誤預測結果已保存到 {wrong_output_file}！")

# 計算 Accuracy, Precision, Recall
accuracy = accuracy_score(true_labels, predicted_labels) * 100
precision = precision_score(true_labels, predicted_labels) * 100
recall = recall_score(true_labels, predicted_labels) * 100

# 打印結果
print(f"Accuracy: {accuracy:.3f}%")
print(f"Precision: {precision:.3f}%")
print(f"Recall: {recall:.3f}%")

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
