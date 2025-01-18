import os
import csv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings("ignore")

# 配置模型名稱
# model_name = 'paraphrase-MiniLM-L6-v2'
model_name = 'paraphrase-mpnet-base-v2'  # 替換為其他嵌入模型名稱進行測試
model = SentenceTransformer(model_name)

print(f"正在使用 {model_name} 模型進行分類...")

# 文件名轉換（替換 - 為 _）
model_file_name = model_name.replace('-', '_').replace('/', '_')

# 動態設置主資料夾路徑
base_output_dir = "D:/RAG/SQL_legality/result/retrieval"

# 確保模型對應的輸出資料夾存在
os.makedirs(base_output_dir, exist_ok=True)

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

# Initialize a list to store all results
all_results = []

for k_value in range(1, 6):
    print(f"正在處理 k = {k_value} 的結果...")

    # 動態設置主資料夾路徑
    model_output_dir = os.path.join(base_output_dir, model_file_name, "k = " + str(k_value))

    # 確保模型對應的輸出資料夾存在
    os.makedirs(model_output_dir, exist_ok=True)

    # 動態設置輸出檔案路徑
    output_file = os.path.join(model_output_dir, f"testing_results_{model_file_name} - k = {k_value}.csv")
    wrong_output_file = os.path.join(model_output_dir, f"testing_results_wrong_{model_file_name} - k = {k_value}.csv")
    confusion_matrix_file = os.path.join(model_output_dir, f"confusion_matrix_{model_file_name} - k = {k_value}.png")

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

    start_time = time.time()

    # 處理每筆數據
    for row in tqdm(data, desc="處理測試數據進度", unit="筆"):
        user_query = row["Query"]
        true_label = row["Label"]

        # 判斷語句合法性
        result = classify_sql_legality(user_query, k=k_value)

        # 定義映射
        mapped_label = {"legal": 0, "illegal": 1}  

        results.append({
            "query": user_query,
            "true_label": int(true_label),  # 確保 true_label 為數字
            "predicted_label": mapped_label[result["legality"]],  # 轉換 predicted_label
            "reason": result["reason"]  # 已經處理小數位數
        })
        true_labels.append(int(true_label))
        predicted_labels.append(mapped_label[result["legality"]])

    end_time = time.time()
    total_time = end_time - start_time
    average_time = (total_time / data_count) * 1000  # in milliseconds

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

    # Append the results to the all_results list
    all_results.append({
        "k": k_value,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "total_time": total_time,
        "average_time": average_time
    })

    # 打印結果
    print(f"Accuracy: {accuracy:.3f}%")
    print(f"Precision: {precision:.3f}%")
    print(f"Recall: {recall:.3f}%")
    print(f"Total Time: {int(total_time // 60)}min {int(total_time % 60)}sec")
    print(f"Average Time: {average_time:.2f}ms")

    # 繪製混淆矩陣
    print("繪製混淆矩陣...")
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["legal", "illegal"])
    disp.plot(cmap=plt.cm.Blues, colorbar=False, values_format='.0f')

    # 設置標題與標籤
    plt.title(f"retrieval system: {model_name} - k = {k_value}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # 保存混淆矩陣圖像
    plt.savefig(confusion_matrix_file)

    print(f"混淆矩陣已保存為：{confusion_matrix_file}")

# Save all results to a single file
summary_file = os.path.join(base_output_dir, model_file_name, "summary_results.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    for result in all_results:
        f.write(f"k = {result['k']}\n")
        f.write(f"Accuracy: {result['accuracy']:.3f}%\n")
        f.write(f"Precision: {result['precision']:.3f}%\n")
        f.write(f"Recall: {result['recall']:.3f}%\n")
        f.write(f"Total Time: {int(result['total_time'] // 60)}min {int(result['total_time'] % 60)}sec\n")
        f.write(f"Average Time: {result['average_time']:.2f}ms\n")
        f.write("\n")

print(f"所有結果已保存到 {summary_file}！")
