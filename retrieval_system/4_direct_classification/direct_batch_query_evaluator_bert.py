import os
import csv
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# 配置模型名稱
model_name = "microsoft/codebert-base"
# model_name = "cssupport/mobilebert-sql-injection-detect" 
# model_name = "jackaduma/SecBERT" # 使用 Hugging Face 的 SecBERT 模型

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)



print(f"正在使用 {model_name} 模型進行 SQL Injection 檢測...")

# 定義 SQL 合法性分類函數
def classify_sql_legality(user_query):
    """
    使用 MobileBERT 模型判斷 SQL 語句合法性。
    Args:
        user_query (str): 輸入的 SQL 語句。
    Returns:
        dict: 包含判斷結果和詳細信息的字典。
    """
    # 將語句進行編碼，並轉為 PyTorch 張量
    inputs = tokenizer(user_query, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 獲取 logits，計算分類概率 
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()
    
    # 判斷分類結果
    predicted_label = np.argmax(probabilities)
    label_map = {0: "legal", 1: "illegal"}  # 0 表示合法，1 表示非法
    
    return {
        "input_query": user_query,
        "legality": label_map[predicted_label],
        "probabilities": {label_map[0]: round(probabilities[0], 4), label_map[1]: round(probabilities[1], 4)}
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
    result = classify_sql_legality(user_query)

    # 定義映射
    mapped_label = {"legal": 0, "illegal": 1}

    results.append({
        "query": user_query,
        "true_label": int(true_label),  # 確保 true_label 為數字
        "predicted_label": mapped_label[result["legality"]],  # 轉換 predicted_label
        "probabilities": result["probabilities"]
    })
    true_labels.append(int(true_label))
    predicted_labels.append(mapped_label[result["legality"]])

# 過濾錯誤預測
wrong_predictions = [
    result for result in results if result["true_label"] != result["predicted_label"]
]

# 動態設置主資料夾路徑
base_output_dir = "D:/RAG/SQL_legality/result/direct"
model_output_dir = os.path.join(base_output_dir, model_name.replace('-', '_').replace('/', '_'))

# 確保模型對應的輸出資料夾存在
os.makedirs(model_output_dir, exist_ok=True)

# 動態設置輸出檔案路徑
output_file = os.path.join(model_output_dir, f"testing_results_{model_name.replace('-', '_').replace('/', '_')}.csv")
wrong_output_file = os.path.join(model_output_dir, f"testing_results_wrong_{model_name.replace('-', '_').replace('/', '_')}.csv")
confusion_matrix_file = os.path.join(model_output_dir, f"confusion_matrix_{model_name.replace('-', '_').replace('/', '_')}.png")

# 寫入結果到 CSV
print(f"正在將結果寫入到 {output_file}...")
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["query", "true_label", "predicted_label", "probabilities"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

print(f"結果已保存到 {output_file}！")

# 寫入錯誤預測結果到 CSV
print(f"正在將錯誤預測結果寫入到 {wrong_output_file}...")
with open(wrong_output_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["query", "true_label", "predicted_label", "probabilities"]
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
disp.plot(cmap=plt.cm.Blues, colorbar=False, values_format='.0f')

# 設置標題與標籤
plt.title(f"Confusion Matrix_{model_name.replace('-', '_').replace('/', '_')}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# 保存混淆矩陣圖像
plt.savefig(confusion_matrix_file)
plt.show()

print(f"混淆矩陣已保存為：{confusion_matrix_file}")
