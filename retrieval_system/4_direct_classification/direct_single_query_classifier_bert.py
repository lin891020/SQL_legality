import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# 配置模型名稱
model_name = 'microsoft/codebert-base'  # 使用 Hugging Face 的 CodeBERT 模型
# model_name = "cssupport/mobilebert-sql-injection-detect"  # 使用 Hugging Face 的 MobileBERT SQL Injection Detection 模型

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
    inputs = tokenizer(user_query, return_tensors="pt", truncation=True, padding=True)
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

# 測試 SQL 判斷功能
user_query = "SELECT * FROM users WHERE id = 1;" # 合法語句    
# user_query = "select * from users where id = 1 %!<1 or 1 = 1 -- 1" # 非法語句
# user_query = "SELECT AVG ( Price ) FROM sail;" # 合法語句
# user_query = "SELECT hall, origin, becomingFROM wear WHERE hat IS NOT NULL;" # 非法語句

# 執行分類
result = classify_sql_legality(user_query)

# 輸出結果
print("\n分類結果:")
print(f"輸入語句: {result['input_query']}")
print(f"判斷結果: {result['legality']}")
print(f"分類概率: {result['probabilities']}")
print(f"4.1 單筆SQL語句檢索完成，使用模型: {model_name}！")
