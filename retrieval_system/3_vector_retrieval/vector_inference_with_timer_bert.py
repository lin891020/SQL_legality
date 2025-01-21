from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# 嵌入模型名稱
# model_name = "microsoft/codebert-base"
# model_name = "cssupport/mobilebert-sql-injection-detect"
model_name = "jackaduma/SecBERT"

print(f"模型名稱: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 測試語句
sentence1 = "SELECT * FROM users WHERE id = '1';"
sentence2 = "SELECT username, password FROM users WHERE role = 'admin';"

# 定義嵌入函數
def get_embedding(sentence):
    # Token 化
    inputs = tokenizer(
        sentence,
        return_tensors="pt",  # 返回 PyTorch 張量
        padding="max_length", # 補零至最大長度
        truncation=True,      # 截斷長輸入
        max_length=512        # 最大輸入長度
    )
    print(f"查詢語句: {sentence}")

    # 格式化 Input IDs 為對齊格式，並以頓號分隔
    input_ids = inputs['input_ids'][0, :20].tolist()
    formatted_input_ids = ", ".join([f"{id_:>5}" for id_ in input_ids])  # 每個數字占 5 個空間，頓號分隔
    print(f"- Input IDs: {formatted_input_ids}")

    # 嵌入生成
    with torch.no_grad():
        outputs = model(**inputs)

    # 提取最後隱藏層，並進行平均池化
    last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_length, embedding_dim]
    sentence_embedding = last_hidden_state.mean(dim=1).squeeze().numpy()  # 平均池化為句子嵌入
    formatted_embedding = ", ".join([f"{x:.4f}" for x in sentence_embedding[:5]])  # 格式化前 5 個嵌入值
    print(f"- 句子嵌入向量 (維度: {len(sentence_embedding)}): {formatted_embedding} ...")

    return sentence_embedding

# 獲取嵌入向量
embedding1 = get_embedding(sentence1)
embedding2 = get_embedding(sentence2)

# 正規化嵌入向量
normalized_embedding1 = embedding1 / np.linalg.norm(embedding1, keepdims=True)
normalized_embedding2 = embedding2 / np.linalg.norm(embedding2, keepdims=True)

# 打印正規化後部分嵌入
formatted_normalized_embedding1 = ", ".join([f"{x:.4f}" for x in normalized_embedding1[:5]])
formatted_normalized_embedding2 = ", ".join([f"{x:.4f}" for x in normalized_embedding2[:5]])
print("\n正規化後嵌入向量（部分）：")
print(f"- 語句 1 嵌入向量: {formatted_normalized_embedding1} ...")
print(f"- 語句 2 嵌入向量: {formatted_normalized_embedding2} ...")

# 計算餘弦距離
cosine_similarity = np.dot(normalized_embedding1, normalized_embedding2)
distance = 1 - cosine_similarity  # 餘弦距離 = 1 - 餘弦相似度
print(f"\n餘弦相似度: {cosine_similarity:.4f}")
print(f"餘弦距離: {distance:.4f}")
