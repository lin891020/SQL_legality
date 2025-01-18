import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm  # 用於進度條顯示

# 嵌入模型名稱
model_name = 'paraphrase-MiniLM-L6-v2' 
# model_name = 'paraphrase-mpnet-base-v2'  # 替換為其他嵌入模型名稱進行測試
model = SentenceTransformer(model_name)
print(f"正在使用 {model_name} 模型進行向量化...")

# 將模型名稱轉換為文件友好的格式（替換 - 為 _）
model_file_name = model_name.replace('-', '_').replace('/', '_')

# 指定主輸出目錄和模型專屬子目錄
base_output_dir = "D:/RAG/SQL_legality/dataset/vector"
model_output_dir = os.path.join(base_output_dir, model_file_name)
os.makedirs(model_output_dir, exist_ok=True)  # 如果目錄不存在則創建

# 讀取處理後的語句和標籤
with open("D:/RAG/SQL_legality/dataset/processed_queries.txt", "r", encoding="utf-8") as f:
    queries = [line.strip() for line in f.readlines()]

with open("D:/RAG/SQL_legality/dataset/processed_labels.txt", "r", encoding="utf-8") as f:
    labels = [int(line.strip()) for line in f.readlines()]

# 確保數據數量一致
assert len(queries) == len(labels), "語句和標籤數量不一致！"

# 初始化嵌入進度條
print(f"使用模型 {model_name} 嵌入語句...")

# 嵌入第一條語句以獲取模型的向量維度
sample_embedding = model.encode([queries[0]])  # 嵌入單條語句
embedding_dimension = sample_embedding.shape[1]  # 獲取嵌入向量的維度
print(f"模型 {model_name} 的嵌入向量維度為：{embedding_dimension}")

# 使用 tqdm 對所有語句進行嵌入，並顯示進度
embeddings = []
for query in tqdm(queries, desc="嵌入語句進度", unit="句"):
    embedding = model.encode([query])
    embeddings.append(embedding[0])

# 將嵌入結果轉換為 NumPy 數組
embeddings = np.array(embeddings)
print(f"向量化完成，共生成 {len(embeddings)} 條語句的向量。")

# 正規化嵌入向量，準備使用 Cosine Similarity
normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
print("向量正規化完成，適用於餘弦相似度檢索。")

# 使用 tqdm 為 FAISS 構建索引並添加進度條
print("構建向量索引並添加向量到索引庫中...")
faiss_index = faiss.IndexFlatIP(embedding_dimension)  # 使用內積索引，適配 Cosine Similarity
for i in tqdm(range(0, len(normalized_embeddings), 1000), desc="向量添加進度", unit="批"):
    faiss_index.add(np.array(normalized_embeddings[i:i+1000], dtype='float32'))  # 批量添加向量

print(f"向量庫中已存入 {faiss_index.ntotal} 條向量。")

# 保存向量索引和標籤到模型專屬資料夾
faiss_index_file = os.path.join(model_output_dir, f"vector_index_{model_file_name}.faiss")
faiss.write_index(faiss_index, faiss_index_file)
print(f"索引已保存為 '{faiss_index_file}'。")

# 保存標籤和語句到文件
labels_file = os.path.join(model_output_dir, f"vector_labels_{model_file_name}.npy")
queries_file = os.path.join(model_output_dir, f"queries_{model_file_name}.npy")

np.save(labels_file, labels)  # 保存標籤
np.save(queries_file, queries)  # 保存原始語句

# 提示保存完成
print(f"標籤已保存為 '{labels_file}'。")
print(f"語句已保存為 '{queries_file}'。")
print(f"2.1 向量化語句資料庫完成，使用模型: {model_name}！")
