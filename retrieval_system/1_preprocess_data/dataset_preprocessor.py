import pandas as pd

# 讀取 CSV 文件
file_path = "D:/RAG/SQL_legality/dataset/trainingdata.csv"  # 替換為您的 CSV 文件路徑
data = pd.read_csv(file_path)

# 查看數據結構（可選）
print(data.head())

# 提取 Query 和 Label 列
queries = data['Query'].tolist()  # 提取 SQL 語句
labels = data['Label'].tolist()  # 提取對應標籤

# 查看處理後的數據（可選）
print("SQL 語句數量:", len(queries))
print("第一條語句:", queries[0])
print("第一條標籤:", labels[0])

# 清洗語句
queries = [query.strip() for query in queries]

# 保存處理後的數據
output_queries_path = "D:/RAG/SQL_legality/dataset/processed_queries.txt"
output_labels_path = "D:/RAG/SQL_legality/dataset/processed_labels.txt"

with open(output_queries_path, "w", encoding="utf-8") as qf:
    qf.writelines("\n".join(queries))

with open(output_labels_path, "w", encoding="utf-8") as lf:
    lf.writelines("\n".join(map(str, labels)))

# 提示保存完成
print("處理完成，已保存語句和標籤！")
print("1.2 資料前處裡完成！")
