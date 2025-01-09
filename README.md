# 工作流
| 步驟 | 內容 |
| --- | --- |
| **1. 資料準備** | 清理數據，提取 SQL 語句與標籤，構造結構化數據集。 |
| **2. 向量資料庫構建** | 使用 Sentence-BERT 將語句轉換為向量，建立 FAISS 向量檢索庫。 |
| **3. 合法性檢測** | 輸入語句後向量化，檢索最相關內容，返回標籤作為判斷結果。 |
| **4. 系統整合** | 整合檢索與判斷邏輯，封裝為 API 或終端應用，支持輸入和輸出。 |
| **5. 測試與優化** | 使用測試集檢測準確率，調整檢索參數和優化分類性能。 |
| **6. 部署與監控** | 部署到服務器，配置日誌和異常監控以確保穩定運行。 |

資料來源

執行程式

產生的數據及資料

## 1. 資料準備

### 1.1 清洗與格式化數據

D:\RAG\SQL_legality\dataset\**trainingdata.csv**

- 目標：
    - 從原始數據中提取 SQL 語句和標籤。
    - 刪除冗餘信息（如非 SQL 的長文本）。
- 步驟：
    - 篩選數據集中 Query 列，保留純 SQL 語句。
    - 確保每條語句對應一個標籤（合法 0 / 非法 1）。
- 結果：
    - 生成清理後的 CSV 文件或結構化數據集，格式如下：

```
Query, Label
SELECT * FROM users WHERE id = 1;, 0
select pg_sleep(5);, 1
```

### 1.2 切分語句

- **目標**：
    - 將每條語句分別向量化，保證可以高效檢索。
- **方法**：
    - 使用 NLP 工具（如 Sentence-BERT）將每條 SQL 語句轉換為嵌入向量。

D:\RAG\SQL_legality\retrieval_system\1_preprocess_data\**dataset_preprocessor.py**

產生 D:\RAG\SQL_legality\dataset\**processed_labels.txt**

   D:\RAG\SQL_legality\dataset\**processed_queries.txt**

## 2. 向量資料庫構建

### 2.1 建構向量化語句資料

D:\RAG\SQL_legality\retrieval_system\2_construct_vector_database\**convert_queries_to_vectors.py**

- 使用嵌入模型（如 `Sentence-BERT`）計算數據集中每一條語句的嵌入向量。
- 每條語句的向量是固定維度（`'word_embedding_dimension': 384,`）的數組，代表該語句的語義。

例如：

- SQL 語句：`"SELECT * FROM users WHERE id = 1;"`
- 嵌入向量（示例）：`[0.123, 0.456, -0.789, ..., 0.001]`。

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
sql_queries = [
    "SELECT * FROM users WHERE id = 1;",
    "select pg_sleep(5);"
]
	embeddings = model.encode(sql_queries)
```

Hint：

安裝 faiss時 使用

```bash
conda install -c pytorch faiss-cpu
```

因為windows底下並不支援faiss-gpu

若是在**Linux 或 WSL**。可以使用

```bash
conda install -c pytorch faiss-gpu
```

儲存在 D:\RAG\SQL_legality\dataset\vector\**{model_name}**

D:\RAG\SQL_legality\dataset\vector\**queries_{model_name}.npy**

D:\RAG\SQL_legality\dataset\vector\**vector_index_{model_name}.faiss**

D:\RAG\SQL_legality\dataset\vector\**vector_labels_{model_name}.npy**

- **`queries.npy`**：存儲原始語句內容，方便後續檢索時還原。
- **`vector_index.faiss`**：存儲所有嵌入向量的索引。
- **`vector_labels.npy`**：存儲每條語句對應的標籤（如合法性標記）。

### 2.2 SQL 語句檢索

D:\RAG\SQL_legality\retrieval_system\2_construct_vector_database\**query_similarity_search.py**

1. **嵌入查詢語句**：
- 使用與 2.1 相同的嵌入模型，將查詢語句轉換為嵌入向量。
1. **計算距離**：
- 將查詢嵌入向量與索引中的所有向量逐一計算距離。
- 使用 **L2 距離（歐幾里得距離）** 衡量語句之間的相似性。

```python
index = faiss.IndexFlatL2(dimension)
```

**L2 距離公式**

L2 距離的計算公式為：

$$
d(x,y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
$$

- $x$ 是查詢語句的嵌入向量。
- $y$ 是索引中某語句的嵌入向量。
- $n$ 是向量的維度（384 ）。

L2 距離越小，表示兩個向量越相似；反之，距離越大，表示相似度越低。 

- **距離越小**：
    - 表示查詢語句與索引語句的語義越相似。
- **距離越大**：
    - 表示查詢語句與索引語句的語義差異越大。

### **2.3 示例計算**

假設查詢向量和索引向量分別為：

$$
x=[0.1,0.2,0.3],y=[0.4,0.1,0.5]
$$

L2 距離計算如下：

$$
\text{距離} = \sqrt{(0.1-0.4)^2 + (0.2-0.1)^2 + (0.3-0.5)^2}
= \sqrt{(-0.3)^2 + (0.1)^2 + (-0.2)^2}
= \sqrt{0.09 + 0.01 + 0.04}
= \sqrt{0.14} \approx 0.374
$$

## 3. SQL 合法性檢測

### 3.1 單筆SQL語句檢索

D:\RAG\SQL_legality\retrieval_system\3_sql_legality_retrieval\**single_sql_legality_classifier.py**

### 3.2 單筆SQL輸入檢索推論

D:\RAG\SQL_legality\retrieval_system\3_sql_legality_retrieval\**inference_sql_legality.py**

```python
加載模型 microsoft/codebert-base 的向量資料...
向量索引中包含 98275 條語句。
輸入語句: select * from users where id = 1 %!<1 or 1 = 1 -- 1

判斷結果：
語句合法性：illegal
原因：基於檢索結果，標籤 {0: 0, 1: 2}

詳細信息：
第 1 筆：
  - 索引: 109
  - 標籤: 1
  - 距離: 0.0
  - 原始語句: select * from users where id = 1 %!<1 or 1 = 1 -- 1
第 2 筆：
  - 索引: 123
  - 標籤: 1
  - 距離: 1.0744056701660156
  - 原始語句: select * from users where id = 1 %!<@ or 1 = 1 -- 1
3. SQL 語句合法性判斷完成，使用模型: {model_name}！
```

### 3.3 多筆資料檢索

D:\RAG\SQL_legality\retrieval_system\3_sql_legality_retrieval\**testing_sql_legality_classifier.py**

### 3.4 判斷結果

## 4. 系統整合

### 4.1 判斷結果輸出

### 4.2 可選擴展

## 5. 測試與優化

### 5.1 測試數據集準備

### 5.2 測試模型性能

### 5.3 優化檢索效率

## 6. 部署與監控

### 6.1 部署到服務器

### 6.2 配置監控
