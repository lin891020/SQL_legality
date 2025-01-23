from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# 定義模型名稱清單
model_names = [
    'paraphrase-MiniLM-L6-v2', 
    'paraphrase-mpnet-base-v2', 
    'microsoft/codebert-base', 
    'cssupport/mobilebert-sql-injection-detect',
    'jackaduma/SecBERT'
]

# 定義檢查函數
def print_model_info(model_name):
    print(f"模型名稱: {model_name}")
    
    # SentenceTransformer 模型
    if 'MiniLM' in model_name or 'mpnet' in model_name:
        model = SentenceTransformer(model_name)
        tokenizer = model.tokenizer
        print(f"- 嵌入維度 (Embedding Dimension): {model.get_sentence_embedding_dimension()}")
    else:
        # Transformers 模型
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print(f"- 嵌入維度 (Embedding Dimension): {model.config.hidden_size}")
    
    # 最大輸入長度
    max_length = tokenizer.model_max_length
    print(f"- 最大輸入長度 (Max Token Length): {max_length}")
    
    # 模型參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"- 模型總參數量 (Total Parameters): {total_params}")
    print(f"- 可訓練參數量 (Trainable Parameters): {trainable_params}")
    
    # Token 化示例
    sample_text = "SELECT * FROM users WHERE id = '1';"
    tokens = tokenizer.tokenize(sample_text)
    print(f"- 示例 SQL 查詢: {sample_text}")
    print(f"- Token 數量: {len(tokens)}")
    print(f"- Token 化結果: {tokens}")
    print("=" * 50)

# 執行檢查
for model_name in model_names:
    print_model_info(model_name)
