import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# モデル名
model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.2"
# デバイス設定（Apple シリコンの場合は MPS、それ以外は CPU）
device = "mps" if torch.backends.mps.is_available() else "cpu"
# モデルのロード（MPS では float32 が安定）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Apple シリコンでは float32 の方が安定
    device_map="auto"
).to(device)
# トークナイザのロード
tokenizer = AutoTokenizer.from_pretrained(model_name)
def ask_model(question: str, max_tokens: int = 200) -> str:
    """
    ユーザーの質問に対して AI の回答を生成する関数。
    Args:
        question (str): ユーザーの質問
        max_tokens (int): 生成する最大トークン数（デフォルトは200）
    Returns:
        str: AI の回答
    """
    # 入力をトークン化
    inputs = tokenizer(question, return_tensors="pt").to(device)
    # 生成実行
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,  # 最大トークン数を制限
        temperature=0.7,  # ランダム性を調整
        top_p=0.9,  # 上位確率の高いトークンを選択
        repetition_penalty=1.1  # 同じ単語の繰り返しを防ぐ
    )
    # 結果をデコード
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response
# 例: ユーザーの質問に対して AI の回答を表示
question = "廃用症候群とは何ですか？"
answer = ask_model(question)
print(f"質問: {question}")
print(f"回答: {answer}")