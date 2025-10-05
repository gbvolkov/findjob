import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
from agents.retrievers.chunker import chunk_text

def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def embed_texts(texts: List[str], tokenizer, model) -> torch.Tensor:
    batch = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**batch)
    embeddings = average_pool(outputs.last_hidden_state, batch["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def extract_relevant_passages(question: str, passages: List[str], tokenizer, model, top_k=3) -> List[str]:
    try:
        # Форматируем вход с префиксами, как рекомендуется
        query_text = f"query: {question}"
        passage_texts = [f"passage: {p}" for p in passages]

        # Получаем эмбеддинги
        query_emb = embed_texts([query_text], tokenizer, model)
        passages_emb = embed_texts(passage_texts, tokenizer, model)

        # Считаем похожесть
        scores = (query_emb @ passages_emb.T).squeeze(0)  # Косинусное сходство после нормализации

        # Получаем топ k по убыванию
        _, top_indices = torch.topk(scores, k=min(top_k, len(passages)))

        # Обработка ситуации с выходом за пределы индекса
        relevant_passages = []
        for idx in top_indices.tolist():
            try:
                relevant_passages.append(passages[idx])
            except IndexError:
                # Логируем и пропускаем индекс вне диапазона
                print(f"Warning: selected index {idx} out of range for passages list")
                continue

        return relevant_passages

    except Exception as e:
        print(f"Ошибка при извлечении релевантной информации: {e}")
        return []

if __name__ == "__main__":
    import fitz
    doc = fitz.open("data/test.pdf")
    text = ''
    for page in doc:
        text += page.get_text()

    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")

    # Пример вопроса и длинного текста, разбитого на отрывки (passages)
    question = "Какие должностные обязанности стюартов?"
    passages = chunk_text(text)

    relevant = extract_relevant_passages(question, passages, tokenizer, model, top_k=2)
    print("Релевантные отрывки:")
    for passage in relevant:
        print("-", passage)
