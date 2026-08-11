import time
from api_client import call_predict, call_explain_llm
from groundedness import score_groundedness
from io_utils import load_samples, save_results

REQUEST_DELAY_SECONDS = 1.5

def evaluate_one(sample: dict):
    text = sample["text"]
    try:
        pred = call_predict(text)
        llm = call_explain_llm(text)
        print(llm.get("signals", {}))
        groundedness = score_groundedness(llm["bullets"], llm.get("signals", {}))
        return {
            "id": sample["id"],
            "generated_type_true": sample["generated_type"],
            "text_preview": text[:80] + ("..." if len(text) > 80 else ""),
            "predicted_class": pred["predicted_class"],
            "bullets": " | ".join(llm["bullets"]),
            "known_words_referenced": groundedness["known_words_referenced"],
            "known_words_total": groundedness["known_words_total"],
            "numeric_claims_matched": groundedness["numeric_claims_matched"],
            "status": "ok",
            "example_note": "",
        }
    except Exception as e:
        return {
            "id": sample["id"], 
            "generated_type_true": sample["generated_type"],
            "text_preview": text[:80], 
            "predicted_class": "ERROR",
            "bullets": "", 
            "known_words_referenced": "", 
            "known_words_total": "",
            "numeric_claims_matched": "", 
            "status": f"error: {e}", 
            "example_note": "",
        }

def main():
    samples = load_samples()

    print(f"Đang đánh giá {len(samples)} mẫu...")
    rows = []
    for i, sample in enumerate(samples, 1):
        print(sample.keys())
        print(f"[{i}/{len(samples)}] id = {sample['id']} generated_type = {sample['generated_type']}")
        rows.append(evaluate_one(sample))
        time.sleep(REQUEST_DELAY_SECONDS)

    save_results(rows)
    print("\nXong. Mở eval_results.csv — chọn vài dòng làm ví dụ minh họa.")

if __name__ == "__main__":
    main()