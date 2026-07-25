import torch
import numpy as np
from captum.attr import LayerIntegratedGradients

class AttributionExplainer:
    """
    Wrapper for calculating Integrated Gradients
    """

    def __init__(self, model, tokenizer, device, max_length=1024):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()

        self.lig = LayerIntegratedGradients(self._forward_func, self.model.bert.embeddings)

    def _forward_func(self, input_ids, attention_mask):
        logits = self.model(input_ids, attention_mask)
        return torch.softmax(logits, dim=1)

    def explain(self, text, target_label, n_steps=50):
        encoding = self.tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding='max_length', return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)

        with torch.no_grad():
            probs = self._forward_func(input_ids, attention_mask)
            pred_label = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0, pred_label].item()

        attributions, delta = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=target_label,
            return_convergence_delta=True,
            n_steps=n_steps
        )

        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / (torch.norm(attributions) + 1e-10)

        ids = input_ids[0].cpu().tolist()
        scores = attributions.cpu().detach().numpy()

        words, agg_scores = self._decode_and_aggregate(ids, scores)

        return {
            'tokens': words,
            'scores': agg_scores,
            'predicted_label': pred_label,
            'pred_prob': pred_prob,
            'convergence_delta': delta.item()
        }

    def _decode_and_aggregate(self, ids, scores):
        """
        Gộp token ids thành từ tiếng Việt thật, dựa vào 'Ġ' (byte-level BPE)
        đánh dấu điểm BẮT ĐẦU của 1 từ mới. Dùng tokenizer.decode() để
        tự động xử lý byte-level mapping ngược lại đúng UTF-8.
        """
        special_ids = set(self.tokenizer.all_special_ids)

        # Nhóm các id liên tiếp thành từng "cụm từ" dựa vào marker Ġ
        groups = []          # list[list[int]]
        current_group = []

        for tok_id in ids:
            if tok_id in special_ids:
                continue  # Bỏ [CLS], [SEP], [PAD]

            piece = self.tokenizer.convert_ids_to_tokens([tok_id])[0]
            starts_new_word = piece.startswith('Ġ') or piece.startswith('▁')

            if starts_new_word and current_group:
                groups.append(current_group)
                current_group = [tok_id]
            else:
                current_group.append(tok_id)

        if current_group:
            groups.append(current_group)

        # Decode từng nhóm id → từ tiếng Việt thật (dùng chính tokenizer.decode)
        words = []
        id_to_pos = {tok_id: i for i, tok_id in enumerate(ids)}  # map ngược để lấy score

        agg_scores = []
        pos = 0
        for group in groups:
            word = self.tokenizer.decode(group).strip()
            words.append(word if word else '<?>')

            # Trung bình attribution score của các sub-token trong nhóm
            group_len = len(group)
            group_scores = scores[pos:pos + group_len]
            agg_scores.append(float(np.mean(group_scores)))
            pos += group_len

        return words, np.array(agg_scores)

    def top_tokens(self, result, k=15):
        tokens, scores = result['tokens'], result['scores']
        # Drop special tokens and padding
        valid = [(t, s) for t, s in zip(tokens, scores)
                 if t not in ('<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]')]
        valid.sort(key=lambda x: x[1], reverse=True)
        return valid[:k]