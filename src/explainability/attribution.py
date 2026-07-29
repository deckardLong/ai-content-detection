import torch
import numpy as np
from captum.attr import LayerIntegratedGradients
from underthesea import word_tokenize as vn_word_tokenize

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
        return logits

    def explain(self, text, target_label, n_steps=50):
        # Step 1: Get attribution at syllable level (as usual)
        encoding = self.tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding='max_length', return_tensors='pt',
            return_offsets_mapping=True # return position of each token
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offsets = encoding['offset_mapping'][0].tolist()  # [(start,end), ...] from original text

        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)

        with torch.no_grad():
            logits = self._forward_func(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0, pred_label].item()

        attributions, delta = self.lig.attribute(
            inputs=input_ids, baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=target_label, return_convergence_delta=True, n_steps=n_steps
        )
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / (torch.norm(attributions) + 1e-10) # keep in stable range
        syllable_scores = attributions.cpu().detach().numpy()

        # Step 2: Combine syllables into compound nouns
        vn_words = vn_word_tokenize(text)  # for example: ['Samsung', 'đang', 'gặp', 'khó khăn', ...]

        # Step 3: Map for each compound nouns -> positional range from original text
        word_spans = []
        cursor = 0
        for w in vn_words:
            start = text.find(w, cursor)
            if start == -1: # unnecessary, defensive programming
                continue
            end = start + len(w)
            word_spans.append((w, start, end))
            cursor = end

        # Step 4: Combine attribution of syllable-tokens to fall in range of each compound nouns
        word_results = []
        for word, w_start, w_end in word_spans:
            scores_in_span = [
                syllable_scores[i] for i, (s, e) in enumerate(offsets)
                if s < w_end and e > w_start and not (s == 0 and e == 0)  # drop special tokens
            ]
            if scores_in_span:
                word_results.append((word, float(np.mean(scores_in_span))))

        words = [w for w, s in word_results]
        scores = np.array([s for w, s in word_results])

        return {
            'tokens': words,
            'scores': scores,
            'predicted_label': pred_label,
            'pred_prob': pred_prob,
            'convergence_delta': delta.item()
        }

    def _decode_subword_tokens(self, subword_tokens):
        """
        Convert BPE tokens to Vietnamese
        """
        decoded = []
        for token in subword_tokens:
            if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                continue  
            
            if token.startswith('##'):
                token = token[2:]
            
            decoded.append(token)
        
        return decoded

    def _aggregate_scores(self, subword_tokens, scores):
        """
        Aggregate tokens to one word
        """
        
        aggregated = []
        current_word = ''
        current_scores = []
        
        for token, score in zip(subword_tokens, scores):
            if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                continue
            
            if token.startswith('##'):
                current_word += token[2:]
                current_scores.append(score)
            else:
                if current_word:
                    aggregated.append((current_word, np.mean(current_scores)))

                current_word = token
                current_scores = [score]

        if current_word:
            aggregated.append((current_word, np.mean(current_scores)))

        words = [w for w, s in aggregated]
        agg_scores = np.array([s for w, s in aggregated])
        
        return agg_scores

    def top_tokens(self, result, k=15, dedupe='sum'):
        tokens, scores = result['tokens'], result['scores']
        valid = [(t, s) for t, s in zip(tokens, scores)
                 if t not in ('<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]', '<?>')]

        if dedupe in ('sum', 'mean', 'max'):
            agg, counts = {}, {}
            for word, score in valid:
                agg[word] = agg.get(word, 0) + score if dedupe in ('sum','mean') else max(agg.get(word, score), score)
                counts[word] = counts.get(word, 0) + 1
            if dedupe == 'mean':
                agg = {w: s / counts[w] for w, s in agg.items()}
            valid = list(agg.items())

        valid.sort(key=lambda x: x[1], reverse=True)
        return valid[:k]