import torch
import numpy as np
from captum.attr import LayerIntegratedGradients
from captum.metrics import infidelity
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

    def _build_perturb_mask(self, inputs, attention_mask, perturb_ratio):
        random_values = torch.rand_like(inputs, dtype=torch.float)
        mask = (random_values < perturb_ratio).long()
        mask = mask * attention_mask # only keep mask in a real token position

        # Remove <CLS> and <SEP> tokens
        mask[:, 0] = 0
        seq_lengths = attention_mask.sum(dim=1) - 1
        mask[torch.arange(mask.size(0)), seq_lengths] = 0
        return mask

    def evaluate_faithfulness(self, text, target_label, n_steps=15, perturb_ratio=0.2, n_perturb_samples=5):
        encoding = self.tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding='max_length', return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        baseline_id = self.tokenizer.pad_token_id

        attributions = self.lig.attribute(
            inputs=input_ids, baselines=torch.full_like(input_ids, baseline_id),
            additional_forward_args=(attention_mask,), target=target_label, n_steps=n_steps
        )
        attributions = attributions.sum(dim=-1)

        def perturb_func_infidelity(inputs):
            mask = self._build_perturb_mask(inputs, attention_mask, perturb_ratio)

            perturbed = inputs * (1 - mask) + baseline_id * mask
            pertubation = mask.float()
            return pertubation, perturbed # which tokens have changed & changed inputs

        infid = infidelity(
            self._forward_func, perturb_func_infidelity, input_ids, attributions,
            additional_forward_args=(attention_mask,), target=target_label,
            n_perturb_samples=n_perturb_samples
        )

        original_attr_flat = attributions.view(-1)
        original_norm = torch.norm(original_attr_flat) + 1e-10

        max_sensitivity = 0.0

        for _ in range(n_perturb_samples):
            mask = self._build_perturb_mask(input_ids, attention_mask, perturb_ratio)
            perturbed_input_ids = input_ids * (1 - mask) + baseline_id * mask

            perturbed_attr = self.lig.attribute(
                inputs=perturbed_input_ids, baselines=torch.full_like(perturbed_input_ids, baseline_id),
                additional_forward_args=(attention_mask,), target=target_label, n_steps=n_steps
            )
            perturbed_attr = perturbed_attr.sum(dim=-1).view(-1)
 
            diff_norm = torch.norm(original_attr_flat - perturbed_attr)
            sensitivity = (diff_norm / original_norm).item()
            max_sensitivity = max(max_sensitivity, sensitivity)
        
        return {
            'infidelity': infid.item(),
            'sensitivity_max': max_sensitivity
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