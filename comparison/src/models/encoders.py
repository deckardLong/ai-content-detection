from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

def load_encoder(name: str):
    """Return model pretrained"""
    return AutoModel.from_pretrained(name)

def load_tokenizer(name: str):
    """BamiBERT declares tokenizer_class=XLMRobertaTokenizer but only ships tokenizer.json
    (byte-level BPE), which AutoTokenizer in transformers 5.x cannot load -> fall back to the fast tokenizer."""
    try:
        return AutoTokenizer.from_pretrained(name)
    except (TypeError, ValueError, OSError):
        return PreTrainedTokenizerFast.from_pretrained(name)
