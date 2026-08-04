import logging
import torch
from transformers import AutoTokenizer
from src.training.model import AIContentModel
from src.evaluation import inference
from src.explainability.attribution import AttributionExplainer
from ..core.config import Settings

logger = logging.getLogger(__name__)

AL_LABEL = 1

class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings

        if settings.device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.tokenizer = None
        self.model = None
        self.explainer = None

    def load(self):
        logger.info('Loading tokenizer: %s', self.settings.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name)

        self.model = AIContentModel(model_name=self.settings.model_name, num_classes=2)
        checkpoint = torch.load(self.settings.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.explainer = AttributionExplainer(self.model, self.tokenizer, self.device, max_length=self.settings.max_length)
        logger.info('Model service ready on device: %s', self.device)

    def predict(self, text):
        if self.model is None:
            raise RuntimeError('Model chưa được load')
        return inference.predict_single_text(
            self.model, self.tokenizer, text, self.device,
            max_length=self.settings.max_length, clean=True
        )

    def explain(self, text):
        if self.explainer is None:
            raise RuntimeError('Model chưa được load')
        result = self.explainer.explain(
            text, target_label=AL_LABEL, n_steps=self.settings.ig_n_steps
        )

        return {
            'tokens': result['tokens'],
            'scores': result['scores'].tolist(),
            'predicted_label': result['predicted_label'],
            'pred_prob': result['pred_prob']
        }