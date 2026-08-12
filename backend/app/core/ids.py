import secrets
import string
from datetime import datetime, timezone

_ALPHABET = string.ascii_lowercase + string.digits


def generate_meaningful_id(prefix: str, length: int = 6):
    """
    Meaningful ID: <prefix>_<timestamp>_<short random>
    Example: usr_20260812153045_a1b2c3, pred_20260812153102_x7k9m2
    """
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
    random_suffix = ''.join(secrets.choice(_ALPHABET) for _ in range(length))
    return f'{prefix}_{timestamp}_{random_suffix}'