import logging
import os
import sys

from transformers import AutoModel, AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python cache_model.py <model_name_or_path>")
        sys.exit(1)

    model_name = sys.argv[1]
    # It's often necessary to trust remote code for newer models
    trust_remote_code = True

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
    logger.info(f"Running on CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    logger.info(f"Attempting to cache model and tokenizer for: {model_name}")

    try:
        # Cache tokenizer
        logger.info(f"Caching tokenizer '{model_name}'...")
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        logger.info("Tokenizer cached successfully.")

        # Cache model
        logger.info(f"Caching model '{model_name}' (this might take a while)...")
        AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        logger.info("Model cached successfully.")

        logger.info(f"Caching complete for: {model_name}")

    except Exception as e:
        logger.exception(f"Error caching model/tokenizer {model_name}: {e}")
        sys.exit(1)
