import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
from model import MiniSeg

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(pretrained_dir: str, device: str = DEFAULT_DEVICE):
    print(f"Loading model from {pretrained_dir}...")
    pretrained_dir = pretrained_dir.rstrip("/")
    tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_dir}/sentence_encoder")

    sentence_encoder = AutoModel.from_pretrained(f"{pretrained_dir}/sentence_encoder")
    doc_encoder = AutoModelForTokenClassification.from_pretrained(
        f"{pretrained_dir}/doc_encoder"
    )
    model = MiniSeg(sentence_encoder, doc_encoder).to(device)
    model.load_state_dict(
        torch.load(f"{pretrained_dir}/miniseg_model.pth", map_location=device)
    )
    model.eval()

    return model, tokenizer
