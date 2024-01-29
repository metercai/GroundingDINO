from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast
from download import download
import os


def load_model_and_save_to_local(text_encoder_type):
    model_path = os.path.expanduser(os.path.join(os.getenv("BERT_HOME", os.path.join(os.getcwd(), "models/llms/")), text_encoder_type))
    #print(f'bert_model_path:{model_path}')
    if not os.path.exists(model_path):
        os.mkdirs(model_path)
    model_file_path = os.path.join(model_path, 'model.safetensors')
    if not os.path.exists(model_file_path):
        url = "https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors"
        if os.getenv("HUF_MIRROR"):
            url.relpace("huggingface.co", os.getenv("HUF_MIRROR"))
        download(url, model_file_path, progressbar=True)
    return model_path

def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    print("final text_encoder_type: {}".format(text_encoder_type))

    tokenizer = AutoTokenizer.from_pretrained(load_model_and_save_to_local(text_encoder_type))
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    if text_encoder_type == "bert-base-uncased":
        return BertModel.from_pretrained(load_model_and_save_to_local(text_encoder_type))
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(load_model_and_save_to_local(text_encoder_type))

    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
