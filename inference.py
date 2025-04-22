# inference.py
import io, json, os, base64, logging
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(num_classes: int):
    model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
    in_feats = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_feats, num_classes)
    return model

def preprocess(spec_np: np.ndarray) -> torch.Tensor:
    """Convert (128,H,W or H,W,3) ndarray → torch (1,3,128,W) ready for EfficientNet."""
    spec = torch.from_numpy(spec_np).float()
    spec = spec.permute(2, 0, 1)
    if spec.max() > 1:
        spec = spec / 255.0

    # flip / normalise – use only normalise at inference
    norm = transforms.Normalize(MEAN, STD)
    spec = norm(spec)
    return spec.unsqueeze(0)  # add batch dim

def topk_probs(logits, k=5):
    probs = torch.softmax(logits, dim=1)
    topv, topi = probs.topk(k)
    return topv.cpu().tolist()[0], topi.cpu().tolist()[0]

def model_fn(model_dir):
    # load label mapping saved during training
    print(f"[inference.py] model_fn called with model_dir={model_dir}", flush=True)
    with open(os.path.join(model_dir, 'species_mapping.json')) as f:
        idx_to_label = json.load(f)

    num_classes = len(idx_to_label)
    model = build_model(num_classes)
    weights = torch.load(os.path.join(model_dir, 'best_model.pth'),
                         map_location=DEVICE)
    model.load_state_dict(weights)
    model.to(DEVICE).eval()
    model.idx_to_label = idx_to_label
    logging.info("Model loaded with %d classes", num_classes)
    return model

def input_fn(request_body, content_type):
    print(f"[inference.py] input_fn got content_type={content_type}, "
      f"body_len={len(request_body)}", flush=True)
    if content_type == 'application/x-npy':
        data = np.load(io.BytesIO(request_body), allow_pickle=True)
        print(f"[input_fn] Loaded array shape: {data.shape}, dtype: {data.dtype}")
        return data
    if content_type == 'application/json':
        payload = json.loads(request_body)
        data_b64 = payload['array']
        return np.load(io.BytesIO(base64.b64decode(data_b64)))
    raise ValueError(f'Unsupported content type: {content_type}')

def predict_fn(input_data, model):
    print(f"[inference.py] predict_fn got data.shape={input_data.shape}", flush=True)
    with torch.no_grad():
        tensor = preprocess(input_data).to(DEVICE)
        logits = model(tensor)
    topv, topi = topk_probs(logits, k=5)
    top_labels = [model.idx_to_label[str(i)] for i in topi]
    return {'top5_labels': top_labels,
            'top5_probs':  topv}

def output_fn(prediction, accept):
    print(f"[inference.py] output_fn returning accept={accept}", flush=True)
    return json.dumps(prediction), accept
