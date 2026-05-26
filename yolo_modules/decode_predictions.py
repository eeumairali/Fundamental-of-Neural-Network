import torch


def decode_predictions(preds, image_size=448, conf_threshold=0.25, S=7, B=2, C=20):
    """Turn one model output tensor into box, score, and class lists.

    preds: Tensor with shape (N, S, S, B*5 + C) or (S, S, B*5 + C) or (1, S, S, ...)
    Returns: boxes (K,4), scores (K,), classes (K,)
    """
    if preds.dim() == 4 and preds.size(0) == 1:
        preds = preds[0]

    # Ensure preds has shape (S, S, B*5 + C)
    assert preds.dim() == 3, "preds must have shape (S, S, B*5 + C)"

    boxes_out = []
    scores_out = []
    classes_out = []
    cell_size = image_size / S

    class_probs = torch.softmax(preds[..., B * 5:], dim=-1)

    for row in range(S):
        for col in range(S):
            cell = preds[row, col]
            classes = class_probs[row, col]
            for box_idx in range(B):
                offset = box_idx * 5
                x, y, w, h, conf = cell[offset:offset + 5]
                score, class_idx = classes.max(dim=-1)
                final_score = conf * score
                if final_score < conf_threshold:
                    continue

                abs_cx = (col + x) * cell_size
                abs_cy = (row + y) * cell_size
                abs_w = w * image_size
                abs_h = h * image_size
                x1 = abs_cx - abs_w / 2
                y1 = abs_cy - abs_h / 2
                x2 = abs_cx + abs_w / 2
                y2 = abs_cy + abs_h / 2

                boxes_out.append([x1.item() if isinstance(x1, torch.Tensor) else x1,
                                  y1.item() if isinstance(y1, torch.Tensor) else y1,
                                  x2.item() if isinstance(x2, torch.Tensor) else x2,
                                  y2.item() if isinstance(y2, torch.Tensor) else y2])
                scores_out.append(final_score.item() if isinstance(final_score, torch.Tensor) else final_score)
                classes_out.append(int(class_idx.item() if isinstance(class_idx, torch.Tensor) else class_idx))

    if not boxes_out:
        return torch.empty((0, 4)), torch.empty(0), torch.empty(0, dtype=torch.long)

    return torch.tensor(boxes_out), torch.tensor(scores_out), torch.tensor(classes_out, dtype=torch.long)


if __name__ == "__main__":
    # quick smoke test
    preds = torch.rand(1, 7, 7, 30)
    boxes, scores, classes = decode_predictions(preds, image_size=448, S=7, B=2, C=20)
    print(boxes.shape, scores.shape, classes.shape)
