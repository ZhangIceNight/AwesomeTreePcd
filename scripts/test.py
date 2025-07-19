# test.py

import torch
from config import cfg
from models import pointnet
from datasets.pls_dataset import PLSDataset

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for points, labels in dataloader:
            points, labels = points.to(device), labels.to(device)
            points = points.transpose(2, 1)
            outputs, _ = model(points)
            preds = outputs.max(1)[1]
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    return acc

if __name__ == '__main__':
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

    test_dataset = PLSDataset(
        root=cfg['data_root'],
        split='test',
        npoints=cfg['num_points'],
        use_uniform_sample=cfg['use_uniform_sample'],
        use_normals=cfg['use_normals']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers']
    )

    num_classes = len(test_dataset.classes)
    model = pointnet.get_model(num_classes, normal_channel=cfg['use_normals']).to(device)

    model.load_state_dict(torch.load(cfg['save_path']))
    acc = evaluate(model, test_loader, device)
    print(f'[✓] Test Accuracy: {acc:.4f}')
