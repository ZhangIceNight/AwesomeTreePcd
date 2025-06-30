# train.py

from trainers.cls_trainer import ClassificationTrainer
from config import cfg

if __name__ == '__main__':
    trainer = ClassificationTrainer(cfg)
    trainer.train()
