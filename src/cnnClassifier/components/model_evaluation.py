import torch
from pathlib import Path
import mlflow
import mlflow.pytorch
from urllib.parse import urlparse
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from torch.utils.data import DataLoader
from cnnClassifier.utils.common import save_json
from cnnClassifier.entity.config_entity import EvaluationConfig


class MyLazyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def set_loaders(self):
        # This method allows the user to define which train_loader (and val_loader, optionally) to use
        # Both loaders are then assigned to attributes of the class
        # So they can be referred to later
        # Image transformations
        image_transforms = {
            # Train uses data augmentation
            'train':
            transforms.Compose([
                transforms.RandomResizedCrop(size=200, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=64),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])  # Imagenet standards
            ]),
            # Validation does not use augmentation
            'val':
            transforms.Compose([
                transforms.Resize(size=64),
                transforms.CenterCrop(size=180),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        self.dataset = ImageFolder(root=self.config.training_data)
        train, val = torch.utils.data.random_split(self.dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
        traindataset = MyLazyDataset(train, transform=image_transforms['train'])
        valdataset = MyLazyDataset(val, transform=image_transforms['val'])
        trainLoader = DataLoader(traindataset , batch_size=self.config.params_batch_size, shuffle=True)
        valLoader = DataLoader(valdataset, batch_size=self.config.params_batch_size)
        
        return trainLoader, valLoader


    @staticmethod
    def load_model(path: Path) -> torch.nn.Module:
        checkpoint = torch.load(path)
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']
        # Load in the state dict
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to('cuda')
        return model


    def evaluation(self, val_on_gpu=True):
        self.model = self.load_model(self.config.path_of_model)
        _, valLoader = self.set_loaders()
        # Test the model
        self.model.eval()

        val_loss = 0
        val_acc = 0

        for data, target in valLoader:
            if val_on_gpu:
                data, target = data.cuda(), target.cuda()
            pred = self.model(data)
            
            loss = self.loss_fn(pred, target)

            val_loss += loss.item() * data.size(0)

            
            _, pred = torch.max(pred, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(
                correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples
            val_acc += accuracy.item() * data.size(0)

        val_loss /= len(valLoader.dataset)
        val_acc /= len(valLoader.dataset)
        
        self.score = (val_loss, val_acc)
        
        print(f'Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
        
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)


    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.pytorch.log_model(self.model, "model")