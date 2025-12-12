"""
Lightning DataModule for DAiSEE dataset
Supports both binary and multi-class classification with pre-split data
Simplified version without complex binary mapping logic
"""
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pathlib import Path
from .dataset import ImagePathDataset
from PIL import Image



class DAiSEEDataModule(L.LightningDataModule):
    """
    Lightning DataModule for DAiSEE Confusion Detection
    Supports both binary (not confused vs confused) and multi-class (4 levels) classification
    
    Args:
        data_dir: Path to dataset root directory containing Train/Validation/Test folders
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        img_size: Image size for resizing
        seed: Random seed for reproducibility
        task_type: Classification type - 'binary' or 'multiclass'
    """
    def __init__(
        self, 
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 224,
        seed: int = 42,
        task_type: str = 'multiclass',
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.seed = seed
        self.task_type = task_type.lower()
        
        # Validate task type
        if self.task_type not in ['binary', 'multiclass']:
            raise ValueError(f"task_type must be 'binary' or 'multiclass', got '{task_type}'")
        
        # Set num_classes and class_names immediately
        self.num_classes = 2 if self.task_type == 'binary' else 4
        self.class_names = (
            ['Not Confused', 'Confused'] if self.task_type == 'binary' else
            ['Not Confused', 'Slightly Confused', 'Confused', 'Very Confused']
        )
        
        # Verify directories exist
        self.train_dir = self.data_dir / "Train"
        self.val_dir = self.data_dir / "Validation"
        self.test_dir = self.data_dir / "Test"
        
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            if not split_dir.exists():
                raise ValueError(f"Directory not found: {split_dir}")
        
        # Define transforms
        self.test_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_transform = T.Compose([
            T.Resize((img_size, img_size)),
            # Lighting augmentations
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
            T.RandomApply([T.RandomGrayscale(p=1.0)], p=0.1),
            # Geometric augmentations (mild)
            T.RandomAffine(
                degrees=3,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            T.RandomPerspective(distortion_scale=0.05, p=0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Dataset placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """
        Setup datasets from pre-split directories
        Converts to binary labels if needed (class 0 vs rest)
        """
        # Only create splits once
        if not hasattr(self, '_splits_created'):
            # Load train split
            train_paths, train_labels = self._load_split(self.train_dir)
            self._train_paths = train_paths
            self._train_labels = train_labels
            
            # Load validation split
            val_paths, val_labels = self._load_split(self.val_dir)
            self._val_paths = val_paths
            self._val_labels = val_labels
            
            # Load test split
            test_paths, test_labels = self._load_split(self.test_dir)
            self._test_paths = test_paths
            self._test_labels = test_labels
            
            self._splits_created = True
            
            # Print summary
            print(f"\n✓ Loaded DAiSEE dataset ({self.task_type} classification)")
            print(f"  Training: {len(self._train_paths)} samples")
            print(f"  Validation: {len(self._val_paths)} samples")
            print(f"  Test: {len(self._test_paths)} samples")
            
            if self.task_type == 'binary':
                print(f"\n  Binary distribution:")
                print(f"    Train - Class 0: {self._train_labels.count(0)}, Class 1: {self._train_labels.count(1)}")
                print(f"    Val   - Class 0: {self._val_labels.count(0)}, Class 1: {self._val_labels.count(1)}")
                print(f"    Test  - Class 0: {self._test_labels.count(0)}, Class 1: {self._test_labels.count(1)}")
        
        # Create datasets based on stage
        if stage == "fit" or stage is None:
            if self.train_dataset is None:
                self.train_dataset = ImagePathDataset(
                    self._train_paths, 
                    self._train_labels, 
                    transform=self.train_transform
                )
            if self.val_dataset is None:
                self.val_dataset = ImagePathDataset(
                    self._val_paths, 
                    self._val_labels, 
                    transform=self.test_transform
                )
        
        if stage == "test" or stage is None:
            if self.test_dataset is None:
                self.test_dataset = ImagePathDataset(
                    self._test_paths, 
                    self._test_labels, 
                    transform=self.test_transform
                )
    
    def _load_split(self, split_dir: Path):
        """
        Load paths and labels from a split directory
        Handles conversion to binary labels if needed
        
        Args:
            split_dir: Path to split directory (Train/Validation/Test)
            
        Returns:
            paths: List of image paths
            labels: List of labels (converted to binary if task_type='binary')
        """
        paths = []
        labels = []
        
        # Mapping from folder name to class index
        folder_to_class = {
            '0_not_confused': 0,
            '1_slightly_confused': 1,
            '2_confused': 2,
            '3_very_confused': 3
        }
        
        # Collect all images
        for folder_name, class_idx in folder_to_class.items():
            folder_path = split_dir / folder_name
            if not folder_path.exists():
                continue
                
            for img_path in folder_path.glob('*.jpg'):
                paths.append(str(img_path))
                
                # Convert to binary if needed: class 0 stays 0, rest become 1
                if self.task_type == 'binary':
                    binary_label = 0 if class_idx == 0 else 1
                    labels.append(binary_label)
                else:
                    labels.append(class_idx)
        
        return paths, labels
    
    def get_test_sample(self, idx: int):
        """
        Return (image: Tensor, label: int) for test set index `idx`.
        
        The image is returned **as transformed by test_transform** (normalized, resized).
        This matches exactly what your test_dataloader returns.
        
        Args:
            idx: Index in the test split (0 to len(test_dataset)-1)
            
        Returns:
            (image: torch.Tensor in [C, H, W], label: int)
        """
        if not hasattr(self, '_test_paths') or not hasattr(self, '_test_labels'):
            self.setup(stage='test')
        
        if idx < 0 or idx >= len(self._test_paths):
            raise IndexError(f"Test set index {idx} out of range (0–{len(self._test_paths)-1})")
        
        path = self._test_paths[idx]
        label = self._test_labels[idx]
        
        # Load and transform
        image = Image.open(path).convert('RGB')
        image = self.test_transform(image)
        
        return image, label
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
            persistent_workers=True if self.num_workers > 0 else False,
        )