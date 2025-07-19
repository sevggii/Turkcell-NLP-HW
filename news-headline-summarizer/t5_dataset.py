import torch
from torch.utils.data import Dataset
import numpy as np

class T5SummarizationDataset(Dataset):
    def __init__(self, data_path, tokenizer=None):
        """
        T5 modeli için özel dataset sınıfı
        
        Args:
            data_path: .npz dosyasının yolu
            tokenizer: T5 tokenizer (opsiyonel)
        """
        self.data = np.load(data_path)
        self.tokenizer = tokenizer
        
        # Data keys'lerini kontrol et
        self.keys = list(self.data.keys())
        print(f"Dataset yüklendi: {data_path}")
        print(f"Available keys: {self.keys}")
        print(f"Dataset boyutu: {len(self.data[self.keys[0]])}")
    
    def __len__(self):
        return len(self.data[self.keys[0]])
    
    def __getitem__(self, idx):
        """
        Tek bir sample döndür
        
        Returns:
            dict: input_ids, attention_mask, labels
        """
        item = {}
        
        if 'input_ids' in self.keys:
            item['input_ids'] = torch.tensor(self.data['input_ids'][idx], dtype=torch.long)
        
        if 'attention_mask' in self.keys:
            item['attention_mask'] = torch.tensor(self.data['attention_mask'][idx], dtype=torch.long)
        
        if 'labels' in self.keys:
            item['labels'] = torch.tensor(self.data['labels'][idx], dtype=torch.long)
        
        return item

def create_t5_datasets(train_path, val_path, test_path=None):
    """
    T5 için dataset'leri oluştur
    
    Args:
        train_path: Train data yolu
        val_path: Validation data yolu  
        test_path: Test data yolu (opsiyonel)
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    print("T5 Dataset'leri oluşturuluyor...")
    
    train_dataset = T5SummarizationDataset(train_path)
    val_dataset = T5SummarizationDataset(val_path)
    
    test_dataset = None
    if test_path:
        test_dataset = T5SummarizationDataset(test_path)
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    if test_dataset:
        print(f"Test dataset: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset

def test_dataset_loading():
    """Dataset yükleme testi"""
    cnn_dir = "cnn_dailymail"
    train_path = f"{cnn_dir}/train_processed_t5.npz"
    val_path = f"{cnn_dir}/validation_processed_t5.npz"
    
    try:
        train_dataset, val_dataset, _ = create_t5_datasets(train_path, val_path)
        
        # İlk sample'ı test et
        sample = train_dataset[0]
        print("İlk sample:")
        for key, value in sample.items():
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        
        print("Dataset testi başarılı!")
        return True
        
    except Exception as e:
        print(f"Dataset testi başarısız: {e}")
        return False

if __name__ == "__main__":
    test_dataset_loading() 