
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import glob

class FakeNewsDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_len=512, sample_size=None, kg_builder=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.kg_builder = kg_builder
        self.data = self._load_data(data_dir, sample_size)

    def _load_data(self, data_dir, sample_size):
        # Load all CSVs
        # BuzzFeed_fake, BuzzFeed_real, PolitiFact_fake, PolitiFact_real
        files = glob.glob(os.path.join(data_dir, "*_content.csv"))
        dfs = []
        for f in files:
            df = pd.read_csv(f)
            # Determine label from filename
            if "fake" in os.path.basename(f).lower():
                df['label'] = 0 # FAKE
            elif "real" in os.path.basename(f).lower():
                df['label'] = 1 # REAL
            else:
                continue
            
            # Keep only necessary columns
            if 'text' in df.columns:
                 dfs.append(df[['text', 'label']])
        
        if not dfs:
            raise ValueError("No data files found in " + data_dir)
            
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Clean
        full_df.dropna(subset=['text'], inplace=True)
        full_df = full_df[full_df['text'].str.strip().astype(bool)]
        
        # Sample if requested
        if sample_size and sample_size < len(full_df):
            full_df = full_df.sample(n=sample_size, random_state=42)
            
        print(f"Loaded {len(full_df)} samples. Class balance: {full_df['label'].value_counts().to_dict()}")
        return full_df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['text'])
        label = row['label']

        # Tokenize (Subword tokenization via XLM-R)
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # KG Features
        kg_features = torch.zeros(3) # [nodes, edges, density]
        if self.kg_builder:
             try:
                 stats = self.kg_builder.extract_features(text)
                 kg_features = torch.tensor(stats, dtype=torch.float)
             except Exception as e:
                 # Fallback if KG fails (rare)
                 pass

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'kg_features': kg_features,
            'label': torch.tensor(label, dtype=torch.float) 
        }
