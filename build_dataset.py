import os
from datasets import load_dataset
from tqdm import tqdm

def build_dataset_folders():
    TRAIN_SAMPLES = 2000
    VAL_SAMPLES = 500
    base_dir = "dataset"
    
    # 1. Create folders safely
    for split in ["train", "val"]:
        for cat in ["real", "ai"]:
            os.makedirs(os.path.join(base_dir, split, cat), exist_ok=True)
            
    print("⏳ Connecting to Hugging Face...")
    try:
        # Using a verified, high-quality deepfake detection dataset
        dataset = load_dataset("prithivMLmods/OpenDeepfake-Preview", split="train")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return
        
    counts = {"train": {"real": 0, "ai": 0}, "val": {"real": 0, "ai": 0}}
    
    print("🚀 Extracting and saving images to local folders...")
    for item in tqdm(dataset, desc="Processing Images"):
        image = item['image']
        label = item['label'] 
        
        # In this specific dataset: 0 = Fake (AI), 1 = Real
        cat_str = "real" if label == 1 else "ai"
            
        if counts["train"][cat_str] < TRAIN_SAMPLES:
            split_str = "train"
        elif counts["val"][cat_str] < VAL_SAMPLES:
            split_str = "val"
        else:
            continue 
            
        save_path = os.path.join(base_dir, split_str, cat_str, f"{split_str}_{cat_str}_{counts[split_str][cat_str]}.jpg")
        
        # Convert to standard RGB to prevent training crashes
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image.save(save_path, "JPEG", quality=95)
        counts[split_str][cat_str] += 1
        
        # Stop early once we hit our target image counts
        if (counts["train"]["real"] == TRAIN_SAMPLES and counts["train"]["ai"] == TRAIN_SAMPLES and
            counts["val"]["real"] == VAL_SAMPLES and counts["val"]["ai"] == VAL_SAMPLES):
            break

    print("\n✅ Dataset construction complete!")
    print(f"📁 Training Folder:   {counts['train']['real']} Real | {counts['train']['ai']} AI")
    print(f"📁 Validation Folder: {counts['val']['real']} Real | {counts['val']['ai']} AI")

if __name__ == "__main__":
    build_dataset_folders()