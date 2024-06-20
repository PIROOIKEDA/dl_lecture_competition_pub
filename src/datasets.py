import os
import torch

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        print(f"Initializing dataset: {split} from {data_dir}")  # デバッグ用プリント文
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        # 各ファイルの読み込みにプリント文を追加
        try:
            print(f"Loading {split}_X.pt")
            file_path = os.path.join(data_dir, f"{split}_X.pt")
            print(f"Checking file existence: {os.path.exists(file_path)}")  # ファイルの存在確認
            print(f"File size: {os.path.getsize(file_path) / (1024 * 1024)} MB")  # ファイルのサイズ確認

            self.X = torch.load(file_path)
            print(f"Loaded {split}_X.pt")

            print(f"Loading {split}_subject_idxs.pt")
            self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
            print(f"Loaded {split}_subject_idxs.pt")

            if split in ["train", "val"]:
                print(f"Loading {split}_y.pt")
                self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
                print(f"Loaded {split}_y.pt")
                assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        except Exception as e:
            print(f"An error occurred while loading data: {e}")

        print(f"Loaded dataset: {split} with {len(self.X)} samples.")  # デバッグ用プリント文

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
