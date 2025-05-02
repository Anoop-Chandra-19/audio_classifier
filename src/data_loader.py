import os
import torch
from torch.utils.data import Dataset
import pandas as pd

def read_fma_tracks(path: str) -> pd.DataFrame:
    """
    Reads the FMA tracks.csv file (with multi-line header) and returns 
    a DataFrame indexed by track_id, with flattened and unique column names.

    Args:
        path (str): Path to the tracks.csv file.
    
    Returns:
        pd.DataFrame: DataFrame with track metadata.
    """
    with open(path, encoding='utf-8') as f:
        grp = next(f).strip().split(',') # End result is a list of strings ["", "album", "artist", ...]
        fld = next(f).strip().split(',') # End result is a list of strings ["", "comments", "date_created", ...]
        _   = next(f)                    # Skip the third line so that on subsequent reads, 
                                         # the next line starts with the data. 
                                         # Not necessary but makes it easier to understand.

    names = []
    for g, h in zip(grp, fld):
        if h == "":        # Notice that the first column is empty, so we use this logic to append 'track_id'
            names.append("track_id")
        elif g:            # If the group is not empty, we flatten and append the group name to the field name
            names.append(f"{g}_{h}")
        else:              # If the group is empty, we just append the field name
            names.append(h)
        
    df = pd.read_csv(
        path,
        skiprows=3,        # Skip the first three lines (header)
        header=None,       # No header since we are providing our own names
        names=names,
        index_col='track_id', 
    )                      
    df.index = df.index.astype(int)
    return df

class AudioDataset(Dataset):
    """
    Custom dataset for loading audio spectrograms and their corresponding labels.
    
    This dataset is designed to work with the FMA dataset, specifically the processed spectrograms.
    The dataset expects the processed spectrograms to be stored in a directory, with each spectrogram
    saved as a .pt file named after its track ID. The metadata is read from a CSV file, and the labels
    are extracted from a specified field in the metadata.
    
    Args:
        processed_data_dir (str): Directory containing the processed spectrogram files.
        metadata_file (str): Path to the CSV file containing metadata for the audio tracks.
        label_field (str): Column name in the metadata file to use as labels. Default is 'track_genre_top'.
        transform (callable, optional): Optional transform to be applied on a sample.
    
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns a sample from the dataset.    
    """
    def __init__(
            self,
            processed_data_dir: str,
            metadata_file: str,
            label_field: str = 'track_genre_top',
            transform=None,
        ):
            """
            Initializes the AudioDataset.
            
            Raises:
                KeyError: If the label field is not found in the metadata.
                FileNotFoundError: If the processed data directory or metadata file does not exist.    
            """
        
            self.processed_data_dir = processed_data_dir
            # Load metadata
            self.df = read_fma_tracks(metadata_file)

            # Verify label field
            if label_field not in self.df.columns:
                raise KeyError(f"Label field '{label_field}' not found in metadata.")


            # Find all track IDs for which we have a .pt file
            available = {
                int(os.path.splitext(f)[0])
                for f in os.listdir(processed_data_dir)
                if f.endswith('.pt')
            }

            # Keep only metadata rows with available spectrograms
            self.df = self.df.loc[self.df.index.intersection(available)]

            # Build a label-to-index map from the chosen column
            genres = sorted(self.df[label_field].unique())
            self.label_map = {g : i for i, g in enumerate(genres)}

            self.transform = transform

            # Prepare list of samples: {track_id, genre}
            self.samples = [
                (tid, row[label_field])
                for tid, row in self.df.iterrows()
            ]
    
    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        
        Raises:
            ValueError: If the dataset is empty.
        
        Example:
            >>> dataset = AudioDataset(processed_data_dir='data/processed', metadata_file='data/raw/fma_small/tracks.csv')
            >>> len(dataset)
            7997

        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (spec, label) where spec is the spectrogram tensor and label is the genre index.
        
        Raises:
            IndexError: If the index is out of range.

        Example:
            >>> dataset = AudioDataset(processed_data_dir='data/processed', metadata_file='data/raw/fma_small/tracks.csv')
            >>> spec, label = dataset[0]
            >>> print(spec.shape, label)
            torch.Size([128, 937]), 3        
        """
        track_id, genre = self.samples[idx]
        fname = str(track_id).zfill(6) + ".pt"  # Ensure track_id is zero-padded to 6 digits
        spec_path = os.path.join(self.processed_data_dir, fname)
        spec = torch.load(spec_path)
        spec = spec.squeeze()
        if spec.ndim != 2:
            spec = spec.view(spec.shape[-2], spec.shape[-1])
        if self.transform:
            spec = self.transform(spec)
        label = self.label_map[genre]
        return spec, label

if __name__ == "__main__":
    # Example usage
    dataset = AudioDataset(
        processed_data_dir='data/processed',
        metadata_file='data/raw/fma_small/tracks.csv'
    )
    print(f"Number of samples: {len(dataset)}")
    spec, label = dataset[0]
    print(f"Sample spec shape: {spec.shape}, Sample label: {label}")
    