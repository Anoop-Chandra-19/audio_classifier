
import os
import argparse
import torch
import torch.nn as nn
from torch import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from src.data_loader import AudioDataset
from src.model import AudioClassifier
from src.transforms import SpecAugment 


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    runningloss =  0.0
    for specs, labels in loader:
        specs = specs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast(device_type=device.type):
            logits = model(specs)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        runningloss += loss.item() * specs.size(0)
    
    return runningloss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    runningloss = 0.0
    correct = 0

    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(device)
            labels = labels.to(device)

            logits = model(specs)
            loss = criterion(logits, labels)

            runningloss += loss.item() * specs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
    avg_loss = runningloss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)

    return avg_loss, accuracy

def pad_collate(batch):
    specs, labels = zip(*batch)
    max_len = max(s.size(1) for s in specs)
    specs_padded = torch.stack([F.pad(s, (0, max_len - s.size(1))) for s in specs])
    labels = torch.tensor(labels, dtype=torch.long)
    return specs_padded, labels

def main():
    parser = argparse.ArgumentParser(description="Train AST on FMA-Small")
    parser.add_argument("--processed_data_dir", default="data/processed")
    parser.add_argument("--metadata_file", default="data/raw/fma_small/tracks.csv")
    parser.add_argument("--label_field", default="track_genre_top")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.1, help="fraction of data to use for validation")
    parser.add_argument("--output_dir", default="models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    spec_augment = SpecAugment(time_mask_param = 20,
                              freq_mask_param = 10,
                              num_time_masks = 1,
                              num_freq_masks = 1)
    

    # dataset and split
    full_ds = AudioDataset(processed_data_dir=args.processed_data_dir,
                            metadata_file=args.metadata_file,
                            label_field=args.label_field, 
                            transform=spec_augment)
    n_val = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"), collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"), collate_fn=pad_collate)
    
    print(f"Train dataset size: {len(train_ds)}, Validation dataset size: {len(val_ds)}")

    # model, loss, optimizer
    model = AudioClassifier(num_labels=len(full_ds.label_map)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)

    # Reduce learning rate on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, 
                                                           patience=3, verbose=True)
    
    scaler = GradScaler("cuda", enabled= True)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    no_improve_epochs = 0
    patience = 3
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        scheduler.step(val_loss)

        # Early-stopping logic based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            print(f"   Val loss improved to {val_loss:.4f}, resetting no_improve_epochs.")
        else:
            no_improve_epochs += 1
            print(f"   No improvement in val loss for {no_improve_epochs} epoch(s).")
            if no_improve_epochs >= patience:
                print(f"Stopping early at epoch {epoch} (no improvement for {patience} epochs).")
                break

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, "best_ast_fma_small.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path}")
        
    
    final_path = os.path.join(args.output_dir, "final_ast_fma_small.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training Complete! Saved final model to {final_path}")

if __name__ == "__main__":
    main()


