import torch
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint
import os

def train_and_evaluate(model, trainloader, testloader, criterion, optimizer, config, early_stopping, device):
    print("L'Addestramento sta iniziando...", flush=True)
    best_loss = float('inf')

    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {epoch_loss:.4f}", flush=True)


        # Salva il modello se la loss migliora
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, True, config['checkpoint_path_valid'], config['checkpoint_path_no_validation'])  # Salva in valid
        else:
            # Salva il modello non valido se la loss non migliora
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, False, config['checkpoint_path_valid'], config['checkpoint_path_no_validation'])  # Salva in no_validation

        # Early stopping
        early_stopping(epoch_loss, model)  # Passa il modello per eventuali salvataggi
        if early_stopping.early_stop:
            print("Early stopping")
            # Salva l'ultimo modello al termine dell'early stopping
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, False, config['checkpoint_path_valid'], config['checkpoint_path_no_validation'])  # Salva in no_validation
            print(f"Checkpoint non valido salvato in {config['checkpoint_path_no_validation']} a causa dell'early stopping")
            break

            