import torch
import os

def save_checkpoint(state, is_valid, valid_path, no_validation_path):
    if is_valid:
        torch.save(state, valid_path)
        print(f"Checkpoint salvato in {valid_path}")
    else:
        torch.save(state, no_validation_path)
        print(f"Checkpoint non valido salvato in {no_validation_path}")


class EarlyStopping:
    def __init__(self, patience=5, loss_threshold=None):
        self.patience = patience
        self.loss_threshold = loss_threshold  # Aggiunge il parametro di soglia di perdita
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss, model):
        # Controlla se la loss è inferiore alla soglia (se è specificata)
        if self.loss_threshold is not None and loss < self.loss_threshold:
            print(f"Loss {loss:.4f} inferiore alla soglia {self.loss_threshold}. Interrompo l'addestramento.")
            self.early_stop = True
            return
        
        # Logica di early stopping 
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0



            
