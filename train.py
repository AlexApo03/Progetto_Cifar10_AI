import torch
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint

# Mappatura delle etichette CIFAR-10
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def imshow(img):
    """Funzione per visualizzare l'immagine"""
    img = img / 2 + 0.5  # Inverti la normalizzazione per visualizzarla correttamente
    npimg = img.numpy()  # Converti l'immagine in un array numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Cambia la forma per matplotlib (C x H x W -> H x W x C)
    plt.show()

def train_and_evaluate(model, trainloader, testloader, criterion, optimizer, config, early_stopping, device):
    print("L'Addestramento sta iniziando...", flush=True)
    best_loss = float('inf')

    # Aggiungi TensorBoard per il logging delle immagini
    writer = SummaryWriter()  # Crea un oggetto SummaryWriter

    epoch = 0
    while not early_stopping.early_stop and epoch < config['num_epochs']:
        epoch += 1
        model.train()
        running_loss = 0.0
        
        print(f"Inizio Epoch {epoch}/{config['num_epochs']}")  # Debug: Stampa quando inizia una nuova epoca

        for batch_idx, (images, labels) in enumerate(trainloader):
            # Sposta i dati sul dispositivo
            images, labels = images.to(device), labels.to(device)

            # Verifica che il batch contenga dati
            if batch_idx == 0:  # Visualizza solo il primo batch di ogni epoca
                print(f"Batch {batch_idx}, Numero immagini nel batch: {len(images)}")  # Debug: Stampa numero di immagini nel batch

                for i in range(min(5, len(images))):  # Mostra al massimo 5 immagini per epoca
                    label_name = class_names[labels[i].item()]  # Converti l'etichetta in un nome
                    print(f"Epoch {epoch}, Immagine {i + 1}, Etichetta: {label_name}")
                    
                    # Visualizza l'immagine
                    imshow(images[i])

                    # Aggiungi l'immagine su TensorBoard
                    writer.add_image(f"Train Image {i + 1}", images[i], epoch)

            # Ottimizzazione
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch}/{config['num_epochs']}], Loss: {epoch_loss:.4f}", flush=True)

        # Aggiungi la loss su TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)

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
            print("Early stopping attivato!", flush=True)
            # Salva l'ultimo modello al termine dell'early stopping
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, False, config['checkpoint_path_valid'], config['checkpoint_path_no_validation'])  # Salva in no_validation
            print(f"Checkpoint non valido salvato in {config['checkpoint_path_no_validation']} a causa dell'early stopping")
            break

    # Chiudi il logger di TensorBoard
    writer.close()





