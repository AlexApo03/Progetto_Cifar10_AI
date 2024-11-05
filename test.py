import torch 
import json
import argparse
from model import get_resnet_model
from dataloader import get_dataloaders
import torchvision.transforms as transforms

def main(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Setup dei DataLoader per il test
    train_batch_files = [
        './data/data_batch_1',
        './data/data_batch_2',
        './data/data_batch_3',
        './data/data_batch_4',
        './data/data_batch_5'
    ]
    
    test_batch_file = './data/test_batch'
    
    trainloader, testloader = get_dataloaders(
        batch_size=config['batch_size'],
        train_batch_files=train_batch_files,
        test_batch_file=test_batch_file
    )

    model = get_resnet_model(num_classes=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])  
        print(f"Checkpoint caricato da {checkpoint_path}")
    except FileNotFoundError:
        print(f"Errore: Il file di checkpoint '{checkpoint_path}' non è stato trovato.")
        return
    except KeyError:
        print("Errore: La chiave 'model_state_dict' non è stata trovata nel checkpoint.")
        return

    model = model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    """
    # Stampa il primo batch per controllare il formato dei dati
    for data in testloader:
        print("Data:", data)  # Mostra il contenuto del batch
        break  # Ferma qui per controllare solo il primo batch"""

    accuracy, test_loss = evaluate(model, testloader, criterion, device)
    print(f'Accuratezza sul set di test: {accuracy:.2f}%')
    print(f'Loss sul set di test: {test_loss:.3f}')


def evaluate(model, testloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for data in testloader:
            """
            print(f"Data output (size): {len(data)}")  # Stampa il numero di elementi nel batch
            print(f"Data output (content): {data}")  # Stampa il contenuto del batch
            """
            if isinstance(data, tuple) and len(data) == 2:  
                images, labels = data
            elif isinstance(data, list) and len(data) == 2:  # Controlla se data è una lista
                images, labels = data[0], data[1]
            else:
                print("Unexpected data format:", data)
                continue  

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total > 0:
        accuracy = 100 * correct / total
    else:
        accuracy = 0.0  

    avg_loss = running_loss / len(testloader) if len(testloader) > 0 else 0.0
    return accuracy, avg_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testa il modello ResNet18 su CIFAR-10')
    parser.add_argument('--config', type=str, default='config.json', help='Percorso al file di configurazione')
    parser.add_argument('--checkpoint', type=str, required=True, help='Percorso al file di checkpoint del modello')

    args = parser.parse_args()
    main(args.config, args.checkpoint)
