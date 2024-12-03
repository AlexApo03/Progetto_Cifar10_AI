import torch
import json
from model import get_resnet_model
from dataloader import get_dataloaders
from utils import EarlyStopping 
from train import train_and_evaluate  # Funzione principale per addestrare e valutare

if __name__ == '__main__':
    # Carica la configurazione dal file JSON
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Verifica se le configurazioni sono caricate correttamente
    print("Configurazioni caricate:", config)

    # Imposta i DataLoader
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

    # Verifica che i DataLoader siano stati inizializzati correttamente
    print(f"Numero di batch nel trainloader: {len(trainloader)}")
    print(f"Numero di batch nel testloader: {len(testloader)}")

    # Imposta il modello, la perdita, l'ottimizzatore, l'early stopping e il dispositivo
    model = get_resnet_model(num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Stampa per verificare che il modello sia caricato sul dispositivo giusto
    print(f"Il modello è stato caricato su {device}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    early_stopping = EarlyStopping(patience=config['early_stop_patience'], loss_threshold=config['loss_threshold'])

    # Verifica che early_stopping sia stato inizializzato correttamente
    print(f"Early stopping configurato con pazienza di {config['early_stop_patience']} e soglia di loss di {config['loss_threshold']}")

    # Avvia il training
    train_and_evaluate(model, trainloader, testloader, criterion, optimizer, config, early_stopping, device)
