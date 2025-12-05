# training/main_federated.py
import torch
from torch.utils.data import DataLoader

from dataset import get_dataloaders
from model import EfficientNetLSTM
from federated import split_dataset, train_one_client, fed_avg, evaluate


def main():
    # --------------------------
    # CONFIG
    # --------------------------
    NUM_CLIENTS = 3
    GLOBAL_ROUNDS = 15      # increased for better learning
    LOCAL_EPOCHS = 3        # train more on each client
    BATCH_SIZE = 16
    LR = 1e-4
    DATA_ROOT = "../data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------
    # DATA
    # --------------------------
    train_loader, val_loader, test_loader, train_ds = \
        get_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE)

    print("Training classes:", train_ds.classes)
    print("Class to idx:", train_ds.class_to_idx)
    # Usually: {'fake': 0, 'real': 1}

    client_subsets = split_dataset(train_ds, NUM_CLIENTS)
    client_loaders = [
        DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
        for subset in client_subsets
    ]
    client_sizes = [len(subset) for subset in client_subsets]
    total_size = sum(client_sizes)
    client_weights = [size / total_size for size in client_sizes]

    # --------------------------
    # MODEL
    # --------------------------
    global_model = EfficientNetLSTM(num_classes=2)
    global_model.to(device)

    # --------------------------
    # FEDERATED ROUNDS
    # --------------------------
    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n--- Global Round {round_idx}/{GLOBAL_ROUNDS} ---")

        state_dicts = []
        for i, loader in enumerate(client_loaders):
            print(f" Client {i+1}/{NUM_CLIENTS} local training...")
            updated_state = train_one_client(
                global_model, loader, device,
                epochs=LOCAL_EPOCHS, lr=LR
            )
            state_dicts.append(updated_state)

        new_global_state = fed_avg(state_dicts, client_weights)
        global_model.load_state_dict(new_global_state)

        val_acc = evaluate(global_model, val_loader, device)
        print(f" Validation accuracy after round {round_idx}: {val_acc:.4f}")

    # --------------------------
    # FINAL TEST
    # --------------------------
    test_acc = evaluate(global_model, test_loader, device)
    print(f"\nFinal TEST accuracy: {test_acc:.4f}")

    # --------------------------
    # SAVE MODEL
    # --------------------------
    torch.save(global_model.state_dict(), "model.pth")
    print("Model saved successfully to training/model.pth")


if __name__ == "__main__":
    main()
