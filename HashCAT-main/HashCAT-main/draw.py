import matplotlib.pyplot as plt

def drawloss(epoch_losses, avg_losses, aucs, accs):
    epochs = list(epoch_losses.keys())
    train_losses = list(epoch_losses.values())
    assert len(epochs) == len(avg_losses) == len(aucs) == len(accs), "列表长度不一致"
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(epochs, train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
    ax1.plot(epochs, avg_losses, marker='^', linestyle='--', color='r', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(epochs, aucs, marker='s', linestyle='-.', color='g', label='AUC')
    ax2.plot(epochs, accs, marker='x', linestyle=':', color='y', label='Accuracy')
    ax2.set_ylabel('AUC / Accuracy', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(0.5, 0.9)
    ax2.legend(loc='upper right')
    plt.title('Training and Validation Metrics per Epoch')
    ax1.grid(True)
    plt.tight_layout()
    plt.show()