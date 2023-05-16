import matplotlib.pyplot as plt

def load_data(filename):
    with open(filename, 'r') as f:
        data = [float(line.strip()) for line in f]
    return data

def plot_data(train_accuracy, test_accuracy, train_loss, test_loss):
    epochs = range(1, len(train_accuracy) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy, color='magenta', label='Train Accuracy', linestyle='-')
    plt.plot(epochs, test_accuracy, color='blue', label='Test Accuracy', linestyle='-')
    plt.title('DenseNet121 Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, color='magenta', label='Train Loss', linestyle='-')
    plt.plot(epochs, test_loss, color='blue', label='Test Loss', linestyle='-')
    plt.title('DenseNet121 Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    train_accuracy_file = 'CNN_train_accuracy.txt'
    valid_accuracy_file = 'CNN_validation_accuracy.txt'
    train_loss_file = 'CNN_train_loss.txt'
    valid_loss_file = 'CNN_validation_loss.txt'

    train_accuracy = load_data(train_accuracy_file)
    valid_accuracy = load_data(valid_accuracy_file)
    train_loss = load_data(train_loss_file)
    valid_loss = load_data(valid_loss_file)

    plot_data(train_accuracy, valid_accuracy, train_loss, valid_loss)

if __name__ == "__main__":
    main()
