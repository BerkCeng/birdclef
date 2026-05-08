import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def plot_history(history, save_path='outputs/egitim_gecmisi.png'):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Eğitim Kaybı')
    plt.title('Kayıp (Loss) Geçmişi')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()

    if 'train_acc' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Eğitim Doğruluğu')
        plt.title('Doğruluk (Accuracy) Geçmişi')
        plt.xlabel('Epoch')
        plt.ylabel('Doğruluk (%)')
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_cm(y_true, y_pred, classes, save_path='outputs/karmasiklik_matrisi.png'):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = len(classes)

    size = max(12, min(num_classes // 5, 50))
    plt.subplots(figsize=(size, size))

    sns.heatmap(
        cm, annot=False, cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        cbar=True, square=True
    )

    plt.title(f'Karmaşıklık Matrisi ({num_classes} Sınıf)', fontsize=30, pad=30)
    plt.ylabel('Gerçek Sınıf', fontsize=24)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=24)

    label_fontsize = max(2, min(10, 1000 / num_classes))
    plt.xticks(rotation=90, fontsize=label_fontsize)
    plt.yticks(fontsize=label_fontsize)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Karmaşıklık matrisi kaydedildi: {save_path}")


def plot_top_confusions(y_true, y_pred, classes, save_path='outputs/en_cok_karisan_siniflar.png', top_n=30):
    cm = confusion_matrix(y_true, y_pred)

    errors = cm.copy()
    np.fill_diagonal(errors, 0)

    class_errors = errors.sum(axis=1) + errors.sum(axis=0)
    top_indices = np.argsort(class_errors)[-top_n:]

    sub_cm = cm[np.ix_(top_indices, top_indices)]
    sub_classes = [classes[i] for i in top_indices]

    plt.figure(figsize=(18, 14))
    sns.heatmap(
        sub_cm, annot=True, fmt='d', cmap='Reds',
        xticklabels=sub_classes, yticklabels=sub_classes,
        annot_kws={"size": 10}
    )

    plt.title(f'En Çok Karıştırılan {top_n} Sınıf', fontsize=18)
    plt.ylabel('Gerçek Sınıf', fontsize=14)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def save_classification_report(y_true, y_pred, classes, save_path='outputs/siniflandirma_raporu.txt'):
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    with open(save_path, 'w') as f:
        f.write("Sınıflandırma Raporu\n")
        f.write(report)


def print_metrics(y_true, y_pred, classes):
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
