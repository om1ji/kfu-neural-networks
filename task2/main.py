from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torch.utils.data import Dataset
import pandas as pd
from torch.optim import Adam, SGD
from torch import nn
from torch.nn import ReLU, Sigmoid, Tanh
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, augment_factor=5):
        self.original_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.augment_factor = augment_factor
        
        # Создаем расширенный датафрейм с метками
        self.img_labels = pd.concat([self.original_labels] * augment_factor, ignore_index=True)
        
        # Определяем преобразования для аугментации
        self.augmentation = transforms.RandomAffine(
            degrees=(-10, 10),
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=(-3, 3),
            fill=(255)
        )
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        # Определяем, является ли это оригинальным или аугментированным изображением
        is_augmented = idx >= len(self.original_labels)
        
        # Получаем индекс оригинального изображения
        original_idx = idx % len(self.original_labels)
        
        image_path = os.path.join(self.img_dir, self.original_labels.iloc[original_idx, 0])
        image = Image.open(image_path).convert('L')
        image = image.resize((74, 75))
        
        # Применяем аугментацию только к дополнительным изображениям
        if is_augmented:
            image = self.augmentation(image)
            
        image = np.array(image) / 255.0
        image = 1 - image
        image = torch.FloatTensor(image)
        label = self.img_labels.iloc[idx, 1]
        return image, label

# Создаем датасеты
train_dataset = ImageDataset(
    annotations_file='annotations.csv', 
    img_dir='train-dataset',
    augment_factor=10  # Датасет будет в 10 раз больше исходного
)

# Для тестового набора не используем аугментацию
test_dataset = ImageDataset(
    annotations_file='annotations-test.csv', 
    img_dir='test-dataset',
    augment_factor=1  # Без аугментации
)

batch_size = 64 #размер пачки
n_iters = 10000 # количество итераций
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)
print(num_epochs)

# Обновляем создание загрузчиков данных
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Первый скрытый слой
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = ReLU()
        # Второй скрытый слой
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation2 = ReLU()
        # Выходной слой
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation1(out)
        out = self.fc2(out)
        out = self.activation2(out)
        out = self.fc3(out)
        return out

input_dim = 74*75
hidden_dim = 100
output_dim = 10

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

learning_rate = 0.0001
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.02)

#Loss function -- функция потерь
criterion = nn.CrossEntropyLoss()

train_losses = []
accuracies = []

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        batch_size = images.size(0)
        images = images.view(batch_size, input_dim)

        optimizer.zero_grad()
        outputs = model(images)
        
        # Добавляем вывод предсказаний и меток
        _, predicted = torch.max(outputs.data, 1)
        # print(f"{predicted[0].item()} -> {labels[0].item()}")
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        iter += 1
        
        train_losses.append(loss.item())

        if iter % 100 == 0:
            correct = 0
            total = 0
            # Итерация на тестовом наборе данных
            for images, labels in test_loader:
                images = images.view(-1, input_dim).requires_grad_()

                # вычисления выходного значения НС
                outputs = model(images)

                # находим прогнозированной номер картинки
                _, predicted = torch.max(outputs.data, 1)

                # общее количество образов
                total += labels.size(0)

                # количество корректно классифицированных образов
                correct += (predicted == labels).sum()
                
                accuracy = 100 * correct / total
                accuracies.append(accuracy)
                print(f"{accuracy}")
                print(f"{epoch}/{num_epochs}")
                if accuracy > 99:
                    break
            
plt.figure(figsize=(12, 4))

# График точности
plt.subplot(1, 2, 1)
plt.title('Точность на тестовом наборе')
plt.xlabel('Итерация')
plt.ylabel('Точность (%)')
plt.plot(accuracies)

# График ошибок
plt.subplot(1, 2, 2)
plt.title('Ошибка')
plt.ylabel('Ошибка (%)')
plt.plot(train_losses)

plt.show()


