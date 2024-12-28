from PIL import Image
import os
import csv

image_path = 'hwd-test.jpg'
img = Image.open(image_path)

width, height = img.size

grid_x = 5  # количество частей по горизонтали
grid_y = 40  # количество частей по вертикали
OFFSET_Y = 10

tile_width = width // grid_x
tile_height = height // grid_y

output_dir = 'test-dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Создаем список меток для столбца Y
labels_y = []
for digit in range(10):  # цифры от 0 до 9
    labels_y.extend([digit] * 4)  # каждая цифра повторяется 4 раза

# Создаем CSV файл для аннотаций
csv_path = 'annotations-test.csv'
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_path', 'label'])

    for y in range(grid_y):
        for x in range(grid_x):
            left = x * tile_width
            upper = y * tile_height + OFFSET_Y
            right = left + tile_width
            lower = upper + tile_height
            
            tile = img.crop((left, upper, right, lower))
            image_filename = f'tile_{x}_{y}.png'
            tile.save(os.path.join(output_dir, image_filename))
            
            # Получаем метку из списка labels_y по индексу y
            label = labels_y[y]
            writer.writerow([image_filename, label])

print(f'Изображение успешно разделено на {grid_x * grid_y} частей')
print(f'Создан файл аннотаций: {csv_path}')