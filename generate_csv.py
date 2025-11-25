import os
import pandas as pd

dataset_dir = 'archive/images/Images'
data = []

for breed_folder in os.listdir(dataset_dir):
    breed_path = os.path.join(dataset_dir, breed_folder)
    if os.path.isdir(breed_path):
        breed = breed_folder.split('-', 1)[1]
        for img_file in os.listdir(breed_path):
            if img_file.endswith('jpg'):
                img_path = os.path.join(breed_path, img_file)
                data.append({'image_path': img_path, 'breed': breed})

df = pd.DataFrame(data)
df.to_csv('dog_breeds.csv', index = False)
print('CSV created: dog_breeds.csv')

