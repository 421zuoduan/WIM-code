import os
from PIL import Image

base_dir = './datasets'
norain_dir = os.path.join(base_dir, 'norain')
rainy_dir = os.path.join(base_dir, 'rain')
output_dir = os.path.join(base_dir, 'Rain100L/train_c')

os.makedirs(output_dir, exist_ok=True)

for i in range(1, 201):
    # norain_filename = f'norain-{i:03d}.png'
    # rainy_filename = f'rain-{i:03d}.png'
    # output_filename = f'norain-{i:03d}.png'
    norain_filename = f'norain-{i}.png'
    rainy_filename = f'norain-{i}x2.png'
    output_filename = f'norain-{i}.png'

    norain_path = os.path.join(norain_dir, norain_filename)
    rainy_path = os.path.join(rainy_dir, rainy_filename)
    output_path = os.path.join(output_dir, output_filename)

    norain_image = Image.open(norain_path)
    rainy_image = Image.open(rainy_path)

    width, height = norain_image.size

    combined_image = Image.new('RGB', (width * 2, height))

    combined_image.paste(norain_image, (0, 0))
    combined_image.paste(rainy_image, (width, 0))

    combined_image.save(output_path)

    norain_image.close()
    rainy_image.close()

print("save to 'train_c' folder successfully!")
