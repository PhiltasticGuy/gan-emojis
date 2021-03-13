import os, logging

from PIL import Image

from matplotlib import pyplot

from numpy import asarray
from numpy import savez_compressed
from numpy import load

# https://stackoverflow.com/questions/35859140/remove-transparency-alpha-from-any-image-using-pil
def remove_transparency(im, bg_colour=(255, 255, 255)):
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im

def preprocess_emojis(emojis_directory, fixed_directory):
    # Make sure that the preprocessed images directory exists
    if (not os.path.exists(fixed_directory)):
        os.makedirs(fixed_directory)
    
    # Gather all the filenames in the input images directory
    files = [f for f in os.listdir(emojis_directory) if os.path.isfile(os.path.join(emojis_directory, f))]

    for filename in files:
        # Open the input image
        image = Image.open(f'{emojis_directory}/{filename}')

        # Resize down to 32x32 pixels
        image = image.resize((48, 48))

        # Apply data augmentation transformations
        data_augmentation(image, f'{fixed_directory}/{filename}')

        # Remove the transparency and replace alpha channel with 'white'
        image = remove_transparency(image)

        # Save the processed image
        image.save(f'{fixed_directory}/{filename}')

def data_augmentation(image, filename):
    # Disable the logging for Pillow
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    # Convert image to RGBA to prevent transformations from 
    # leaving a colored background
    if (image.mode != 'RGBA'):
        image = image.convert('RGBA')

    # NOTE:
    # Don't forget to call remove_transparency(...) to replace 
    # the alpha channel with 'white' before saving the image

    # Apply 15 degree rotation to the left
    filename = filename.replace('.png', '-da-nrot.png')
    remove_transparency(image.copy().rotate(-15)).save(filename)

    # Apply 15 degree rotation to the right
    filename = filename.replace('.png', '-da-nrot.png')
    remove_transparency(image.copy().rotate(15)).save(filename)

    # Apply 5 pixel translation to the left
    filename = filename.replace('.png', '-da-tl.png')
    remove_transparency(image.copy().transform(image.size, Image.AFFINE, (1, 0, -5, 0, 1, 0))).save(filename)

    # Apply 5 pixel translation to the right
    filename = filename.replace('.png', '-da-tr.png')
    remove_transparency(image.copy().transform(image.size, Image.AFFINE, (1, 0, 5, 0, 1, 0))).save(filename)

    # Apply 5 pixel translation up
    filename = filename.replace('.png', '-da-tu.png')
    remove_transparency(image.copy().transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, 5))).save(filename)

    # Apply 5 pixel translation down
    filename = filename.replace('.png', '-da-td.png')
    remove_transparency(image.copy().transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, -5))).save(filename)

    # Apply 5 pixel translation to the top left
    filename = filename.replace('.png', '-da-tlu.png')
    remove_transparency(image.copy().transform(image.size, Image.AFFINE, (1, 0, -5, 0, 1, -5))).save(filename)

    # Apply 5 pixel translation to the top right
    filename = filename.replace('.png', '-da-tru.png')
    remove_transparency(image.copy().transform(image.size, Image.AFFINE, (1, 0, 5, 0, 1, -5))).save(filename)

    # Apply 5 pixel translation to the bottom left
    filename = filename.replace('.png', '-da-tld.png')
    remove_transparency(image.copy().transform(image.size, Image.AFFINE, (1, 0, -5, 0, 1, -5))).save(filename)

    # Apply 5 pixel translation to the bottom right
    filename = filename.replace('.png', '-da-trd.png')
    remove_transparency(image.copy().transform(image.size, Image.AFFINE, (1, 0, 5, 0, 1, -5))).save(filename)

    # Only flip images that aren't symetrical.
    # filename = f'{self.path}/{emoji["id"]}-{emoji["name"]}-{emojiType}-da-flip.png'
    # image.copy().transpose(Image.FLIP_LEFT_RIGHT).save(filename)

    # Also apply all of the data augmentation on the flipped version.
    # ...

def load_emojis(directory):
    emojis = list()

    # Gather all the filenames in the preprocessed images directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    for filename in files:
        # Open the input image
        image = Image.open(f'{directory}/{filename}')

        # Convert to 'RGB' mode
        image = image.convert('RGB')
        
        # Extract array of pixels from image
        pixels = asarray(image)

        # Append pixels to list of emojis
        emojis.append(pixels)

    return asarray(emojis)

def plot_emojis(emojis, n_grid_size):
    for i in range(n_grid_size * n_grid_size):
        pyplot.subplot(n_grid_size, n_grid_size, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(emojis[i])

    pyplot.show()
    pyplot.close()

# Apply preprocessing to the dataset
dir_emojis = 'emojis'
dir_emojis_fixed = 'emojis-fixed'
preprocess_emojis(dir_emojis, dir_emojis_fixed)

# Save the clean dataset into a compressed file
emojis = load_emojis(dir_emojis_fixed)
savez_compressed('emojis-dataset.npz', emojis)

# Test reloading the compressed dataset
data = load('emojis-dataset.npz')
emojis = data['arr_0']

# Print dataset details and first X images
print('Loaded: ', emojis.shape)
plot_emojis(emojis, 7)