from PIL import Image as im
import numpy as np
import os


class MNIST:

    @staticmethod
    def load_images(file, n_max=None):
        images = []
        f = open(file, 'rb')
        magic_number = int.from_bytes(f.read(4), 'big')
        n_images = int.from_bytes(f.read(4), 'big')
        if n_max:
            n_images = n_max
        n_rows = int.from_bytes(f.read(4), 'big')
        n_columns = int.from_bytes(f.read(4), 'big')
        for i in range(n_images):
            image = []
            for r in range(n_rows):
                row = []
                for c in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
        f.close()
        return images

    @staticmethod
    def load_labels(file, n_max=None):
        labels = []
        f = open(file, 'rb')
        magic_number = int.from_bytes(f.read(4), 'big')
        n_items = int.from_bytes(f.read(4), 'big')
        if n_max:
            n_items = n_max
        for i in range(n_items):
            label = int.from_bytes(f.read(1), 'big')
            labels.append(label)
        f.close()
        return labels



class Image:

    @staticmethod
    def read_image(path):
        return np.asarray(im.open(path).convert('L'))

    @staticmethod
    def dataset_to_list(X):
        X1 = []
        for sample in X:
            new_list = []
            for row in sample:
                for pixel in row:
                    new_list.append(pixel)
            X1.append(new_list)
        return(X1)


class Files:

    @staticmethod
    def read_images(path):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if '.png' in file:
                    files.append(os.path.join(r, file))
        images = []
        for file in files:
            images.append(Image.read_image(file))
        return images

    @staticmethod
    def write_to_txt(output, output_path, input_path):
        output_files = []
        for r, d, f in os.walk(input_path):
            for file in f:
                if '.png' in file:
                    output_name = file.replace('.png', '.txt')
                    output_files.append(os.path.join(output_path, output_name))
        for i in range(len(output)):
            f = open(output_files[i], 'w')
            f.write(str(output[i]))
            f.close()

