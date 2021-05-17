
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
            label = f.read(1)
            labels.append(label)
        f.close()
        return labels

    @staticmethod
    def to_list(X):
        X1 = []
        for sample in X:
            new_list = []
            for row in sample:
                for pixel in row:
                    new_list.append(pixel)
            X1.append(new_list)
        return(X1)

