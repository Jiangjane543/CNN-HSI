from scipy import io as spio
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors

size = 12


def input_data(DATA_DIR, key):
    data = spio.loadmat(DATA_DIR)
    return data[key]


def colormap(kinds):
    cdict = ['#000000', '#FFDEAD', '#FFC0CB', '#FF1493', '#DC143C', '#FFD700', '#DAA520','#D2691E', '#FF4500', '#00FA9A', '#00BFFF', '#6495ED', '#9932CC', '#8B008B','#228B22', '#000080', '#808080']
    cdict = cdict[:kinds]
    return colors.ListedColormap(cdict, 'indexed')


def dis_groundtruth(gt, title):
    kinds = np.unique(gt).shape[0]
    plt.imshow(gt, cmap=colormap(kinds))
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0.1, right=1, left=0.125, hspace=0, wspace=0)
    plt.margins(0, 0)


class Hyperimage(object):
    def __init__(self, data, label):
        self.Weight = data.shape[0]
        self.Height = data.shape[1]
        self.Bands = data.shape[2]
        self.Kinds = np.unique(label).shape[0]
        self.data = data
        self.label = label

    def dis_Spectral(self, name):
        Spectral_value = np.zeros([self.Kinds - 1, self.Bands])
        for i in range(self.Kinds - 1):
            temp1 = (self.label == (i + 1))
            num = np.sum(temp1)
            for j in range(self.Bands):
                temp2 = (np.sum(np.multiply(self.data[:, :, j], temp1))) / num
                Spectral_value[i, j] = temp2
        plt.figure(name)
        plt.title('Spectral_distribution')
        for i in range(self.Kinds - 1):
            plt.plot(Spectral_value[i, :])
        plt.xlabel('Band')
        plt.ylabel('Spectral_value')
        plt.grid(True)
        plt.show()

    def spectrum_normlized(self):
        x = np.zeros(shape=self.data.shape, dtype='float64')
        for i in range(self.Bands):
            temp = self.data[:, :, i]
            mean = np.mean(temp)
            std = np.std(temp)
            x[:, :, i] = ((temp - mean) / std)
        return x

    def pca_reduction(self, k):
        normlizeddata = self.spectrum_normlized()
        data_reshape = np.reshape(normlizeddata, [-1, self.Bands])
        m = data_reshape.shape[0]
        sigma = np.dot(np.transpose(data_reshape), data_reshape) / m
        U, S, V = np.linalg.svd(sigma)
        U_reduce = U[:, 0:k]
        Z = np.dot(data_reshape, U_reduce)
        Z = Z.reshape([self.Weight, self.Height, k])
        return Z, U, S

    def pca_chosingK(self, k):
        _, _, S = self.pca_reduction(k)
        error = np.sum(S[0:k]) / np.sum(S[:])
        return error

    def batch_select(self):
        test_pixels = self.label.copy()
        n = 0
        for i in range(self.Kinds - 1):
            num = np.sum(self.label == (i + 1))
            n += num
            train_num = int(num*0.5)
            if train_num > 200:
                train_num = 200
            elif train_num < 10:
                train_num = 10
            print('第%d类有%d个样本，训练数：%d  测试数：%d' %(i+1,num,train_num,num-train_num))
            temp1 = np.where(self.label == (i + 1))  # get all the i samples which has num number
            temp2 = np.random.choice(num, train_num, replace=False)  # get random sequence
            for i in temp2:
                test_pixels[temp1[0][temp2], temp1[1][temp2]] = 0
        print('该数据集共有%d个样本\n'%(n))
        train_pixels = self.label - test_pixels
        return train_pixels, test_pixels


def pca_recover(Z, U, k):
    
    U_reduce = U[:, 0:k]
    X_rec = np.dot((Z, np.transpose(U_reduce)))
    return X_rec


def get_batchs(input_data, label_select):
    
    Band = input_data.shape[2]
    kind2 = np.unique(label_select).shape[0] - 1
    paddingdata = np.pad(input_data, ((size // 2, size // 2), (size // 2, size // 2), (0, 0)), "constant")
    paddinglabel = np.pad(label_select, ((size // 2, size // 2), (size // 2, size // 2)), "constant")
    pixel = np.where(paddinglabel != 0)
    num = np.sum(label_select != 0)
    batch_out = np.zeros([num, size, size, Band])
    batch_label = np.zeros([num, kind2])
    for i in range(num):
        row_start = pixel[0][i] - size // 2
        row_end = pixel[0][i] + size // 2
        col_start = pixel[1][i] - size // 2
        col_end = pixel[1][i] + size // 2
        batch_out[i, :, :, :] = paddingdata[row_start:row_end, col_start:col_end, :]
        temp = (paddinglabel[pixel[0][i], pixel[1][i]] - 1)
        batch_label[i, temp] = 1

    batch_out = batch_out.swapaxes(1, 3)
    batch_out = batch_out[:, :, :, :, np.newaxis]
    return batch_out, batch_label


def get_2dimbatches(input_data, label_select):
    pixel_num = np.sum(label_select != 0)
    kind = np.unique(label_select).shape[0] - 1
    size_2dim = 10
    reshape_band = pow(size_2dim, 2)
    batch_out = np.zeros([pixel_num, size_2dim, size_2dim])
    batch_label = np.zeros([pixel_num, kind])
    pixel = np.where(label_select != 0)

    for i in range(pixel_num):
        batch_out[i, :, :] = np.reshape(input_data[pixel[0][i], pixel[1][i], :reshape_band], (size_2dim, size_2dim))
        temp = (label_select[pixel[0][i], pixel[1][i]] - 1)
        batch_label[i, temp] = 1
    batch_out = batch_out.swapaxes(1, 2)
    batch_out = batch_out[:, :, :, np.newaxis]
    return batch_out, batch_label
