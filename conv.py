import numpy
import sys
import matplotlib.pyplot as plt
import skimage

def conv(img, conv_filter):
    if len(img.shape) > 2 or len(conv_filter.shape) > 3:
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()
    if conv_filter.shape[1] != conv_filter.shape[2]:
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
        sys.exit()
    if conv_filter.shape[1]%2==0:
        print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
        sys.exit()

    # Convolution sonrası oluşan çıktı grayscale image (w*h) ---> w*h*<kernel sayısı> Her bir filtre için oluşan özellik matrisleri
    feature_maps = numpy.zeros((img.shape[0]-conv_filter.shape[1]+1,
                                img.shape[1]-conv_filter.shape[1]+1,
                                conv_filter.shape[0]))

    # Her bir kernel için konvolüsyon işlemini tekrarla grayscale için kernel 2*3*3 RGB için <kernel sayısı>*depth*w*h
    for kernel_no in range(conv_filter.shape[0]):
        print("Filtre ", kernel_no + 1)
        anlik_filtre = conv_filter[kernel_no, :]

        #--------------------RGB----------------------------------
        if len(anlik_filtre.shape) > 2:
            conv_map = convolve(img[:, :, 0], anlik_filtre[:, :, 0])
            for ch_num in range(1, anlik_filtre.shape[-1]):
                conv_map = conv_map + convolve(img[:, :, ch_num],
                                  anlik_filtre[:, :, ch_num])
        #----------------------------------------------------------
        else:
            conv_map = convolve(img, anlik_filtre)
        feature_maps[:, :, kernel_no] = conv_map
    return feature_maps


# Konvolüsyon İşlemi
def convolve(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = numpy.zeros((img.shape))
    row=numpy.uint16(numpy.arange(filter_size / 2.0,img.shape[0] - filter_size / 2.0 + 1))
    col=numpy.uint16(numpy.arange(filter_size / 2.0,img.shape[1] - filter_size / 2.0 + 1))

    for r in row:
        for c in col:

            receptive_f = img[r - numpy.uint16(numpy.floor(filter_size / 2.0)):r + numpy.uint16(
                numpy.ceil(filter_size / 2.0)),
                          c - numpy.uint16(numpy.floor(filter_size / 2.0)):c + numpy.uint16(
                              numpy.ceil(filter_size / 2.0))]
            #hadamard Product
            curr_result = receptive_f * conv_filter
            #Çarpım sonucu oluşan matris toplanır.
            conv_sum = numpy.sum(curr_result)
            #receptive field için oluşan konvolüsyon sonucu
            result[r, c] = conv_sum

    final_result = result[numpy.uint16(filter_size / 2.0):result.shape[0] - numpy.uint16(filter_size / 2.0),
                          numpy.uint16(filter_size / 2.0):result.shape[1] - numpy.uint16(filter_size / 2.0)]
    return final_result






def conv_layer(gray_image,filter):
    return conv(gray_image,filter)



