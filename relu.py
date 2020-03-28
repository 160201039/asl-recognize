import numpy


#reLU fonksiyonu eğer özellik matrisinde 0 dan küçük değer varsa 0 a sıfırdan büyük olanlar kendi değerine set edilir
def reLU(feature_maps):
    output = numpy.zeros(feature_maps.shape)
    for map_no in range(feature_maps.shape[2]):
        current_map=feature_maps[:,:,map_no]
        row=current_map.shape[0]
        col=current_map.shape[1]
        print(row," ",col)
        for r in range(row):
            for c in range(col):
                output[r,c,map_no]=numpy.max([current_map[r][c],0])
    return output

