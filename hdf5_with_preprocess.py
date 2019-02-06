import h5py, os
import caffe
import numpy as np
import scipy.ndimage as ndimage
import scipy.misc
import matplotlib.pyplot as plt
import skimage.transform

# ?????????????????????????????? 224 or 227?
IMG_RESHAPE = 224
# IMG_RESHAPE = 227
IMG_UNCROPPED = 256

def resize_convert(img_names, scores_raw, path = None, if_oversample = False):
    '''
    Load images, set to BGR mode and transpose to CxHxW
    and subtract the Imagenet mean. If if_oversample is True, 
    perform data augmentation.

    Parameters:
    ---------
    img_names (list): list of image names to be processed.
    path (string): path to images.
    if_oversample (bool): if True then data augmentation is performed
        on each image, and 10 crops of size 224x224 are produced 
        from each image. If False, then a single 224x224 is produced.
    '''

    path = path if path is not None else ''
    if if_oversample == False:
        all_imgs = np.empty((len(img_names), 3, IMG_RESHAPE, IMG_RESHAPE), dtype='float32')
    else:
        all_imgs = np.empty((len(img_names), 3, IMG_UNCROPPED, IMG_UNCROPPED), dtype='float32')

    #load the imagenet mean
    mean_val = np.load('../ilsvrc12_data/ilsvrc_2012_mean.npy')

    for i, img_name in enumerate(img_names):
        # Read as HxWxC, RGB, range 0-255
        # Caffe needs CxHxW, BGR, range 0-255
        img = ndimage.imread(path+img_name, mode='RGB') 

        #subtract the mean of Imagenet
        #First, resize to 256 so we can subtract the mean of dims 256x256 
        img = img[...,::-1] #Convert RGB TO BGR

        # img = caffe.io.resize_image(img, (IMG_UNCROPPED, IMG_UNCROPPED, 3), interp_order=1)
        # img = scipy.misc.imresize(img, (IMG_UNCROPPED, IMG_UNCROPPED, 3), interp='bilinear')
        img = skimage.transform.resize(img, (IMG_UNCROPPED, IMG_UNCROPPED, 3), 
                order=1, preserve_range=True)
        # print(img.dtype)
        # print('Range of images: max: ' + str(np.amax(img[:,:,0])) + ' ; min: ' + str(np.amin(img[:,:,0])) )
        # print('Range of images: max: ' + str(np.amax(img[:,:,1])) + ' ; min: ' + str(np.amin(img[:,:,1])) )
        # print('Range of images: max: ' + str(np.amax(img[:,:,2])) + ' ; min: ' + str(np.amin(img[:,:,2])) )

        # print('Reshaped shape of img is:' + str(img.shape))
        img = np.transpose(img, (2, 0, 1))  #HxWxC => CxHxW 

        # Since mean is given in Caffe channel order: 3xWxH
        # shape of mean_val is:(3, 256, 256)
        # Assume it also is given in BGR order
        img = img.astype('float32')
        img = img - mean_val

        #set to 0-1 range => I don't think googleNet requires this
        #I tried both and it didn't make a difference
        #img = img/255

        #resize images down since GoogleNet accepts 224x224 crops
        if if_oversample == False:
            img = np.transpose(img, (1,2,0))  # CxHxW => HxWxC 
            # img = caffe.io.resiz_eimage(img, (IMG_RESHAPE, IMG_RESHAPE), interp_order=1)
            # img = scipy.misc.imresize(img, (IMG_RESHAPE, IMG_RESHAPE, 3), interp='bilinear')
            img = skimage.transform.resize(img, (IMG_RESHAPE, IMG_RESHAPE, 3), 
                    order=1, preserve_range=True)
            #skimage.transform.resize changes the dtype to float64
            img = img.astype('float32')
            img = np.transpose(img, (2,0,1)) #convert to CxHxW for Caffe

        all_imgs[i, :, :, :] = img
    # plt.imshow(np.transpose(all_imgs[0,:,:,:] + mean_val, (1,2,0))/255)
    # plt.show()

    # oversampling requires HxWxC order (from CxHxW)
    if if_oversample:
        all_imgs = np.transpose(all_imgs, (0, 2, 3, 1))
        # caffe.io.oversample get 4 corners, 1 center, and their mirrors
        # https://github.com/BVLC/caffe/blob/master/python/caffe/io.py
        # all_imgs = caffe.io.oversample(all_imgs, (IMG_RESHAPE, IMG_RESHAPE))
        all_imgs = oversample(all_imgs, (IMG_RESHAPE, IMG_RESHAPE))
        all_imgs = np.transpose(all_imgs, (0,3,1,2)) #convert to CxHxW for Caffe 
        # repeat a column vector
        scores = np.repeat(scores_raw, 10, axis = 0)

    else:
        scores = scores_raw

    return [all_imgs, scores]

def creat_hdf5(file_name, suffix = '', if_oversample = False):
    with open('./output/' + file_name + '.txt', 'r') as data_txt:
        lines = data_txt.readlines()
    # If you do not have enough memory split data into
    # multiple batches and generate multiple separate h5 files
    img_names = [None] * len(lines)
    scores_raw = np.zeros((len(lines), 1), dtype='f4')
    for i,l in enumerate(lines):
        sp = l.split(' ')
        img_names[i] = sp[0]
        scores_raw[i, 0] = float(sp[1])

    # Switch if_oversample to True if want data augmentation!
    # [images, scores] = resize_convert(img_names = img_names, 
    #                         scores_raw = scores_raw, if_oversample)
    [images, scores] = resize_convert(img_names = img_names, 
                            scores_raw = scores_raw, 
                            if_oversample = if_oversample)

    with h5py.File('../poshmark_734_data/hdf5_patch_224/' + file_name + suffix \
            + '.h5','w') as h5_file:
        h5_file.create_dataset('images', data = images) # note the name given to the dataset!
        h5_file.create_dataset('scores', data = scores) # note the name given to the dataset!
    with open('../poshmark_734_data/hdf5_patch_224/' + file_name + suffix \
            + '_h5_list.txt','w') as h5_list:
        h5_list.write('./poshmark_734_data/hdf5_patch_224/' + file_name \
                + suffix + '.h5') # list all h5 files you are going to use

def oversample(images, crop_dims):
    """
    Crop images into the four corners, center, and their mirrored versions.

    Parameters
    ----------
    image : iterable of (H x W x K) ndarrays
    crop_dims : (height, width) tuple for the crops.

    Returns
    -------
    crops : (10*N x H x W x K) ndarray of crops for number of inputs N.
    """
    # Dimensions and center.
    im_shape = np.array(images[0].shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    # to make crops_ix a 10x4 array
    crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    print ('len(images) = ' + str(len(images)))
    crops = np.empty((10 * len(images), crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)
    ix = 0
    for i, im in enumerate(images):
        for crop in crops_ix:
            crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
            ix += 1
        print('image idx = ' + str(i))
        print('ix = ' + str(ix))

        crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]  # flip for mirrors
    return crops

def main():
    # # the input file name matches the output file name
    # creat_hdf5(file_name = 'train', suffix = '_no_oversample')
    # creat_hdf5(file_name = 'test', suffix = '_no_oversample')

    for i in range(5):
        creat_hdf5(file_name = 'train_' + str(i), 
                suffix = '_mirror_oversample_224',
                if_oversample = True)
        creat_hdf5(file_name = 'test_' + str(i), 
                suffix = '_mirror_oversample_224',
                if_oversample = True)

if __name__ == '__main__':
    main()