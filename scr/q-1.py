import matplotlib.image as img
import numpy as np
import matplotlib.pyplot as plt

def set_filter(order, scale = 1.0):
    sd = scale/np.sqrt(np.prod(order))
    np.random.seed(26)
    return np.random.normal(loc = 0, scale = sd, size = order)

def set_weights(order):
    return np.random.standard_normal(size=order) * 0.01

def softmax(X):
    exps = np.exp(X)
    np.random.seed(26)
    return exps/np.sum(exps)

def zero_padding(x, pad):
    x_pad = np.pad(x,((pad,pad),(pad,pad),(0,0)),'constant',constant_values=(0))
    print('\n\nDemonstration of Padding Capability !!! (If intended for use)')
    print ('\nShape of input image = ', x.shape)
    print ("Shape of padded image =", x_pad.shape)
    fig, axarr = plt.subplots(1, 2)
    axarr[0].set_title('Input Image Before Zero padding')
    axarr[0].imshow(x)
    axarr[1].set_title('Image After Zero padding')
    axarr[1].imshow(x_pad)
    print('\nPadding is complete !')
    
    return x_pad


def conv_layer(x, filtr, bias, stride=1):
  
    (n_f, f_channel, f, f) = filtr.shape
    img_channel, img_dim, img_dim = x.shape
    
    conv_dim = int((img_dim - f)/stride)+1
    
    if (img_channel != f_channel):
        print("Number of channels in filter does not match with number of channels in image")
    
    convolved = np.zeros((n_f,conv_dim,conv_dim))
    
    for f_i in range(n_f):   #over number of filters
        y_i = conv_y = 0
        while y_i + f <= img_dim:
            x_i = conv_x = 0
            while x_i + f <= img_dim:
            	convolved[f_i, conv_y, conv_x] = np.sum(filtr[f_i] * x[:,y_i:y_i+f, x_i:x_i+f]) + bias[f_i]
            	x_i += stride
            	conv_x += 1
            y_i += stride
            conv_y += 1
     
    return convolved

def pool_layer(x, f=2, stride=2):
    
    img_channel, y_prev, x_prev = x.shape
    
    y_dim = int((y_prev - f)/stride)+1
    x_dim = int((x_prev - f)/stride)+1
    
    pooled = np.zeros((img_channel, y_dim, x_dim))
    for i in range(img_channel):
        y_i = conv_y = 0
        while y_i + f <= y_prev:
            x_i = conv_x = 0
            while x_i + f <= x_prev:
                pooled[i, conv_y, conv_x] = np.max(x[i, y_i:y_i+f, x_i:x_i+f])
                x_i += stride
                conv_x += 1
            y_i += stride
            conv_y += 1
    
    return pooled

def cnn_forward(x, y, parameters, conv_stride, pool_f_size, pool_stride):

    [c1, c3, c5, f6, outl, b1, b3, b5, b6, b_out] = parameters

#-------------------------------------------------First Convolution Layer----------------------------------------
    c1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    b1 = np.array([0,0,0])
    conv1 = conv_layer(x, c1, b1, conv_stride)
    print('\nConv1 o/p: ',conv1.shape )   
    conv1[conv1<=0] = 0 # ReLU
    # conv1 = np.tanh(conv1) # Tanh

    fig, conv1_disp = plt.subplots(2, 3)
    ch = 0
    for i in range(2):															
        for j in range(3):
            conv1_disp[i,j].set_title('\n\n1st Conv layer : Channel {}'.format(ch))
            conv1_disp[i,j].imshow(conv1.T[:,:,ch])
            ch+=1
    plt.subplots_adjust(hspace=0.5)
#---------------------------------------------------------------------------------------------------------------

#-------------------------------------------------First Pooling Layer-------------------------------------------
    pool1 = pool_layer(conv1, pool_f_size, pool_stride) 
    print('\npool1 o/p: ',pool1.shape) 
    (nf2,dim2, _) = pool1.shape
    fc = pool1.reshape((nf2 * dim2 * dim2, 1))

    fig, pool1_disp = plt.subplots(2, 3)
    ch = 0
    for i in range(2):
        for j in range(3):
            pool1_disp[i,j].set_title('1st Pool layer : Channel {}'.format(ch))
            pool1_disp[i,j].imshow(pool1.T[:,:,ch])
            ch+=1
    plt.subplots_adjust(hspace=0.5)
#---------------------------------------------------------------------------------------------------------------
  
#-------------------------------------------------Second Convolution Layer-------------------------------------- 
    conv2 = conv_layer(pool1, c3, b3, conv_stride)
    print('\nConv2 o/p: ',conv2.shape ) 
    conv1[conv1<=0] = 0 # ReLU
    # conv1 = np.tanh(conv1) # Tanh

    fig, conv2_disp = plt.subplots(3, 4)
    ch = 0
    for i in range(3):
        for j in range(4):
            conv2_disp[i,j].set_title('2nd Conv layer : Channel {}'.format(ch))
            conv2_disp[i,j].imshow(conv2.T[:,:,ch])
            ch+=1
    plt.subplots_adjust(hspace=0.5)
#---------------------------------------------------------------------------------------------------------------

   
#-------------------------------------------------Second Pooling Layer------------------------------------------
    pool2 = pool_layer(conv2, pool_f_size, pool_stride)
    print('\npool2 o/p: ',pool2.shape) 
    (nf2,dim2, _) = pool2.shape
    fc = pool2.reshape((nf2 * dim2 * dim2, 1))

    fig, pool2_disp = plt.subplots(3, 4)
    ch = 0
    for i in range(3):
        for j in range(4):
            pool2_disp[i,j].set_title('2nd Pool layer : Channel {}'.format(ch))
            pool2_disp[i,j].imshow(pool2.T[:,:,ch])
            ch+=1
    plt.subplots_adjust(hspace=0.5)
#---------------------------------------------------------------------------------------------------------------


    print('\nData sent to fully connected layer !!!')
#-------------------------------------------------First Hidden Layer in NN--------------------------------------   
    z = c5.dot(fc) + b5
    z[z<=0] = 0
#-------------------------------------------------Second Hidden Layer in NN-------------------------------------
    z = f6.dot(z) + b6
    z[z<=0] = 0
#-------------------------------------------------Output Layer in NN--------------------------------------------   
    out = outl.dot(z) + b_out
#-------------------------------------------------Softmax for output Layer in NN--------------------------------
    probs = softmax(out)
    print('\nOutput Probabilities: ',probs)


if __name__ == "__main__":
    img = img.imread('../input_data/automobile10.png')
    x= img.T
    print("\n\nImage Reading Complete !\nThe shape of read image is: {}".format(x.shape))
    y = 1 #Sample label, say 1 for Car

    pad = 2
    channels = 3
    n_f1 = 6 # Number of filters in first conv layer
    n_f2 = 16  #Number of filters in second conv layer
    f_size = 5 # Size of each filter in conv layer 
    conv_stride = 1 # Stride for convolution filter
    pool_f_size = 2 # Size of ooling filter 
    pool_stride = 2 # Stride for pooling filter

#------------------------------------Demonstration of Padding (Extra)---------------------------------------------------

    x_pad = zero_padding(img, pad)

#------------------------------------Modelling CNN based on given parameters--------------------------------------------

    c1, c3, c5, f6, outl = (n_f1 ,channels,f_size,f_size), (n_f2 ,n_f1,f_size,f_size), (120,400),(84,120), (10, 84)
    c1 = set_filter(c1)
    c3 = set_filter(c3)
    c5 = set_weights(c5)
    f6 = set_weights(f6)
    outl = set_weights(outl)

    b1 = np.zeros((c1.shape[0],1))
    b3 = np.zeros((c3.shape[0],1))
    b5 = np.zeros((c5.shape[0],1))
    b6 = np.zeros((f6.shape[0],1))
    b_out = np.zeros((outl.shape[0],1))

    parameters = [c1, c3, c5, f6, outl, b1, b3, b5, b6, b_out]

    cnn_forward(x, y, parameters, conv_stride, pool_f_size, pool_stride)
    plt.show()