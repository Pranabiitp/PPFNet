#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
gpu=int(input("Which gpu number you would like to allocate:"))
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)


# In[2]:


# import zipfile

# # Replace 'my_folder.zip' with the name of your uploaded zip file
# with zipfile.ZipFile('/home/sriparna/Ashutosh_Tripathi/centralized/data(test excluded).zip', 'r') as zip_ref:
#     zip_ref.extractall('/home/sriparna/Ashutosh_Tripathi/centralized')


# In[14]:


from keras.preprocessing.image import ImageDataGenerator
def load_img_data1():
    img_size = (128,128)
    datagen = ImageDataGenerator()
    train_data = datagen.flow_from_directory(
        directory="client1",
        target_size=img_size,
        class_mode="categorical",
        batch_size=32,
    )
    
    return train_data


# In[15]:


from keras.preprocessing.image import ImageDataGenerator
def load_img_data2():
    img_size = (128,128)
    datagen = ImageDataGenerator()
    train_data = datagen.flow_from_directory(
        directory="client2",
        target_size=img_size,
        class_mode="categorical",
        batch_size=32,
    )
    
    return train_data


# In[16]:


def load_img_data3():
    img_size = (128,128)
    datagen = ImageDataGenerator()
    train_data = datagen.flow_from_directory(
        directory="client3",
        target_size=img_size,
        class_mode="categorical",
        batch_size=32,
    )
    
    return train_data


# In[17]:


train1=load_img_data1()
train2=load_img_data2()
train3=load_img_data3()


# In[18]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Reshape, MaxPooling2D, AveragePooling2D, concatenate, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ConvLSTM2D



# from keras.utils import plot_model
# plot_model(model1,show_shapes=True)

def box(lamda):
    IM_SIZE = 128
    r_x = int(np.random.uniform(0, IM_SIZE))

    r_y = int(np.random.uniform(0, IM_SIZE))

    r_w = IM_SIZE * np.sqrt(1 - lamda)
    r_h = IM_SIZE * np.sqrt(1 - lamda)

    r_x = np.clip(r_x - r_w // 2, 0, IM_SIZE)
    r_y = np.clip(r_y - r_h // 2, 0, IM_SIZE)

    x_b_r = np.clip(r_x + r_w // 2, 0, IM_SIZE)
    y_b_r = np.clip(r_y + r_h // 2, 0, IM_SIZE)

    r_w = y_b_r - r_y
    if (r_w == 0):
        r_w = 1
    r_h = y_b_r - r_y
    if (r_h == 0):
        r_h = 1

    return int(r_y), int(r_x), int(r_h), int(r_w)

def cutmix(image1,label1,images,labels):
    np.random.seed(None)
    index = np.random.permutation(len(images))
    lamda=stats.beta(0.4, 0.4).rvs()
    r_y,r_x,r_h,r_w=box(lamda)
    image2 = images[index[0]]
    label2 = labels[index[0]]
    crop2=tf.image.crop_to_bounding_box(image2,r_y,r_x,r_h,r_w)
    pad2=tf.image.pad_to_bounding_box(crop2,r_y,r_x,IM_SIZE,IM_SIZE)
    crop1=tf.image.crop_to_bounding_box(image1,r_y,r_x,r_h,r_w)
    pad1=tf.image.pad_to_bounding_box(crop1,r_y,r_x,IM_SIZE,IM_SIZE)
    image=image1-pad1+pad2
    lamda=1-(r_h*r_w)/(IM_SIZE*IM_SIZE)
    label=lamda*label1+(1-lamda)*label2
    return image,label


def mixup(image1, label1, images, labels):
    index = np.random.permutation(len(images))
    image2 = images[index[0]]

    label2 = labels[index[0]]
    lamda = np.random.beta(0.4, 0.4)

    label_1 = label1
    label_2 = label2
    image = lamda * image1 + (1 - lamda) * image2
    label = lamda * label_1 + (1 - lamda) * label_2

    return image, label

def cutout(images,labels, pad_size=16):
    cut_image=[]
    cut_labels=[]
    for index in tqdm(range(len(images))):
        img=images[index]
        h, w, c = img.shape
        mask = np.ones((h + pad_size*2, w + pad_size*2, c))
        y = np.random.randint(pad_size, h + pad_size)
        x = np.random.randint(pad_size, w + pad_size)
        y1 = np.clip(y - pad_size, 0, h + pad_size*2)
        y2 = np.clip(y + pad_size, 0, h + pad_size*2)
        x1 = np.clip(x - pad_size, 0, w + pad_size*2)
        x2 = np.clip(x + pad_size, 0, w + pad_size*2)
        mask[y1:y2, x1:x2, :] = 0
        img_cutout = img * mask[pad_size:pad_size+h, pad_size:pad_size+w, :]
        cut_image.append(img_cutout)
        cut_labels.append(labels[index])
    return cut_image,cut_labels


def random_augment(image1, label1, images, labels):
    datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )
    x = image1.reshape((1,) + image1.shape)
    my_images = []
    my_labels = []
    counter = 0
    for i in datagen.flow(x):
        if counter == 14:
            break
        my_image = i
        my_image = my_image.reshape((128, 128, 3))
        my_images.append(my_image)
        my_labels.append(label1)
        counter += 1

    return my_images, my_labels


def data_augmentation(normal_files, covid_files, pneumonia_files, tb_files, cat_files):
    aug_normal = []
    aug_covid = []
    thresh_hold = 7
    aug_pneumonia = []
    aug_tb = []
    aug_cat = []

    # x = tf.keras.preprocessing.image.load_img("/content/IM-0001-0001.jpeg")

    datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )
    # normal
    counter = 0
    for _ in range(2):
        for location in tqdm(normal_files):
            counter = 0

            x = Image.open(location)
            x = asarray(x)

            x = cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)
            x = x / 255.0

            x = x.reshape((1,) + x.shape)
            # x=x/255.0

            for i in datagen.flow(x):
                if counter >= 6:
                    break
                # i=i/255.0

                # i = cv2.resize(i,(224,224),interpolation = cv2.INTER_CUBIC)
                aug_normal.append(i)
                counter += 1
    # tb
    counter = 0

    for location in tqdm(tb_files):
        counter = 0

        x = Image.open(location)
        x = asarray(x)

        x = cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)
        x = x / 255.0

        x = x.reshape((1,) + x.shape)
        # x=x/255.0

        for i in datagen.flow(x):
            if counter >= 6:
                break
            # i=i/255.0

            # i = cv2.resize(i,(224,224),interpolation = cv2.INTER_CUBIC)
            aug_tb.append(i)
            counter += 1

    # covid
    counter = 0
    for location in tqdm(covid_files):
        counter = 0
        x = Image.open(location)
        x = asarray(x)

        x = cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)
        x = x / 255.0

        # x=img_to_array(x)
        x = x.reshape((1,) + x.shape)
        # x=x/255.0

        for i in datagen.flow(x):
            if counter >= 6:
                break

            aug_covid.append(i)
            counter += 1
            # pneumonia
    counter = 0
    for location in tqdm(pneumonia_files):
        counter = 0
        x = Image.open(location)
        x = asarray(x)

        x = cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)
        x = x / 255.0

        # x=img_to_array(x)
        x = x.reshape((1,) + x.shape)
        # x=x/255.0

        for i in datagen.flow(x):
            if counter >= 6:
                break
            # i=i/255.0
            # i = cv2.resize(i,(224,224),interpolation = cv2.INTER_CUBIC)
            aug_pneumonia.append(i)
            counter += 1

            # cat
    counter = 0
    for location in tqdm(cat_files):
        counter = 0
        x = Image.open(location)
        x = asarray(x)

        x = cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)
        x = x / 255.0

        # x=img_to_array(x)
        x = x.reshape((1,) + x.shape)
        # x=x/255.0

        for i in datagen.flow(x):
            if counter >= 6:
                break
            # i=i/255.0
            # i = cv2.resize(i,(224,224),interpolation = cv2.INTER_CUBIC)
            aug_cat.append(i)
            counter += 1

    for ele in normal_files:
        # ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)

        pic = cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)
        pic = pic / 255.0
        aug_normal.append(pic)
    for ele in covid_files:
        # ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)

        pic = cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)
        pic = pic / 255.0
        aug_covid.append(pic)
    for ele in pneumonia_files:
        # ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)

        pic = cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)
        pic = pic / 255.0
        aug_pneumonia.append(pic)
    for ele in tb_files:
        # ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)

        pic = cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)
        pic = pic / 255.0
        aug_tb.append(pic)
    for ele in cat_files:
        # ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)

        pic = cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)
        pic = pic / 255.0
        aug_cat.append(pic)

    for i in range(len(aug_normal)):
        aug_normal[i] = aug_normal[i].reshape((128, 128, 3))

    for i in range(len(aug_covid)):
        aug_covid[i] = aug_covid[i].reshape((128, 128, 3))
    for i in range(len(aug_pneumonia)):
        aug_pneumonia[i] = aug_pneumonia[i].reshape((128, 128, 3))
    for i in range(len(aug_tb)):
        aug_tb[i] = aug_tb[i].reshape((128, 128, 3))
    for i in range(len(aug_cat)):
        aug_cat[i] = aug_cat[i].reshape((128, 128, 3))

    # print("Normal files after augmentation:", len(aug_normal))
    # print("Covid files after augmentation:", len(aug_covid))
    # print("Pneumonia files after augmentation:", len(aug_pneumonia))
    # print("TB files after augmentation:", len(aug_tb))
    # print("Cat files after augmentation:", len(aug_cat))
    return aug_normal, aug_covid, aug_pneumonia, aug_tb, aug_cat


def advance_data_aug(images_list, images_labels, full_data, full_label, param=2):
    images_list = np.array(images_list)
    images_labels = np.array(images_labels)

    # create the original array
    arr = full_label

    # define the value of the element to delete
    value_to_delete = images_labels[0]
    index = []
    for i in range(len(arr)):
        if np.array_equal(arr[i], value_to_delete):
            index.append(i)
    my_full_data = []
    my_full_label = []
    for i in range(len(full_label)):
        if i in index:
            continue
        else:
            my_full_data.append(full_data[i])
            my_full_label.append(full_label[i])
    full_data = my_full_data.copy()
    full_label = my_full_label.copy()
    full_data = np.array(full_data)
    full_label = np.array(full_label)
    aug_list = []
    aug_labels = []
    print("adding original images")
    for i in range(len(images_list)):
        aug_labels.append(images_labels[i])
        aug_list.append(images_list[i])
    print(np.array(aug_list).shape, np.array(aug_labels).shape)
    print("cutmix")
    for i in range(2):
        for j in tqdm(range(len(images_list))):
            new_image, new_label = cutmix(images_list[j], images_labels[j], full_data, full_label)
            aug_labels.append(new_label)
            aug_list.append(new_image)
    print(np.array(aug_list).shape, np.array(aug_labels).shape)
    print("mixup")
    for i in range(2):
        for j in tqdm(range(len(images_list))):
            new_image, new_label = mixup(images_list[j], images_labels[j], full_data, full_label)
            aug_labels.append(new_label)
            aug_list.append(new_image)
    print(np.array(aug_list).shape, np.array(aug_labels).shape)
    print("random augmentation")
    for j in tqdm(range(len(images_list))):
        new_image, new_label = random_augment(images_list[j], images_labels[j], full_data, full_label)
        for index in range(len(new_image)):
            aug_labels.append(new_label[index])
            aug_list.append(new_image[index])
    print(np.array(aug_list).shape, np.array(aug_labels).shape)
    print("cutout")
    aug_list = np.array(aug_list)
    aug_labels = np.array(aug_labels)
    for i in range(2):
        im, la = cutout(images_list, images_labels)
        aug_list = np.concatenate([aug_list, im])
        aug_labels = np.concatenate([aug_labels, la])

    print(np.array(aug_list).shape, np.array(aug_labels).shape)
    return aug_list, aug_labels


# In[19]:


def making_full_data_train(aug_normal, aug_covid, aug_pneumonia, aug_tb, aug_cat):
    aug_normal = shuffle(aug_normal, random_state=0)
    aug_covid = shuffle(aug_covid, random_state=0)
    aug_pneumonia = shuffle(aug_pneumonia, random_state=0)
    aug_tb = shuffle(aug_tb, random_state=0)
    aug_cat = shuffle(aug_cat, random_state=0)

    aug_normal_labels = []
    for i in range(len(aug_normal)):
        aug_normal_labels.append(0)
    print(np.shape(aug_normal), np.shape(aug_normal_labels))
    aug_covid_labels = []
    for i in range(len(aug_covid)):
        aug_covid_labels.append(1)
    print(np.shape(aug_covid), np.shape(aug_covid_labels))
    aug_pneumonia_labels = []
    for i in range(len(aug_pneumonia)):
        aug_pneumonia_labels.append(2)
    print(np.shape(aug_pneumonia), np.shape(aug_pneumonia_labels))
    aug_tb_labels = []
    for i in range(len(aug_tb)):
        aug_tb_labels.append(3)
    print(np.shape(aug_tb), np.shape(aug_tb_labels))
    aug_cat_labels = []
    for i in range(len(aug_cat)):
        aug_cat_labels.append(4)
    print(np.shape(aug_cat), np.shape(aug_cat_labels))
    aug_normal_labels = on_hot_encode_labels(aug_normal_labels)
    aug_covid_labels = on_hot_encode_labels(aug_covid_labels)
    aug_pneumonia_labels = on_hot_encode_labels(aug_pneumonia_labels)
    aug_tb_labels = on_hot_encode_labels(aug_tb_labels)
    aug_cat_labels = on_hot_encode_labels(aug_cat_labels)
    full_data = []
    full_label = []
    for i in range(len(aug_normal)):
        full_data.append(aug_normal[i])
        full_label.append(aug_normal_labels[i])
    for i in range(len(aug_covid)):
        full_data.append(aug_covid[i])
        full_label.append(aug_covid_labels[i])
    for i in range(len(aug_pneumonia)):
        full_data.append(aug_pneumonia[i])
        full_label.append(aug_pneumonia_labels[i])
    for i in range(len(aug_tb)):
        full_data.append(aug_tb[i])
        full_label.append(aug_tb_labels[i])
    for i in range(len(aug_cat)):
        full_data.append(aug_cat[i])
        full_label.append(aug_cat_labels[i])
    aug_normal, aug_normal_labels = advance_data_aug(aug_normal, aug_normal_labels, full_data, full_label)
    aug_covid, aug_covid_labels = advance_data_aug(aug_covid, aug_covid_labels, full_data, full_label)
    aug_pneumonia, aug_pneumonia_labels = advance_data_aug(aug_pneumonia, aug_pneumonia_labels, full_data, full_label)
    aug_tb, aug_tb_labels = advance_data_aug(aug_tb, aug_tb_labels, full_data, full_label)
    aug_cat, aug_cat_labels = advance_data_aug(aug_cat, aug_cat_labels, full_data, full_label)

    full_data = []
    full_label = []
    for i in range(len(aug_normal)):
        full_data.append(aug_normal[i])
        full_label.append(aug_normal_labels[i])
    for i in range(len(aug_covid)):
        full_data.append(aug_covid[i])
        full_label.append(aug_covid_labels[i])
    for i in range(len(aug_pneumonia)):
        full_data.append(aug_pneumonia[i])
        full_label.append(aug_pneumonia_labels[i])
    for i in range(len(aug_tb)):
        full_data.append(aug_tb[i])
        full_label.append(aug_tb_labels[i])
    for i in range(len(aug_cat)):
        full_data.append(aug_cat[i])
        full_label.append(aug_cat_labels[i])

    full_data = np.array(full_data)
    full_label = np.array(full_label)

    full_data = shuffle(full_data, random_state=0)
    full_label = shuffle(full_label, random_state=0)

    return full_data, full_label


# In[20]:


import os
import glob
from sklearn.utils import shuffle


# In[21]:


normal_dir = "client1/im_Dyskeratotic" #give your normal cases data path here
#vit_datasets/Dataset_ViT/ViT_dataset/Covid-19
dir1 = os.path.join(normal_dir,"*.bmp")
dirc = os.path.join(normal_dir,"*.jpeg")
dir2 = os.path.join(normal_dir,"*.jpg")
normal_files1 = glob.glob(dirc)
normal_1 = glob.glob(dir1)
normal_2 = glob.glob(dir2)
normal_files1.extend(normal_1)
normal_files1.extend(normal_2)

normal_dir = "client1/im_Koilocytotic"  #give your covid 19 cases data path here
dir1 = os.path.join(normal_dir,"*.bmp")
dirc = os.path.join(normal_dir,"*.jpg")
dir2 = os.path.join(normal_dir,"*.jpeg")
covid_files1 = glob.glob(dirc)
covid_files2 = glob.glob(dir2)
covid_files1 = glob.glob(dir1)
covid_files1.extend(covid_files2)
covid_files1.extend(covid_files1)

normal_dir = "client1/im_Metaplastic" #give your pneumonia cases data path here
dir1 = os.path.join(normal_dir,"*.bmp")
dir2 = os.path.join(normal_dir,"*.jpeg")
dirc = os.path.join(normal_dir,"*.jpg")
pneumonia_files1 = glob.glob(dirc)
pneumonia_1 = glob.glob(dir1)
pneumonia_2 = glob.glob(dir2)
pneumonia_files1.extend(pneumonia_1)
pneumonia_files1.extend(pneumonia_2)

normal_dir = "client1/im_Parabasal" #give your pneumonia cases data path here
dir1 = os.path.join(normal_dir,"*.bmp")
dir2 = os.path.join(normal_dir,"*.jpeg")
dirc = os.path.join(normal_dir,"*.jpg")
tb_files1 = glob.glob(dirc)
tb_1 = glob.glob(dir1)
tb_2 = glob.glob(dir2)
tb_files1.extend(tb_1)
tb_files1.extend(tb_2)

normal_dir = "client1/im_Superficial-Intermediate" #give your pneumonia cases data path here
dir1 = os.path.join(normal_dir,"*.bmp")
dir2 = os.path.join(normal_dir,"*.jpeg")
dirc = os.path.join(normal_dir,"*.jpg")
cat_files1 = glob.glob(dirc)
cat_1 = glob.glob(dir1)
cat_2 = glob.glob(dir2)
cat_files1.extend(cat_1)
cat_files1.extend(cat_2)

normal_files1.sort()
covid_files1.sort()
pneumonia_files1.sort()
tb_files1.sort()
cat_files1.sort()
normal_files1 = shuffle(normal_files1, random_state=10)
covid_files1 = shuffle(covid_files1, random_state=10)
pneumonia_files1 = shuffle(pneumonia_files1, random_state=10)
tb_files1 = shuffle(tb_files1, random_state=10)
cat_files1 = shuffle(cat_files1, random_state=10)

print("pneumonia_files:", len(pneumonia_files1))
print("covid_files:", len(covid_files1))
print("normal_files:", len(normal_files1))
print("tb_files:", len(tb_files1))
print("cat_files:", len(cat_files1))

total_files1 = len(normal_files1) + len(covid_files1) + len(pneumonia_files1) + len(tb_files1) + len(cat_files1)

temp_files1 = []
temp_labels1 = []
for i in range(len(pneumonia_files1)):
    temp_files1.append(pneumonia_files1[i])
    temp_labels1.append(0)

for i in range(len(covid_files1)):
    temp_files1.append(covid_files1[i])
    temp_labels1.append(1)

for i in range(len(normal_files1)):
    temp_files1.append(normal_files1[i])
    temp_labels1.append(2)

for i in range(len(tb_files1)):
    temp_files1.append(tb_files1[i])
    temp_labels1.append(3)
    
for i in range(len(cat_files1)):
    temp_files1.append(cat_files1[i])
    temp_labels1.append(4)

temp_files1 = shuffle(temp_files1, random_state=10)
temp_labels1 = shuffle(temp_labels1, random_state=10)


# In[22]:


normal_dir = "client2/im_Dyskeratotic" #give your normal cases data path here
#vit_datasets/Dataset_ViT/ViT_dataset/Covid-19
dir1 = os.path.join(normal_dir,"*.bmp")
dir2 = os.path.join(normal_dir,"*.jpeg")
dirc = os.path.join(normal_dir,"*.jpg")
normal_files2 = glob.glob(dirc)
normal_1 = glob.glob(dir1)
normal_2 = glob.glob(dir2)
normal_files2.extend(normal_1)
normal_files2.extend(normal_2)

normal_dir = "client2/im_Koilocytotic"  #give your covid 19 cases data path here
dir1 = os.path.join(normal_dir,"*.bmp")
dirc = os.path.join(normal_dir,"*.jpg")
dir2 = os.path.join(normal_dir,"*.jpeg")
covid_files2 = glob.glob(dirc)
covid_files2 = glob.glob(dir2)
covid_files1 = glob.glob(dir1)
covid_files2.extend(covid_files2)
covid_files2.extend(covid_files1)

normal_dir = "client2/im_Metaplastic" #give your pneumonia cases data path here
dir1 = os.path.join(normal_dir,"*.bmp")
dir2 = os.path.join(normal_dir,"*.jpeg")
dirc = os.path.join(normal_dir,"*.jpg")
pneumonia_files2 = glob.glob(dirc)
pneumonia_1 = glob.glob(dir1)
pneumonia_2 = glob.glob(dir2)
pneumonia_files2.extend(pneumonia_1)
pneumonia_files2.extend(pneumonia_2)

normal_dir = "client2/im_Parabasal" #give your pneumonia cases data path here
dir1 = os.path.join(normal_dir,"*.bmp")
dir2 = os.path.join(normal_dir,"*.jpeg")
dirc = os.path.join(normal_dir,"*.jpg")
tb_files2 = glob.glob(dirc)
tb_1 = glob.glob(dir1)
tb_2 = glob.glob(dir2)
tb_files2.extend(tb_1)
tb_files2.extend(tb_2)

normal_dir = "client2/im_Superficial-Intermediate" #give your pneumonia cases data path here
dir1 = os.path.join(normal_dir,"*.bmp")
dir2 = os.path.join(normal_dir,"*.jpeg")
dirc = os.path.join(normal_dir,"*.jpg")
cat_files2 = glob.glob(dirc)
cat_1 = glob.glob(dir1)
cat_2 = glob.glob(dir2)
cat_files2.extend(cat_1)
cat_files2.extend(cat_2)

normal_files2.sort()
covid_files2.sort()
pneumonia_files2.sort()
tb_files2.sort()
cat_files2.sort()
normal_files2 = shuffle(normal_files2, random_state=10)
covid_files2 = shuffle(covid_files2, random_state=10)
pneumonia_files2 = shuffle(pneumonia_files2, random_state=10)
tb_files2 = shuffle(tb_files2, random_state=10)
cat_files2 = shuffle(cat_files2, random_state=10)

print("pneumonia_files:", len(pneumonia_files2))
print("covid_files:", len(covid_files2))
print("normal_files:", len(normal_files2))
print("tb_files:", len(tb_files2))
print("cat_files:", len(cat_files2))

total_files = len(normal_files2) + len(covid_files2) + len(pneumonia_files2) + len(tb_files2) + len(cat_files2)

temp_files2 = []
temp_labels2 = []
for i in range(len(pneumonia_files2)):
    temp_files2.append(pneumonia_files2[i])
    temp_labels2.append(0)

for i in range(len(covid_files2)):
    temp_files2.append(covid_files2[i])
    temp_labels2.append(1)

for i in range(len(normal_files2)):
    temp_files2.append(normal_files2[i])
    temp_labels2.append(2)

for i in range(len(tb_files2)):
    temp_files2.append(tb_files2[i])
    temp_labels2.append(3)
    
for i in range(len(cat_files2)):
    temp_files2.append(cat_files2[i])
    temp_labels2.append(4)

temp_files2 = shuffle(temp_files2, random_state=10)
temp_labels2 = shuffle(temp_labels2, random_state=10)


# In[23]:


normal_dir = "client3/im_Dyskeratotic" #give your normal cases data path here
#vit_datasets/Dataset_ViT/ViT_dataset/Covid-19
dir1 = os.path.join(normal_dir,"*.bmp")
dir2 = os.path.join(normal_dir,"*.jpeg")
dirc = os.path.join(normal_dir,"*.jpg")
normal_files3 = glob.glob(dirc)
normal_1 = glob.glob(dir1)
normal_2 = glob.glob(dir2)
normal_files3.extend(normal_1)
normal_files3.extend(normal_2)

normal_dir = "client3/im_Koilocytotic"  #give your covid 19 cases data path here
dir1 = os.path.join(normal_dir,"*.bmp")
dirc = os.path.join(normal_dir,"*.jpg")
dir2 = os.path.join(normal_dir,"*.jpeg")
covid_files3 = glob.glob(dirc)
covid_files2 = glob.glob(dir2)
covid_files1 = glob.glob(dir1)
covid_files3.extend(covid_files2)
covid_files3.extend(covid_files1)

normal_dir = "client3/im_Metaplastic" #give your pneumonia cases data path here
dir1 = os.path.join(normal_dir,"*.bmp")
dir2 = os.path.join(normal_dir,"*.jpeg")
dirc = os.path.join(normal_dir,"*.jpg")
pneumonia_files3 = glob.glob(dirc)
pneumonia_1 = glob.glob(dir1)
pneumonia_2 = glob.glob(dir2)
pneumonia_files3.extend(pneumonia_1)
pneumonia_files3.extend(pneumonia_2)

normal_dir = "client3/im_Parabasal" #give your pneumonia cases data path here
dir1 = os.path.join(normal_dir,"*.bmp")
dir2 = os.path.join(normal_dir,"*.jpeg")
dirc = os.path.join(normal_dir,"*.jpg")
tb_files3 = glob.glob(dirc)
tb_1 = glob.glob(dir1)
tb_2 = glob.glob(dir2)
tb_files3.extend(tb_1)
tb_files3.extend(tb_2)

normal_dir = "client3/im_Superficial-Intermediate" #give your pneumonia cases data path here
dir1 = os.path.join(normal_dir,"*.bmp")
dir2 = os.path.join(normal_dir,"*.jpeg")
dirc = os.path.join(normal_dir,"*.jpg")
cat_files3 = glob.glob(dirc)
cat_1 = glob.glob(dir1)
cat_2 = glob.glob(dir2)
cat_files3.extend(cat_1)
cat_files3.extend(cat_2)

normal_files3.sort()
covid_files3.sort()
pneumonia_files3.sort()
tb_files3.sort()
cat_files3.sort()
normal_files3 = shuffle(normal_files3, random_state=10)
covid_files3 = shuffle(covid_files3, random_state=10)
pneumonia_files3 = shuffle(pneumonia_files3, random_state=10)
tb_files3 = shuffle(tb_files3, random_state=10)
cat_files3 = shuffle(cat_files3, random_state=10)

print("pneumonia_files:", len(pneumonia_files3))
print("covid_files:", len(covid_files3))
print("normal_files:", len(normal_files3))
print("tb_files:", len(tb_files3))
print("cat_files:", len(cat_files3))

total_files3 = len(normal_files3) + len(covid_files3) + len(pneumonia_files3) + len(tb_files3) + len(cat_files3)

temp_files3 = []
temp_labels3 = []
for i in range(len(pneumonia_files3)):
    temp_files3.append(pneumonia_files3[i])
    temp_labels3.append(0)

for i in range(len(covid_files3)):
    temp_files3.append(covid_files3[i])
    temp_labels3.append(1)

for i in range(len(normal_files3)):
    temp_files3.append(normal_files3[i])
    temp_labels3.append(2)

for i in range(len(tb_files3)):
    temp_files3.append(tb_files3[i])
    temp_labels3.append(3)
    
for i in range(len(cat_files3)):
    temp_files3.append(cat_files3[i])
    temp_labels3.append(4)

temp_files3 = shuffle(temp_files3, random_state=10)
temp_labels3 = shuffle(temp_labels3, random_state=10)


# In[24]:


temp_files1=np.array(temp_files1)
temp_labels1=np.array(temp_labels1)
temp_files2=np.array(temp_files2)
temp_labels2=np.array(temp_labels2)
temp_files3=np.array(temp_files3)
temp_labels3=np.array(temp_labels3)


# In[25]:


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
for train_index, val_index in skf.split(temp_files1, temp_labels1):
    
    # Split the data into training and validation sets
        X_train1, X_val1 = temp_files1[train_index], temp_files1[val_index]
        y_train1, y_val1 = temp_labels1[train_index], temp_labels1[val_index]


# In[26]:


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
for train_index, val_index in skf.split(temp_files2, temp_labels2):
    
    # Split the data into training and validation sets
        X_train2, X_val2 = temp_files2[train_index], temp_files2[val_index]
        y_train2, y_val2 = temp_labels2[train_index], temp_labels2[val_index]


# In[27]:


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
for train_index, val_index in skf.split(temp_files3, temp_labels3):
    
    # Split the data into training and validation sets
        X_train3, X_val3 = temp_files3[train_index], temp_files3[val_index]
        y_train3, y_val3 = temp_labels3[train_index], temp_labels3[val_index]


# In[28]:


x_train1= np.concatenate((X_train1,X_val1 ), axis=0)
y_train1= np.concatenate((y_train1,y_val1 ), axis=0)


# In[30]:


# Concatenate along the first axis (rows)
x_train2= np.concatenate((X_train2,X_val2 ), axis=0)
y_train2= np.concatenate((y_train2,y_val2 ), axis=0)


# In[31]:


# Concatenate along the first axis (rows)
x_train3= np.concatenate((X_train3,X_val3 ), axis=0)
y_train3= np.concatenate((y_train3,y_val3 ), axis=0)


# In[32]:


#train
train_normal_files1=[]
for i in range(len(X_train1)):
            if y_train1[i]==2:
                train_normal_files1.append(X_train1[i])

train_covid_files1=[]
for i in range(len(X_train1)):
            if y_train1[i]==1:
                train_covid_files1.append(X_train1[i])

train_pneumonia_files1=[]
for i in range(len(X_train1)):
            if y_train1[i]==0:
                train_pneumonia_files1.append(X_train1[i])

train_tb_files1=[]
for i in range(len(X_train1)):
            if y_train1[i]==3:
                train_tb_files1.append(X_train1[i])
                
        
train_cat_files1=[]
for i in range(len(X_train1)):
            if y_train1[i]==4:
                train_cat_files1.append(X_train1[i])


# In[33]:


#train
train_normal_files2=[]
for i in range(len(X_train2)):
            if y_train2[i]==2:
                train_normal_files2.append(X_train2[i])

train_covid_files2=[]
for i in range(len(X_train2)):
            if y_train2[i]==1:
                train_covid_files2.append(X_train2[i])

train_pneumonia_files2=[]
for i in range(len(X_train2)):
            if y_train2[i]==0:
                train_pneumonia_files2.append(X_train2[i])

train_tb_files2=[]
for i in range(len(X_train2)):
            if y_train2[i]==3:
                train_tb_files2.append(X_train2[i])
                
        
train_cat_files2=[]
for i in range(len(X_train2)):
            if y_train2[i]==4:
                train_cat_files2.append(X_train2[i])


# In[34]:


#train
train_normal_files3=[]
for i in range(len(X_train3)):
            if y_train3[i]==2:
                train_normal_files3.append(X_train3[i])

train_covid_files3=[]
for i in range(len(X_train3)):
            if y_train3[i]==1:
                train_covid_files3.append(X_train3[i])

train_pneumonia_files3=[]
for i in range(len(X_train3)):
            if y_train3[i]==0:
                train_pneumonia_files3.append(X_train3[i])

train_tb_files3=[]
for i in range(len(X_train3)):
            if y_train3[i]==3:
                train_tb_files3.append(X_train3[i])
                
        
train_cat_files3=[]
for i in range(len(X_train3)):
            if y_train3[i]==4:
                train_cat_files3.append(X_train3[i])


# In[35]:


def on_hot_encode_labels(lables):
    aug_list=[]
    for i in range(len(lables)):
        if lables[i]==0:
            aug_list.append([0,1,0,0,0])
        elif lables[i]==1:
            aug_list.append([1,0,0,0,0])
        elif lables[i]==2:
            aug_list.append([0,0,1,0,0])
        elif lables[i]==3:
            aug_list.append([0,0,0,1,0])
        elif lables[i]==4:
            aug_list.append([0,0,0,0,1])
    return aug_list


# In[36]:


import glob
import pickle
import tensorflow as tf
import argparse
import re
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import datetime
import keras
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,Layer,ReLU, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
import numpy as np
#from keras_cv.layers import RandomCutout
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from skimage import data, exposure
from skimage.transform import radon, rescale
from skimage.filters import roberts, sobel, scharr, prewitt
from classification_models.keras import Classifiers
from skimage import feature
import os,glob
import numpy as np
import cv2
import glob
import pickle
import tensorflow as tf
import pickle
import argparse
import re
import datetime
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,ConvLSTM2D,LayerNormalization,GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as rp
from classification_models.keras import Classifiers
from tensorflow.keras.applications.xception import preprocess_input as xp
import sklearn.metrics as metrics
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.utils import get_file
import os,glob
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import keras
from classification_models.keras import Classifiers
import numpy as np
import cv2
import glob
import pickle
#import clahe
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix , classification_report
from matplotlib import pyplot as plt
import tensorflow as tf
import argparse
import re
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
from sympy.solvers import solve
from sympy import Symbol
import seaborn as sns
import numpy as np
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,Layer,ReLU, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import f1_score
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from skimage import data, exposure
from skimage.transform import radon, rescale
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage import feature
import os,glob
import numpy as np
import cv2
import glob
import pickle
import tensorflow as tf
import pickle
import argparse
import re
import datetime
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,ConvLSTM2D,LayerNormalization,GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from skimage import data, exposure
from tensorflow.keras.layers import Layer
from PIL import Image
from numpy import asarray
from sklearn.utils import shuffle
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf   
import keras
from classification_models.keras import Classifiers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from tensorflow.keras.metrics import Recall, Precision
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from tensorflow.keras.metrics import Recall, Precision
from skimage import data, exposure
from tensorflow.keras.layers import Layer
from PIL import Image
from numpy import asarray
from sklearn.utils import shuffle
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import pandas as pd

import cv2
IM_SIZE=128


# In[37]:


from PIL import Image
import numpy as np
import cv2
from numpy import asarray
from tqdm import tqdm
def no_data_augmentation(normal_files,covid_files,pneumonia_files,tb_files,cat_files):
    aug_normal=[]
    aug_covid=[]
    aug_pneumonia=[]
    aug_tb=[]
    aug_cat=[]
    for ele in normal_files:
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(128,128),interpolation = cv2.INTER_CUBIC)
        pic=pic/255.0
        aug_normal.append(pic)
    for ele in covid_files:
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(128,128),interpolation = cv2.INTER_CUBIC)
        pic=pic/255.0
        aug_covid.append(pic)
    for ele in pneumonia_files:
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
      
        pic = cv2.resize(x,(128,128),interpolation = cv2.INTER_CUBIC)
        pic=pic/255.0
        aug_pneumonia.append(pic)
    
    for ele in tb_files:
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(128,128),interpolation = cv2.INTER_CUBIC)
        pic=pic/255.0
        aug_tb.append(pic)    
    for ele in cat_files:
        #ele=ele/255.0
        x = Image.open(ele)
        x = asarray(x)
        
        pic = cv2.resize(x,(128,128),interpolation = cv2.INTER_CUBIC)
        pic=pic/255.0
        aug_cat.append(pic)   
    for i in range(len(aug_normal)):
        aug_normal[i]=aug_normal[i].reshape((128,128,3))
    
    for i in range(len(aug_covid)):
        aug_covid[i]=aug_covid[i].reshape((128,128,3))
    for i in range(len(aug_pneumonia)):
        aug_pneumonia[i]=aug_pneumonia[i].reshape((128,128,3))
    for i in range(len(aug_tb)):
        aug_tb[i]=aug_tb[i].reshape((128,128,3))    
    for i in range(len(aug_cat)):
        aug_cat[i]=aug_cat[i].reshape((128,128,3)) 
    
    print("Normal files without augmentation:",len(aug_normal))
    print("Covid files without augmentation:", len(aug_covid))
    print("Pneumonia files without augmentation:",len(aug_pneumonia))
    print("tb files without augmentation:",len(aug_tb))
    print("cat files without augmentation:",len(aug_cat))
    return aug_normal,aug_covid,aug_pneumonia, aug_tb, aug_cat


# In[20]:


# print(len(train_aug_normal1))
# print(train_aug_covid1.shape)
# print(train_aug_pneumonia1.shape)
# print(train_aug_tb1.shape)
# print(train_aug_cat1.shape)


# In[21]:


train_aug_normal1,train_aug_covid1,train_aug_pneumonia1,train_aug_tb1,train_aug_cat1=no_data_augmentation(train_normal_files1,train_covid_files1,train_pneumonia_files1,train_tb_files1,train_cat_files1)
train_full_data1,train_full_label1=making_full_data_train(train_aug_normal1,train_aug_covid1,train_aug_pneumonia1,train_aug_tb1,train_aug_cat1)  #getting my full data


# In[32]:


train_aug_normal2,train_aug_covid2,train_aug_pneumonia2,train_aug_tb2,train_aug_cat2=no_data_augmentation(train_normal_files2,train_covid_files2,train_pneumonia_files2,train_tb_files2,train_cat_files2)
train_full_data2,train_full_label2=making_full_data_train(train_aug_normal2,train_aug_covid2,train_aug_pneumonia2,train_aug_tb2,train_aug_cat2)  #getting my full data


# In[33]:


train_aug_normal3,train_aug_covid3,train_aug_pneumonia3,train_aug_tb3,train_aug_cat3=no_data_augmentation(train_normal_files3,train_covid_files3,train_pneumonia_files3,train_tb_files3,train_cat_files3)
train_full_data3,train_full_label3=making_full_data_train(train_aug_normal3,train_aug_covid3,train_aug_pneumonia3,train_aug_tb3,train_aug_cat3)  #getting my full data


# In[22]:


# train1c=train_full_data1
# label1c=train_full_label1
# # train2u=train_full_data2
# # label2u=train_full_label2
# # train3u=train_full_data3
# # label3u=train_full_label3
# print(train1c.shape)
# print(label1c.shape)
# # print(train2u.shape)
# # print(label2u.shape)
# # print(train3u.shape)
# # print(label3u.shape)


# In[24]:


# import numpy as np
# np.save('train1c.npy', train1c)
# np.save('label1c.npy', label1c)
# # np.save('train2u.npy', train2u)
# # np.save('label2u.npy', label2u)
# # np.save('train3u.npy', train3u)
# # np.save('label3u.npy', label3u)
# print("save sucessfull")


# In[36]:


# a=np.load('train1u.npy')
# b=np.load('label1u.npy')
# c=np.load('train2u.npy')
# d=np.load('label2u.npy')
# e=np.load('train3u.npy')
# f=np.load('label3u.npy')


# In[ ]:


train1=train_full_data1
label1=train_full_label1
train2=train_full_data2
label2=train_full_label2
train3=train_full_data3
label3=train_full_label3

