import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as td
import torchvision as tv
import xml.etree.ElementTree as ET
from PIL import Image
from matplotlib import pyplot as plt
# from scipy.misc import imread,imresize
from matplotlib.pyplot import imread
from skimage.transform import resize
# im = imread(image.png)


class VOCDataset(td.Dataset):
    def __init__(self, root_dir, mode="train", image_size=(224, 224)):
        super(VOCDataset, self).__init__()
        self.image_size = image_size
        self.mode = mode
        #self.data = pd.read_csv(os.path.join(root_dir, "%s.xml" % mode))
        self.annotations_dir = os.path.join(root_dir, "Annotations")
        self.images_dir = os.path.join(root_dir, "JPEGImages")
        
        # os.listdir returns list in arbitrary order
        self.image_names = os.listdir(self.annotations_dir)
        train_l = int(len(self.image_names)*0.7)
        
        
        self.image_names = [image.rstrip('.xml') for image in self.image_names]
        self.train_list = self.image_names[0:train_l]
        self.valid_list  = self.image_names[train_l:]
        self.test_list=self.image_names
        self.voc_dict = {
                        'person':1, 'bird':2, 'cat':3, 'cow':4, 'dog':5, 
                        'horse':6, 'sheep':7, 'aeroplane':8, 'bicycle':9,
                        'boat':10, 'bus':11, 'car':12, 'motorbike':13, 'train':14, 
                        'bottle':15, 'chair':16, 'diningtable':17, 
                        'pottedplant':18, 'sofa':19, 'tvmonitor':20
                        }

        
        
    def __len__(self):
        if self.mode=='train':
            image_names=self.train_list
        elif self.mode=='val':
            image_names=self.valid_list
        elif self.mode=='test':
            image_names=self.test_list
        
        return len(image_names)

    def __repr__(self):
        return "VOC2012Dataset(mode={}, image_size={})". \
            format(self.mode, self.image_size)
    
    def __getitem__(self, idx):
        # Get file paths for image and annotation (label)
        if self.mode=='train':
            image_names = self.train_list
        elif self.mode=='val': 
            image_names = self.valid_list
        elif self.mode=='test':
            image_names= self.test_list

        img_path = os.path.join(self.images_dir, \
                                "%s.jpg" % image_names[idx])
        lbl_path = os.path.join(self.annotations_dir, \
                                "%s.xml" % image_names[idx])   
        
        # Get objects and bounding boxes from annotations
        lbl_tree = ET.parse(lbl_path)
        objs = []
        gt=[]
        bboxes=[] 
        label=[]
        labels=[]
        
        for obj in lbl_tree.iter(tag='object'):
            name = obj.find('name').text
            for box in obj.iter(tag='bndbox'):
                if name=='person':
                    xmax = box.find('xmax').text
                    xmin = box.find('xmin').text
                    ymax = box.find('ymax').text
                    ymin = box.find('ymin').text
                    break
                xmax = box.find('xmax').text
                xmin = box.find('xmin').text
                ymax = box.find('ymax').text
                ymin = box.find('ymin').text
            attr = (self.voc_dict[name], float((float(xmin)+float(xmax))/2),float((float(ymin)+float(ymax))/2), float(float(xmax)-float(xmin)), float(float(ymax)-float(ymin)), 1)
            attr1=(name,float(xmin),float(ymax),float(ymin),float(xmax))
            objs.append(attr)
            gt.append(attr1)
            label.append(self.voc_dict[name])
        #bboxes.append(torch.Tensor(bbox))
        #labels.append(torch.IntTensor(label))
        

        objs = torch.Tensor(objs)
        # Open and normalize the image
        img = imread(img_path)
        h,w,_=img.shape
        img=resize(img,(224,224))
        transform = tv.transforms.Compose([
            #tv.transforms.Resize((448,448)),
            tv.transforms.ToTensor(),
            
        ])
        d = gt
        x = transform(img)
        if self.mode!='test':
            target = torch.zeros((7,7,30))
            cls=torch.zeros((len(objs),20))
            x_list=torch.Tensor((len(objs)))
            y_list=torch.Tensor((len(objs)))
            w_list=torch.Tensor((len(objs)))
            h_list=torch.Tensor((len(objs)))
            x_index=torch.Tensor((len(objs)))
            y_index=torch.Tensor((len(objs)))
            x_new=torch.Tensor((len(objs)))
            y_new=torch.Tensor((len(objs)))
            del_x=torch.Tensor((len(objs)))
            del_y=torch.Tensor((len(objs)))
            for i in range(len(objs)):
                x_list[i]=objs[i][1]/w
                y_list[i]=objs[i][2]/h
                w_list[i]=objs[i][3]/w
                #w_list[i]=torch.sqrt(w_list[i])
                h_list[i]=objs[i][4]/h
                #h_list[i]=torch.sqrt(h_list[i])
                x_index[i]=(x_list[i]/(1./7)).ceil()-1
                y_index[i]=(y_list[i]/(1./7)).ceil()-1
                x_new[i]=x_index[i]*(1./7)
                y_new[i]=y_index[i]*(1./7)
                del_x[i]=(x_list[i]-x_new[i])/(1./7)
                del_y[i]=(y_list[i]-y_new[i])/(1./7)
            c=torch.ones(len(objs))
            bb_block=torch.cat((del_x.view(-1,1),del_y.view(-1,1),w_list.view(-1,1),h_list.view(-1,1),c.view(-1,1)),dim=1)
            bb_block=bb_block.repeat(1,2)

            for i in range(len(objs)):
                cls[i,int(objs[i][0])-1]=1
            final_bb=torch.cat((bb_block,cls),dim=1)

            for i in range(len(objs)):
                target[int(x_index[i]),int(y_index[i])]=final_bb[i].clone()
                
        if self.mode =='test':
            target=d
        self.target=target
        if self.mode=='test':
            return x,target,image_names[idx]
        else:
            return x, target

    def number_of_classes(self):
        #return self.data['class'].max() + 1
        # TODO: make more flexible
        return 20

    def print_target(self):
        for i in range(7):
             print(self.target[i])

def myimshow(image, ax=plt):
    image = image.to('cpu').detach().numpy()
    image = np.moveaxis(image, [0,1,2], [2,0,1])
    image = (image + 1)/2
    image[image<0] = 0
    image[image>1] = 1
    h = ax.imshow(image)        
    ax.axis('off')
    return h

if __name__ == '__main__':
    xmlparse = VOCDataset('../VOCdevkit/VOC2012')
    print(xmlparse[0])
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    #print(axes)
    x, y = xmlparse[0]
    xmlparse.print_target()
    x1, y1 = xmlparse[1]
    myimshow(x, ax=ax1)
    #myimshow(x1, ax=ax2)
    plt.show()
    
