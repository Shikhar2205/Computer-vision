
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import pandas as pd 
import numpy as np 
import glob
import cv2 as cv
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from google.colab import drive
from glob import glob
 from engine import train_one_epoch, evaluate
import utils


##  Google Collab platform drive import 
drive.mount('/content/drive')

##  Taking all paths 
path='/content/drive/MyDrive/DOTA dataset/Split_train/ships/*.png'
train_split=glob(path)
path='/content/drive/MyDrive/DOTA dataset/validation/ships/*.png'
test=glob(path)
path='/content/drive/MyDrive/DOTA dataset/train/ships/*.png'
train=glob(path)


## Taking cocoutil files
%%shell
pip install cython-
# Install pycocotools, the version by default in Colab
# has a bug fixed in https://github.com/cocodataset/cocoapi/pull/354
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

## Data Manipulation to delete files one by one 
path='/content/drive/MyDrive/DOTA dataset/Split_train/ships/*.png'
path1='/content/drive/MyDrive/DOTA dataset/Split_train/labels/*.txt'
filelist=glob(path)
filelist1=glob(path1)
filelist2=glob(path1)
filelist3=glob(path)
ll=[]
ll1=[]
filelist=[i.split('/')[-1].split('.')[0] for i in filelist]
filelist1=[i.split('/')[-1].split('.')[0] for i in filelist1]
filelist=sorted(filelist)
filelist1=sorted(filelist1)
filelist2=sorted(filelist2)
filelist3=sorted(filelist3)

print (len(filelist),len(filelist1))
not_there=[x for x in filelist1 if x not in filelist]
print (len(not_there))

## Deleting files from split images which does not have any ships  
for i in range(len(filelist2)):
  with open(filelist2[i],'r') as f:
    print (i)
    lines=f.readlines()
    f.close()
  if (len(lines)==0):
    if os.path.exists(filelist2[i]):
      os.remove(filelist2[i])
      os.remove(filelist3[i])
      print('Done')
    else:
      print("File not found in the directory")
 


## converting the dataset to a Image loader function. 


class objectDataset(object):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "ships"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "labels"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "ships", self.imgs[idx])
        labels_path = os.path.join(self.root, "labels", self.masks[idx])            
        img = Image.open(img_path).convert("RGB")
        ## when we had resized image so we have commented that part  
        #x_,y_= img.size
        #targetSize = 2064
        #x_scale = targetSize / x_
        #y_scale = targetSize / y_
        #img = img.resize((targetSize, targetSize));
        f1=open(labels_path, 'r')
        #lines=f1.readlines()[2:]
        lines=f1.readlines()
        boxes = []
        for i in range(len(lines)):
            split=lines[i].split(' ')[:-2]
            split= [int(float(i)) for i in split]
            # xmin=int(np.round(float(min(split[0],split[2],split[4],split[6]))*x_scale))
            # xmax=int(np.round(float(max(split[0],split[2],split[4],split[6]))*x_scale))
            # ymin=int(np.round(float(min(split[1],split[3],split[5],split[7]))*y_scale))
            # ymax=int(np.round(float(max(split[1],split[3],split[5],split[7]))*y_scale))
            xmin=int(np.round(float(min(split[0],split[2],split[4],split[6]))))
            xmax=int(np.round(float(max(split[0],split[2],split[4],split[6]))))
            ymin=int(np.round(float(min(split[1],split[3],split[5],split[7]))))
            ymax=int(np.round(float(max(split[1],split[3],split[5],split[7]))))
            
            if ((xmax-xmin==0) or (ymax-ymin)==0):
                continue 
            else:
                boxes.append([xmin, ymin, xmax, ymax])
        # convert everything into a torch.Tensor
        iscrowd = torch.zeros((len(lines),), dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((len(lines),), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = np.abs((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]))
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(lines),), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"]=iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


    def __len__(self):
        return len(self.imgs)




 ## loading pretrainined models resnet    
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model =torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

## loading pretrained models mobilenet_v2
def get_instance_segmentation_model(num_classes):
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                      aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                      num_classes=2,
                      rpn_anchor_generator=anchor_generator,
                      box_roi_pool=roi_pooler)
    return model

from engine import train_one_epoch, evaluate
import utils
import transforms as T

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5),T.RandomVerticalFlip(0.5),T.CenterCrop(10),T.RandomRotation([-10,20,90]))
    return T.Compose(transforms)

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # our dataset has two classes only - background and ship
    num_classes = 1
    # use our dataset and defined transformations
    dataset = objectDataset('/content/drive/MyDrive/DOTA dataset/Split_train',get_transform(train=True))
    dataset_test = objectDataset('/content/drive/MyDrive/DOTA dataset/Split_train',get_transform(train=False))
    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()

    dataset = torch.utils.data.Subset(dataset, indices[:-70])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-70:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=5, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)

    # move model to the right device
    model.to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        print("That's it!")
    return model

##  Loading the model. 
model1=main()

## Evaluating model and saving the file on google drive.  
dataset_val = objectDataset('/content/drive/MyDrive/DOTA dataset/validation',get_transform(train=False))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
prep=[]
for i in range(len(dataset_val)):
  img, _ = dataset_val[i]
  # put the model in evaluation mode
  model1.eval()
  with torch.no_grad():
    prediction = model1([img.to(device)])
    result=prediction[0]['boxes'].cpu().data.numpy()
    with open(filelist[i], 'w+') as fp:
      for j in range(len(result)):
        final=result[j].tolist()
        final=[str(int(ii)).strip() for ii in final]
        for k in range(len(final)):
          fp.write(final[k]+' ')
        fp.write('\n')
      fp.close()


