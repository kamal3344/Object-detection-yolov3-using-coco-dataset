#!/usr/bin/env python
# coding: utf-8

# - dataset name = coco dataset
# - version      = yolo v3 
# - yolo version 3 search paper = https://arxiv.org/pdf/1804.02767.pdf
# - website      = https://pjreddie.com/darknet/yolo/
# - coco dataset website = https://cocodataset.org/
# - coco dataset research paper = https://arxiv.org/pdf/1405.0312.pdf
# 

# In[1]:


import cv2 
import numpy as np 


# In[2]:


image = cv2.imread('./testing images/2.jpeg')


# In[3]:


print(image)  # pixel intensity in an image 


# In[4]:


# finding the shape of an image 

image.shape  # first parameter for height , width and channels 


# In[5]:


# finding whether its working fine or not 

cv2.imshow('car',image)
cv2.waitKey()
cv2.destroyAllWindows()


# In[6]:


# total 80 on coco dataset (90) extra 10 recently added so total 90 classes 

classes_names = []
k = open('./Files/class_names','r')
for i in k.readlines():
    classes_names.append(i.strip())


# In[7]:


len(classes_names)


# In[8]:


classes = ['car','person','bus']


# In[9]:


original_with , original_height = image.shape[1] , image.shape[0]


# In[10]:


original_with , original_height


# In[11]:


# Loading cfg file and weights file which are already trained on cooc dataset. 

Neural_Network = cv2.dnn.readNetFromDarknet('./Files/yolov3.cfg','./Files/yolov3.weights')


# In[12]:


Neural_Network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
Neural_Network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# In[13]:


blob = cv2.dnn.blobFromImage(image,1/255,(320,320),True , crop = False)


# In[14]:


blob.shape


# In[15]:


## Getting layer Names 

layers = Neural_Network.getLayerNames()
layers


# In[16]:


layer_names = Neural_Network.getUnconnectedOutLayersNames()
layer_names


# In[17]:


layer_index = Neural_Network.getUnconnectedOutLayers()
layer_index


# In[18]:


layer_index = [layers[j-1] for j in Neural_Network.getUnconnectedOutLayers()]

# since index of any os starts with 0 but this layers count start with 1 


# In[19]:


layer_index


# In[20]:


Neural_Network.setInput(blob)  # input for network 


# In[21]:


outputs = Neural_Network.forward(layer_index)  # giving data to last 3 yolo layers 

# each layer predicts its bounding boxes 


# In[22]:


outputs


# In[23]:


outputs[0].shape # 300 bounding boxes in first box and 85 predcitions in each box 


# In[24]:


# 5-> (x,y,h,w,confidence) + 80 labels in coco dataset 


# In[25]:


outputs[1].shape  # this is in second output layer 


# In[26]:


outputs[2].shape  # this is in third output layer 


# In[38]:


Threshold = 0.3
image_size = 320


def finding_locations(outputs):
    
    bounding_box_locations = []
    class_ids = []
    confidence = []   

    for i in outputs:
        for j in i:
            class_prob = j[5:]                     # finding prob values for all classes 
            class_ids1 = np.argmax(class_prob)      # selecting highest one 
            confidence_value = class_prob[class_ids1]     # selecting its confidence value 
            
            if confidence_value > Threshold:
                # finding w and h 
                w , h = int(j[2] * image_size) , int(j[3] * image_size)
               # print(w , h)
                # finding x and y
                x , y = int(j[0] * image_size - w/2) , int(j[1] * image_size - h / 2)
                bounding_box_locations.append([x,y,w,h])
                class_ids.append(class_ids1)
                confidence.append(float(confidence_value))
                
    
    indeces = cv2.dnn.NMSBoxes(bounding_box_locations,confidence,Threshold,0.5)
    return indeces,bounding_box_locations,confidence,class_ids

    
    


# In[39]:


predicted_box , bounding_box , conf , classes = finding_locations(outputs)


# In[29]:


predicted_box


# In[30]:


bounding_box


# In[31]:


conf


# In[32]:


classes


# In[33]:


font = cv2.FONT_HERSHEY_COMPLEX
height_ratio = original_height / 320
width_ration = original_with / 320


# In[34]:


import matplotlib.pyplot as plt 


# In[35]:


for j in predicted_box.flatten():
    
    x, y , w , h = bounding_box[j]
    x = int(x * width_ration)
    y = int(y * height_ratio)
    w = int(w * width_ration)
    h = int(h * height_ratio)
    
    label = str(classes_names[classes[j]])
    conf_ = str(round(conf[j],2))
    cv2.rectangle(image , (x,y) , (x+w , y+h) , (0,0,255) , 2)
    cv2.putText(image , label+' '+conf_ , (x , y-2) , font , .2 , (0,255,0),1)

#cv2.imshow('Yolo image',image)
#cv2.waitKey()
#cv2.destroyAllWindows()
plt.imshow(image[:,:,::-1])


# In[ ]:


## comlete code 

