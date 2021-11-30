"""
    This utitlity program is written by FC Tien for teaching PyTorch
    @Objective: General utitlity program for ResNet_like classification function
                including: predict, load_full_model, predict....
    @File   : pytorch_utitlity.py
    @Author : FC. Tien (Dept. of IE&M, Taipei Tech)
    @E-mail : fctien@ntut.edu.tw
    @Date   : 20200515
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
# from utils import image_processing, TDataSet
from torchvision import transforms
import os
import torchvision.models as models  ## use build-in model
import time
import cv2
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, \
        recall_score, f1_score, classification_report, ConfusionMatrixDisplay


def findAllImagFiles(path = "./train/"):  ## 搜尋目錄下所有相關影像之檔名
    from glob import glob
    pattern = os.path.join(path, '*.bmp') 
    bmp_files = sorted(glob(pattern))
    #print(type(bmp_files))
    pattern = os.path.join(path, '*.jpg')
    jpg_files = sorted(glob(pattern))
    pattern = os.path.join(path, '*.jpeg')
    jpeg_files = sorted(glob(pattern))
    pattern = os.path.join(path, '*.png')
    png_files = sorted(glob(pattern))
    file_list = bmp_files + jpg_files + jpeg_files+ png_files
    return file_list  ## 回傳檔名的 list

def convert_onehot(x, num_class = 4):
    """
    input: a list of number
    output: one_hot: 2d array
    """
    x = np.array(x)
    one_hot = np.eye(num_class)[x]
    one_hot = one_hot.astype(int)
    return one_hot

def read_torch_image(fn, isShow = True, isStdNormalized = False):
    cvImg = cv2.imread(fn, -1)       
    if cvImg is None:
        print("Image does not exist.")
        return
    cvImg_channel_first = np.moveaxis(cvImg, -1, 0)
    torch_image = torch.from_numpy(cvImg_channel_first)
    if isStdNormalized:
        torch_image = transforms.normalize(torch_image/ 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if isShow:
        print(torch_image.shape)
        cv2.imshow(fn, cvImg)
        cv2.waitKey(-1)
    return torch_image

def torch_image_to_cvImg(torch_image, isShow = True):
    cvImg = torch_image.cpu().detach().numpy()
    cvImg = np.moveaxis(cvImg, 0, -1)
    if isShow:
        print(cvImg.shape)
        cv2.imshow("Images", cvImg)
        cv2.waitKey(-1)
    return cvImg

def cv_to_torch_image(cvImage):
    cvImg_channel_first = np.moveaxis(cvImage, -1, 0)
    torch_image = torch.from_numpy(cvImg_channel_first)
    return torch_image

def predict_cvImg(model, device, cvImage, resize_height=224, resize_width =224, isShow=True, isStdNormalized =True):
    """
    Input: 
        model: ResNet (Use load_full_model to load and pass in this function)
        device: "cuda:0"
        cvImage: 輸入 cv2 影像，BGR numpy matrix
        Resize width & Height: ResNet default 224x224
    """
    cvImage = cv2.resize(cvImage, (resize_width, resize_height))
    rgb_img = cv2.cvtColor(cvImage, cv2.COLOR_GRAY2RGB)

    model.eval()
    # do something like transform does  ## using Transform is faster? about the same
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  ## 要同步在 test 修正
    ])
    rgb_img = cv2.cvtColor(cvImage, cv2.COLOR_GRAY2RGB)
    if isStdNormalized:
        rgb_tensor = transform(rgb_img)
    else:
        rgb_img = rgb_img/255.  ## to floating
        rgb_img = np.moveaxis(rgb_img, -1, 0)  ## switch to (c, w, h) format
        rgb_tensor = torch.from_numpy(rgb_img)
        rgb_tensor = rgb_tensor.float()
    rgb_tensor= torch.unsqueeze(rgb_tensor, 0) 
    #print(rgb_tensor.shape)
    inputs = rgb_tensor.to(device)
    outputs = model(inputs)
    ## calculate the prob.
    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(outputs)
    _, predicted = outputs.max(1)
    label = predicted.item()
    prob = probabilities[0][label].item()
    #print("Label/Probl: ", label, prob)
    if isShow:
        cv2.imshow("image", cvImage)
        cv2.waitKey(0)
    return  label, prob

def load_full_model(filename="./model/early_stop_mask_cc_model_0603.pth"):
    model = torch.load(filename)
    return model

# def save_full_model(model, filename = "./model/ResNet_best"):
#     torch.save(model, filename)  ## save entire model
#     return

# def write_training_process(loss_list, acc_list, filename="./model/train_process.txt"):
#     with open(filename, "w") as f:
#         for i in range(len(loss_list)):
#             f.write(str(loss_list[i][0]) + ", " + str(loss_list[i][1]) + ", " + str(acc_list[i][0]) + ", " + str(acc_list[i][1]) + "\n")
#     return

# def write_training_loss_process(loss_list, filename="./model/train_process.txt"):
#     with open(filename, "w") as f:
#         for i in range(len(loss_list)):
#             f.write(str(loss_list[i][0]) + ", " + str(loss_list[i][1]) + "\n")
#     return

# def plot_loss_acc(loss_list, acc_list):
#     import matplotlib.pyplot as plt
#     fig = plt.figure("Loss & Accuracy", figsize= (12, 6))
#     plt.ion()
#     plt.clf()
#     #plt.suptitle("Loss & Accuracy plot")
#     plt.title('The training process - Loss & Accuracy')
#     loss = np.asarray(loss_list)[:, 0]
#     val_loss = np.asarray(loss_list)[:, 1]
#     train_acc = np.asarray(acc_list)[:, 0]
#     val_acc = np.asarray(acc_list)[:, 1]
#     x = [i for i in range(len(train_acc))]
#     # loss
#     plt.subplot(1, 2, 1)
    
#     plt.tight_layout()
#     plt.plot(x, loss, color="red", marker = '.', label = "Train")
#     plt.plot(x, val_loss, color = 'blue', marker = '.', label = 'Test')
#     plt.xlabel("Loss Epoches")
#     plt.ylabel("Loss")
#     plt.legend()
#     #  acc
#     plt.subplot(1, 2, 2)
#     #plt.suptitle("Accuracy plot")
#     plt.tight_layout()
#     plt.plot(x, train_acc, color="red", marker = '.', label = "Train")
#     plt.plot(x, val_acc, color = 'blue', marker = '.', label = 'Test')
#     plt.xlabel("Acc Epoches")
#     plt.ylabel("Accuracy")
#     plt.ylim((0.0, 100.0))
#     plt.legend()
#     plt.pause(1)
#     plt.show()   
#     if not os.path.isdir('./model'):
#         os.mkdir('./model')
#     plt.savefig("./model/Accuracy_figure.png")
    
# callback loss-plot
# def plot_loss(loss_list): ## loss_list  [(train_loss, test_loss)]
#     import matplotlib.pyplot as plt
#     plt.ion()
#     plt.clf()
#     plt.tight_layout()
#     plt.title('The training process - Loss')
#     loss = np.asarray(loss_list)[:, 0]
#     val_loss = np.asarray(loss_list)[:, 1]
#     x = [i for i in range(len(loss))]
#     plt.plot(x, loss, color="red", marker = '.')
#     plt.plot(x, val_loss, color = 'blue', marker = '.')
#     plt.pause(1)
#     plt.show()  
#     if not os.path.isdir('./result'):
#         os.mkdir('./result') 
#     plt.savefig("./result/Loss_figure.png")
#     return

# callback loss-plot
# def plot_acc(acc_list):
#     import matplotlib.pyplot as plt
#     plt.ion()
#     plt.clf()
#     plt.tight_layout()
#     plt.title('The training process - Accuracy')
#     train_acc = np.asarray(acc_list)[:, 0]
#     val_acc = np.asarray(acc_list)[:, 1]
#     x = [i for i in range(len(train_acc))]
#     plt.plot(x, train_acc, color="red", marker = '.')
#     plt.plot(x, val_acc, color = 'blue', marker = '.')
#     plt.pause(0.3)
#     plt.show()   
#     if not os.path.isdir('./result'):
#         os.mkdir('./result')
#     plt.savefig("./result/Accuracy_figure.png")
    # return

# def findAllImagFiles(path = "./train/"):  ## 搜尋目錄下所有相關影像之檔名
#     from glob import glob
#     pattern = os.path.join(path, '*.bmp') 
#     bmp_files = sorted(glob(pattern))
#     #print(type(bmp_files))
#     pattern = os.path.join(path, '*.jpg')
#     jpg_files = sorted(glob(pattern))
#     pattern = os.path.join(path, '*.jpeg')
#     jpeg_files = sorted(glob(pattern))
#     pattern = os.path.join(path, '*.png')
#     png_files = sorted(glob(pattern))
#     file_list = bmp_files + jpg_files + jpeg_files+ png_files
#     return file_list  ## 回傳檔名的 list

def save_class_list(class_list):
    with open('./class_list.txt', 'w', encoding='utf8') as f:
        for l in class_list:
            f.write(l + "\n")
    # with open('./class_list.txt', 'w', encoding='utf8') as f:
    #     for l in class_list:
    #         f.write(l + "\n")
    return

## written by Tien 20200515
def find_no_image_in_dir(path):
    import os
    import gc
    print("[MSG]: Reading the data by classes and data balancing ...")
    No_Image_in_dirs = list()
    dirs = os.listdir(path)
    print("Classes: ", dirs)
    count = 0
    class_list = list()
    ## calcuate the image no in each directory        
    for dir in dirs:
        fullpath = os.path.join(path, dir)
        class_list.append(dir)  ## class label
        if os.path.isdir(fullpath):
            ##files_path = os.path.join(fullpath, '*.jpg')  
            ##file = sorted(glob(files_path))
            file = findAllImagFiles(fullpath)
            No_Image_in_dirs.append(len(file))
            print(dir, " = ", No_Image_in_dirs[count])
            count +=1
    #print("Max No: ", max(No_Image_in_dirs))
    del file
    gc.collect()
    save_class_list(class_list)
    return No_Image_in_dirs, class_list

# def display_confusiotn_matrix(gt_list, predict_list, display_labels = ['0','1', '2'], isShow = True,  isSave = True):
#     import matplotlib.pyplot as plt
#     cm = confusion_matrix(gt_list, predict_list, normalize='true') ## {'true', 'pred', 'all'}
#     cmd = ConfusionMatrixDisplay(cm, display_labels=display_labels)
#     if isShow:
#         cmd.plot()
#     cmd.ax_.set(title='Confusion Matrix', xlabel='Predicted', ylabel='True')
#     fig = cmd.figure_
#     fig.canvas.set_window_title('Confusion Matrix')
#     if isSave:
#         fig.savefig("./model/confusion_matrx.png")
#     plt.show()
#     return cm

# def evaluate_score(image_dir, model, save_path = "./Misclassified", isStdNormalized= True):  ## conduct the training/val images evaluation 9by batch
#     """
#     1. Calculate the confusion matrix by sklearn.metric
#     2. Find all misclassified image and save into a save_path
#     3. Calculate the precision, recall, F1 score
#     """
#     ## find all misclassified images in original training images
#     import os
#     import gc
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else: 
#         device = torch.device("cpu")
#     if not os.path.isdir(save_path):
#         os.mkdir(save_path)
#     print("[MSG]: Reading the data by classes with no data balancing ...")
#     no_image_in_dirs, class_list = find_no_image_in_dir(path=image_dir)
#     # ## read all images in training directory
#     dirIndex = 0
#     tag_list = list()
#     predict_list = list()
#     acc = 0
#     dirs = os.listdir(image_dir)
#     count = 0
#     for dir in dirs:
#         fullpath = os.path.join(image_dir, dir)
#         if os.path.isdir(fullpath):
#             #files_path = os.path.join(fullpath, '*.jpg')
#             #files = sorted(glob(files_path))
#             files = findAllImagFiles(fullpath)
#             ## copy def read_data, but force no_of_copy = 1
#             no_of_copy = 1 ##  int(max(No_Image_in_dirs) / No_Image_in_dirs[dirIndex]+0.5)
#             #print("No of copies: ", no_of_copy)        
#             no_of_image = 0           
#             for f in files:
#                 #try:
#                     for i in range(no_of_copy):
#                         #img = load_img(f, target_size=(self.size_image_x, self.size_image_y, self.no_of_channel)) ## keras read data and reshape
#                         img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), -1) #img = cv2.imread(f, -1)
#                         #prob, tag = self.predit_cvImage(img)# replace this by pytorch predict_cvImg
#                         tag, prob = predict_cvImg(model, device, img, resize_height =224, resize_width = 224, isShow=False, isStdNormalized= isStdNormalized)
#                         #print(prob)
#                         tag_list.append(tag)
#                         #print(dirIndex)
#                         predict_list.append(dirIndex)
#                         count+=1
#                         if tag == dirIndex:
#                             acc = acc + 1
#                             #print("correct")
#                         else:
#                             #print("Wrong")
#                             #save_path = "./Misclassified/"
#                             fn = os.path.basename(f)
#                             filename  = save_path + "/" + class_list[dirIndex]+ "_to_" + class_list[tag] + "_" + fn
#                             cv2.imencode('.png', img)[1].tofile(filename) #imwrite(filename, img)
#                             #x = x.reshape((self.size_image_x, self.size_image_y, self.no_of_channel) )
#                             #wrong_x_list.append(x)    
#                 # except:
#                 #     print("[MSG] Data reading error...", f)
#                 #     continue
#                 #    no_of_image +=1
#             dirIndex +=1  ## store 0, 1, 2, 3, 4
#             #print(dir, ":", no_of_image)
#     print("Total number of image: ", count)
#     acc = acc / count
#     print("Over all ACC = ", acc)

#     accuracy = accuracy_score(tag_list, predict_list)
#     print("Accuracy = ", accuracy)
#     recall_all_micro = recall_score(tag_list, predict_list, average='micro')
#     print("Recall(mirco) = ", recall_all_micro)
#     recall_all_macro = recall_score(tag_list, predict_list, average='macro')
#     print("Recall (marco) = ", recall_all_macro)
#     precision_all_macro = precision_score(tag_list, predict_list, average='macro')
#     print("Precision (marco) = ", precision_all_macro)
#     precision_all_micro = precision_score(tag_list, predict_list, average='micro')
#     print("Precision (mirco) = ", precision_all_micro)
#     f1_all_macro = f1_score(tag_list, predict_list, average='macro')
#     print("F1 Score (macro) = ", f1_all_macro)
#     f1_all_micro = f1_score(tag_list, predict_list, average='micro')
#     print("F1 Score (micro) = ", f1_all_micro)
#     cm_all = confusion_matrix(tag_list, predict_list )
#     print(cm_all)
#     fn = "result.txt"
#     fn = save_path + "/" + fn
#     f = open(fn, "w", encoding='utf8')
#     f.write("Overall Acc = " + str(round(acc, 4)) + "\n" )
#     f.write("Recall(micro) = " + str(recall_all_micro) +"\n")
#     f.write("Recall(marco) = " + str(recall_all_macro)+"\n")
#     f.write("Precision(micro) = " + str(precision_all_micro) +"\n")
#     f.write("Precision(marco) = " + str(precision_all_macro)+"\n")
#     f.write("F1 Score (micro) = " + str(f1_all_micro) +"\n")
#     f.write("F1 Score (marco) = " + str(f1_all_macro)+"\n")
#     f.write("[Result]\n")
#     #f.write("Model = " + path + "\n")
#     f.write("Confusion Matrix (Overall): \n")
#     f.write(str(cm_all))
#     f.close()
#     display_labeles = [str(i) for i in range(len(class_list))]
#     display_confusiotn_matrix(tag_list, predict_list, display_labels = display_labeles, isSave=True)
#     return acc, cm_all, precision_all_micro, recall_all_micro, f1_all_micro