import glob
import shutil, os

path=r"E:/DOTA DATASET/DOTA-v1.5_train/*.txt"
path1=r"E:/DOTA DATASET/*/*/*.png"
filelist=glob.glob(path)
filelist1=glob.glob(path1)
ship_images=[]

def read_text_file(file_path,file_path1):
    f=open(file_path, 'r')
    pp=f.read()
    f1=open(file_path, 'r')
    lines=f1.readlines()
    if ('ship' in pp):
       ship_images.append(file_path1)
       file="E:/DOTA DATASET/labels/{}.txt".format(file_path1.split("\\")[-1].split('.')[0])
       ff=open(file,"w+") # file name and mode
       print (lines)
       ff.writelines(lines[0])
       ff.writelines(lines[1])

       for i in lines:
           if('ship' in i):
               ff.writelines(i)
       ff.close()
       f1.close()
       f.close()            
                

for i in range(len(filelist)):
    read_text_file(filelist[i],filelist1[i])

#for f in ship_images:
#    shutil.copy(f, 'E:\DOTA DATASET\ships')
