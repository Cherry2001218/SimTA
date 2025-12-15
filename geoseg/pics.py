import os  
import cv2  
import glob
# 指定文件夹路径  
folder_path = 'F:\\TESTData\\TestImage\\'  
  
# 获取文件夹中所有文件名  
filenames = os.listdir(folder_path)  
  
# 筛选出文件名最后两位为"B0"的图片文件  
#filtered_filenames = [filename for filename in filenames if filename[-2:] == 'B0' and filename.endswith('.png')]  
# 获取文件夹中所有.png文件  
png_files = glob.glob(os.path.join(folder_path, '*.png'))  
  
# 筛选出后两位为"B0"的.png文件  
filtered_files = [file for file in png_files if os.path.basename(file)[-6:] == 'B0.png']  
#print(filtered_files)
# 指定保存图片的文件夹路径  
output_folder = 'F:\\TESTData\\EImage'  # 修改为你想要保存的文件夹路径  
  
# 创建保存图片的文件夹（如果不存在）  
if not os.path.exists(output_folder):  
    os.makedirs(output_folder)  
  
# 遍历筛选出的图片文件，并使用OpenCV保存到指定文件夹  
for filename in filtered_files:  
    # 读取图片文件  
    img = cv2.imread(os.path.join(folder_path, filename))  
    
   # print(img.shape,filename)
    index = filename.find('0') 

    result = filename[index:]
   # print(result)
    output_filename = os.path.join(output_folder, result)  
      
    # 保存图片到指定文件夹  
    cv2.imwrite(output_filename , img)
    print(output_filename)