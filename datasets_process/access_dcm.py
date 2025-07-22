import pydicom
import matplotlib.pyplot as plt

im = pydicom.dcmread(r'D:\GitHub\chest-ct-report-generator\datasets\Lung-PET-CT-Dx\manifest-1608669183333\Lung-PET-CT-Dx\Lung_Dx-A0001\04-04-2007-NA-Chest-07990\2.000000-5mm-40805\1-01.dcm')

# 获取 UID
uid = im.SOPInstanceUID

# 获取像素矩阵
img_arr = im.pixel_array
# 打印矩阵大小
print(img_arr.shape)

# 绘制图像
plt.imshow(img_arr,cmap=plt.cm.bone)
plt.title("UID:{}".format(uid))
plt.show()