# 查看 A0001 号数据的图像文件和xml标注文件
import os
import pydicom

# 查看指定目录下所有的dcm文件名
dcm_file=[]
search_path = r'D:\GitHub\chest-ct-report-generator\datasets\Lung-PET-CT-Dx\manifest-1608669183333\Lung-PET-CT-Dx\Lung_Dx-A0001'
print(f"Searching for DICOM files in: {search_path}")

for root, dirs, files in os.walk(search_path):
    print(f"Checking directory: {root}")
    print(f"Files found: {files}")
    for file in files:
        file_path = os.path.join(root, file)
        if 'dcm' in file_path:
            dcm_file.append(file_path)

print(f"Total DICOM files found: {len(dcm_file)}")
if len(dcm_file) > 0:
    print(dcm_file[0])
else:
    print("No DICOM files found!")

# 查看指定目录下所有的dcm文件的 SOP Instance UID
dcm_file_uid=[]
if len(dcm_file) > 0:
    for dcm in dcm_file:
        try:
            im=pydicom.dcmread(dcm)
            dcm_uid=im.SOPInstanceUID
            dcm_file_uid.append(dcm_uid)
        except Exception as e:
            print(f"Error reading {dcm}: {e}")
    
    if len(dcm_file_uid) > 0:
        print(dcm_file_uid[0])
    else:
        print("No UIDs extracted!")
else:
    print("No DICOM files to process for UID extraction.")

# 查看指定Annotation的指定目录下所有的XML文件
xml_file=[]
xml_search_path = r'D:\GitHub\chest-ct-report-generator\datasets\Lung-PET-CT-Dx\Lung-PET-CT-Dx-Annotations-XML-Files-rev12222020\Annotation\A0001'
print(f"Searching for XML files in: {xml_search_path}")

for root, dirs, files in os.walk(xml_search_path):
    print(f"Checking directory: {root}")
    print(f"Files found: {files}")
    for file in files:
        file_path = os.path.join(root, file)      
        if 'xml' in file_path:
            xml_file.append(file_path)

print(f"Total XML files found: {len(xml_file)}")
if len(xml_file) > 0:
    print(xml_file[0])
else:
    print("No XML files found!")

# 從XML文件名中提取SOP Instance UID
xml_uids = []
for xml_path in xml_file:
    # XML文件名就是SOP Instance UID（去掉.xml擴展名）
    filename = os.path.basename(xml_path)
    uid = filename.replace('.xml', '')
    xml_uids.append(uid)

print(f"\nXML文件中的UID數量: {len(xml_uids)}")
print(f"第一個XML UID: {xml_uids[0] if xml_uids else 'None'}")

# 找出有對應XML標注的DICOM文件
matched_pairs = []
unmatched_dcm = []
unmatched_xml = []

print(f"\n開始匹配DICOM文件和XML標注...")
print(f"DICOM UID總數: {len(dcm_file_uid)}")
print(f"XML UID總數: {len(xml_uids)}")

# 建立XML UID的集合，提高查找效率
xml_uid_set = set(xml_uids)

for i, dcm_uid in enumerate(dcm_file_uid):
    if dcm_uid in xml_uid_set:
        # 找到對應的XML文件
        xml_index = xml_uids.index(dcm_uid)
        matched_pairs.append({
            'dcm_file': dcm_file[i],
            'dcm_uid': dcm_uid,
            'xml_file': xml_file[xml_index]
        })
    else:
        unmatched_dcm.append({
            'dcm_file': dcm_file[i],
            'dcm_uid': dcm_uid
        })

# 找出沒有對應DICOM的XML文件
dcm_uid_set = set(dcm_file_uid)
for i, xml_uid in enumerate(xml_uids):
    if xml_uid not in dcm_uid_set:
        unmatched_xml.append({
            'xml_file': xml_file[i],
            'xml_uid': xml_uid
        })

# 顯示匹配結果
print(f"\n=== 匹配結果 ===")
print(f"成功匹配的對數: {len(matched_pairs)}")
print(f"沒有標注的DICOM文件: {len(unmatched_dcm)}")
print(f"沒有對應DICOM的XML文件: {len(unmatched_xml)}")

# 顯示前5個匹配的對
print(f"\n=== 前5個匹配的DICOM-XML對 ===")
for i, pair in enumerate(matched_pairs[:5]):
    print(f"\n匹配對 {i+1}:")
    print(f"  DICOM: {os.path.basename(pair['dcm_file'])}")
    print(f"  XML: {os.path.basename(pair['xml_file'])}")
    print(f"  UID: {pair['dcm_uid']}")

# 儲存匹配結果到文件
output_file = "matched_dicom_xml_pairs.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("=== DICOM-XML 匹配結果 ===\n\n")
    f.write(f"總匹配對數: {len(matched_pairs)}\n")
    f.write(f"沒有標注的DICOM: {len(unmatched_dcm)}\n")
    f.write(f"沒有對應DICOM的XML: {len(unmatched_xml)}\n\n")
    
    f.write("=== 所有匹配的對 ===\n")
    for i, pair in enumerate(matched_pairs):
        f.write(f"\n匹配對 {i+1}:\n")
        f.write(f"  DICOM文件: {pair['dcm_file']}\n")
        f.write(f"  XML文件: {pair['xml_file']}\n")
        f.write(f"  SOP Instance UID: {pair['dcm_uid']}\n")
    
    if unmatched_dcm:
        f.write(f"\n=== 沒有標注的DICOM文件 ===\n")
        for dcm in unmatched_dcm:
            f.write(f"  {dcm['dcm_file']} (UID: {dcm['dcm_uid']})\n")
    
    if unmatched_xml:
        f.write(f"\n=== 沒有對應DICOM的XML文件 ===\n")
        for xml in unmatched_xml:
            f.write(f"  {xml['xml_file']} (UID: {xml['xml_uid']})\n")

print(f"\n結果已保存到: {output_file}")