import shutil
import os

# # 확장자에 따른 파일 제거
# file_list = os.listdir('./SFR-03(abnormal)')

# for f in file_list:
#     filename, file_extension = os.path.splitext(f)
#     # print("파일명: ", f, "     확장자: ", file_extension)
#     if file_extension == '.JPG':
#         file_path = os.path.join('./SFR-03(abnormal)', f)
#         print("파일 경로 + 파일명 = ", file_path)
#         # os.remove(file_path)


# 파일 이름에 따른 파일 저장
file_list = os.listdir('./SFR-03(Abnormal)/Sagging_Image')

for f in file_list:
    filename, file_extension = os.path.splitext(f)
    # print("파일명: ", f, "     확장자: ", file_extension)
    if filename[-6:] == 'NORMAL':
        file_path = os.path.join('./SFR-03(abnormal)/Sagging_Image', f)
        new_file_path = os.path.join('./SFR-03(abnormal)/NORMAL', f)
        print("파일 경로 + 파일명 = ", file_path)
        
        shutil.copy(file_path, new_file_path)