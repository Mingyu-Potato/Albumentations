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


# # 파일 이름에 따른 파일 저장
# file_list = os.listdir('./SFR-03(Abnormal)/Sagging_Image')

# for f in file_list:
#     filename, file_extension = os.path.splitext(f)
#     # print("파일명: ", f, "     확장자: ", file_extension)
#     if filename[-6:] == 'NORMAL':
#         file_path = os.path.join('./SFR-03(abnormal)/Sagging_Image', f)
#         new_file_path = os.path.join('./SFR-03(abnormal)/NORMAL', f)
#         print("파일 경로 + 파일명 = ", file_path)
        
#         shutil.copy(file_path, new_file_path)


# # 파일 이름 바꾸기
# file_list = os.listdir('./image/SFR-03(Normal)/원본')

# for f in file_list:
#     filename, file_extension = os.path.splitext(f)
#     file_num = filename[5:]
#     # print("파일명: ", f, "     확장자: ", file_extension)
#     if filename[:4] == 'HARD':
#         new_file_name = "HIGH_" + file_num + ".jpg"
#         file_old_name = os.path.join('./image/SFR-03(abnormal)/포토샵작업_처짐_Aug', f)
#         file_new_name = os.path.join('./image/SFR-03(abnormal)/포토샵작업_처짐_Aug', new_file_name)
#         print("파일 경로 + 파일명 = ", file_new_name)
        
#         os.rename(file_old_name, file_new_name)

file_list = os.listdir('./image/SFR-03(Normal)/원본')

for f in file_list:
    filename, file_extension = os.path.splitext(f)
    print("파일명: ", f, "     확장자: ", file_extension)
    new_file_name = filename + ".jpg"
    file_old_name = os.path.join('./image/SFR-03(Normal)/원본', f)
    file_new_name = os.path.join('./image/SFR-03(Normal)/원본_Aug', new_file_name)
    print("파일 경로 + 파일명 = ", file_new_name)
    
    shutil.copy(file_old_name, file_new_name)