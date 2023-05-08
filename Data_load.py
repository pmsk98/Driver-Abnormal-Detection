# -*- coding: utf-8 -*-
"""
Created on Mon May  8 23:37:11 2023

@author: user
"""
import tensorflow

import tarfile

#%% 압축해제

# # tar 파일 경로
# tar_file_path = "C:/Users/user/Desktop/capstone/089.차량 내 탑승자 상황 인식 영상 데이터/01.데이터/2.Validation/라벨링데이터/abnormal_230303_add/VL1.tar"

# # tar 파일 객체 생성
# tar = tarfile.open(tar_file_path)

# # 압축 해제
# tar.extractall('C:/Users/user/Desktop/capstone/089.차량 내 탑승자 상황 인식 영상 데이터/01.데이터/2.Validation/라벨링데이터/abnormal_230303_add/')

# # tar 파일 객체 닫기
# tar.close()




# # tar 파일 경로
# tar_file_path = "C:/Users/user/Desktop/capstone/089.차량 내 탑승자 상황 인식 영상 데이터/01.데이터/2.Validation/원천데이터/abnormal_230303_add/VS1.tar"

# # tar 파일 객체 생성
# tar = tarfile.open(tar_file_path)

# # 압축 해제
# tar.extractall('C:/Users/user/Desktop/capstone/089.차량 내 탑승자 상황 인식 영상 데이터/01.데이터/2.Validation/라벨링데이터/abnormal_230303_add/')

# # tar 파일 객체 닫기
# tar.close()


#%% 라벨 데이터 프레임 생성

#SGA2100920 <- 이폴더 경로 잘못들어가있어서 바꿨음 (SGA2101523 이폴더 안에 들어가 있길래 밖으로 뺐음(날짜보니까 잘못 들어간 데이터 같기도하고) (SGA2100920S0108 여기엔 라벨도 없음))
import json
import os
from glob import glob
import pandas as pd

os.chdir('C:\\Users\\user\\Desktop\\capstone\\089.차량 내 탑승자 상황 인식 영상 데이터\\01.데이터\\2.Validation\\라벨링데이터\\abnormal_230303_add\\')

#상위파일목록
path = glob("*")
path.remove('VL1.tar')
path.remove('logs')
path.remove('model.hdf5')
# path.remove('yolov5s.pt')
#path.remove("SGA2100920") #<- 얘 하위 폴더에 라벨 없는 데이터 몇개있음

#빈 df 생성
result_df = pd.DataFrame(columns=["상위파일", "id", "label"]) 
new_df = pd.DataFrame(columns=["상위파일", "ECG", "EEG_0", "EEG_1", "PPG", "SPO2", "emotion"]) 


full_image_df = pd.DataFrame(columns=["상위파일", "id", "label",'img_name','body_b_box']) 

for i in path:
    #중간파일 목록
    new_path = os.listdir("C:\\Users\\user\\Desktop\\capstone\\089.차량 내 탑승자 상황 인식 영상 데이터\\01.데이터\\2.Validation\\라벨링데이터\\abnormal_230303_add\\" + i)
    
    for j in new_path:
        label_path = os.listdir("C:\\Users\\user\\Desktop\\capstone\\089.차량 내 탑승자 상황 인식 영상 데이터\\01.데이터\\2.Validation\\라벨링데이터\\abnormal_230303_add\\" + i +"\\"+ j + "\\label")
        
        with open(("C:\\Users\\user\\Desktop\\capstone\\089.차량 내 탑승자 상황 인식 영상 데이터\\01.데이터\\2.Validation\\라벨링데이터\\abnormal_230303_add\\" + i +"\\"+ j + "\\label\\"+ label_path[0]), 'r', encoding='UTF8') as f:
            json_data = json.load(f)
            
            print("상위파일:", i, "id :" , json_data["scene_info"]['scene_id'] , "라벨명:",json_data["scene_info"]["category_name"])
            
            # #result_df
            # result_list = []
            # result_list.append([i,
            #                     json_data["scene_info"]['scene_id'],
            #                     json_data["scene_info"]["category_name"]])
            img_name = []
            body_b_box =[]
            ecg=[]
            eeg=[]
            eeg_1=[]
            ppg=[]
            # # df = pd.DataFrame(result_list, columns=["상위파일", "id", "label"])
            
            # for e in range(len(json_data)):
            #     img_name.append(json_data["scene"]['data'][e]['img_name'])
            #     body_b_box.append(json_data["scene"]['data'][e]['occupant'][0]['body_b_box'])
                

            
            result_list = [[i, json_data["scene_info"]['scene_id'], json_data["scene_info"]["category_name"], img_name, body_b_box,ecg,eeg,eeg_1,ppg] for e in range(len(json_data))]

            for e in range(len(json_data)):
                result_list[e][3] = json_data["scene"]['data'][e]['img_name']
                result_list[e][4] = json_data["scene"]['data'][e]['occupant'][0]['body_b_box']
                result_list[e][5] = json_data["scene"]["sensor"][0]["ECG"]
                result_list[e][6] = json_data["scene"]["sensor"][0]["EEG"][0]
                result_list[e][7] = json_data["scene"]["sensor"][0]["EEG"][1]
                result_list[e][8] = json_data["scene"]["sensor"][0]["PPG"]

            df = pd.DataFrame(result_list, columns=["상위파일", "id", "label", "img_name", "body_b_box",'ecg','eeg','eeg_1','ppg'])
            result_df = pd.concat([result_df, df])
                

            # #new_df 
            # new_list = []
            # new_list.append([i,
            #                  json_data["scene"]["sensor"][0]["ECG"], 
            #                  json_data["scene"]["sensor"][0]["EEG"][0],
            #                  json_data["scene"]["sensor"][0]["EEG"][1],
            #                  json_data["scene"]["sensor"][0]["PPG"],
            #                  json_data["scene"]["sensor"][0]["SPO2"],
            #                  json_data["scene"]["data"][0]["occupant"][0]["emotion"]
            #                  ])
            # df_2 = pd.DataFrame(new_list,columns=["상위파일", "ECG", "EEG_0", "EEG_1", "PPG", "SPO2", "emotion"] )
            # new_df = pd.concat([new_df,df_2])


# result_df.info()
# result_df.describe()
# result_df["label"].unique()

# json_data["scene"]["sensor"][0]["ECG"]
# json_data["scene"]["sensor"][0]["EEG"][0]
# json_data["scene"]["sensor"][0]["EEG"][1]
# json_data["scene"]["sensor"][0]["PPG"]
# json_data["scene"]["sensor"][0]["SPO2"]
# json_data["scene"]["data"][0]["occupant"][0]["emotion"]



result_df["label"].value_counts()


result_df_copy = result_df.copy()


result_df_copy = result_df_copy.reset_index(drop=True)

result_df_copy.columns

result_df_copy['image_path'] = 'C:/Users/user/Desktop/capstone/089.차량 내 탑승자 상황 인식 영상 데이터/01.데이터/2.Validation/라벨링데이터/abnormal_230303_add/' + result_df_copy['상위파일'] + "/" + result_df_copy['id'] + "/img/" + result_df_copy['img_name']

# full_image['b_box'] = result_df['body_b_box']

result_df_copy['label'] = result_df_copy['label']

result_df_copy['id'] = result_df_copy['id']




#%% 새로 받은 train data 라벨링 설정


os.chdir('C:\\Users\\user\\Desktop\\capstone\\089.차량 내 탑승자 상황 인식 영상 데이터\\01.데이터\\1.Training\\라벨링데이터\\abnormal_230303_add\\TL1')

path = glob("*")

# path.remove('TL1.tar')
# path.remove('TL2.tar')
# path.remove('TL3.tar')
# path.remove('TL4.tar')
# path.remove('TL5.tar')
# path.remove('TL6.tar')
# path.remove('TL7.tar')


#빈 df 생성
result_train = pd.DataFrame(columns=["상위파일", "id", "label"]) 
new_train = pd.DataFrame(columns=["상위파일", "ECG", "EEG_0", "EEG_1", "PPG", "SPO2", "emotion"]) 


full_image_train = pd.DataFrame(columns=["상위파일", "id", "label",'img_name','body_b_box']) 



for i in path:
    #중간파일 목록
    new_path = os.listdir("C:\\Users\\user\\Desktop\\capstone\\089.차량 내 탑승자 상황 인식 영상 데이터\\01.데이터\\1.Training\\라벨링데이터\\abnormal_230303_add\\TL1\\" + i)
    
    for j in new_path:
        label_path = os.listdir("C:\\Users\\user\\Desktop\\capstone\\089.차량 내 탑승자 상황 인식 영상 데이터\\01.데이터\\1.Training\\라벨링데이터\\abnormal_230303_add\\TL1\\" + i +"\\"+ j + "\\label")
        
        with open(("C:\\Users\\user\\Desktop\\capstone\\089.차량 내 탑승자 상황 인식 영상 데이터\\01.데이터\\1.Training\\라벨링데이터\\abnormal_230303_add\\TL1\\" + i +"\\"+ j + "\\label\\"+ label_path[0]), 'r', encoding='UTF8') as f:
            json_data = json.load(f)
            
            print("상위파일:", i, "id :" , json_data["scene_info"]['scene_id'] , "라벨명:",json_data["scene_info"]["category_name"])
            
            # #result_df
            # result_list = []
            # result_list.append([i,
            #                     json_data["scene_info"]['scene_id'],
            #                     json_data["scene_info"]["category_name"]])
            img_name = []
            body_b_box =[]
            ecg=[]
            eeg=[]
            eeg_1=[]
            ppg=[]
            # # df = pd.DataFrame(result_list, columns=["상위파일", "id", "label"])
            
            # for e in range(len(json_data)):
            #     img_name.append(json_data["scene"]['data'][e]['img_name'])
            #     body_b_box.append(json_data["scene"]['data'][e]['occupant'][0]['body_b_box'])
                

            
            result_list = [[i, json_data["scene_info"]['scene_id'], json_data["scene_info"]["category_name"], img_name, body_b_box,ecg,eeg,eeg_1,ppg] for e in range(len(json_data))]

            for e in range(len(json_data)):
                result_list[e][3] = json_data["scene"]['data'][e]['img_name']
                result_list[e][4] = json_data["scene"]['data'][e]['occupant'][0]['body_b_box']
                result_list[e][5] = json_data["scene"]["sensor"][0]["ECG"]
                result_list[e][6] = json_data["scene"]["sensor"][0]["EEG"][0]
                result_list[e][7] = json_data["scene"]["sensor"][0]["EEG"][1]
                result_list[e][8] = json_data["scene"]["sensor"][0]["PPG"]

            df = pd.DataFrame(result_list, columns=["상위파일", "id", "label", "img_name", "body_b_box",'ecg','eeg','eeg_1','ppg'])
            result_train = pd.concat([result_train , df])
                



result_train["label"].value_counts()


result_train_copy = result_train.copy()


result_train_copy  = result_train_copy.reset_index(drop=True)

result_train_copy['image_path'] = 'C:/Users/user/Desktop/capstone/089.차량 내 탑승자 상황 인식 영상 데이터/01.데이터/1.Training/원천데이터/abnormal_230303_add/' + result_train_copy['상위파일'] + "/" + result_train_copy['id'] + "/img/" + result_train_copy['img_name']

# full_image['b_box'] = result_df['body_b_box']

result_train_copy['label'] = result_train_copy['label']

result_train_copy['id'] = result_train_copy['id']


result_train_copy["label"].value_counts()



#%%클래스 균일하게 맞추기


#졸음운전 (6만개 추출)
# df_1 = result_train_copy[result_train_copy["label"]=="졸음운전"].sample(n=60000)
# final_df = pd.concat([df_1, result_df_copy])

# final_df["label"].value_counts()

 
# #통화 (21000개 추출)
# df_2 = result_train_copy[result_train_copy["label"]=="통화"].sample(n=21000)
# final_df = pd.concat([df_2, final_df])

# final_df["label"].value_counts()

# #물건찾기 (13000개 추출)
# df_3 = result_train_copy[result_train_copy["label"]=="물건찾기"].sample(n=13000)
# final_df = pd.concat([df_3, final_df])

# final_df["label"].value_counts()

# #차량제어 (30000개 추출)
# df_4 = result_train_copy[result_train_copy["label"]=="차량제어"].sample(n=30000)
# final_df = pd.concat([df_4, final_df])

# final_df["label"].value_counts()

# #휴대폰조작 (36000개 추출)
# df_5 = result_train_copy[result_train_copy["label"]=="휴대폰조작"].sample(n=36000)
# final_df = pd.concat([df_5, final_df])

# final_df["label"].value_counts()


#%%
#validation set에 졸음운전만 추가
df_1 = result_train_copy[result_train_copy["label"]=="졸음운전"].sample(n=32000)
final_df = pd.concat([df_1, result_df_copy])


final_df['label'].value_counts()

final_df = final_df.reset_index(drop=True)

