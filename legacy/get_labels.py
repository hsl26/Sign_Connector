import os
import json

# 폴더 경로
folder_path = 'morpheme/01'

# 단어들을 저장할 리스트
word_list = []

# 파일 이름 얻어오기
file_names = [f for f in os.listdir(folder_path) if f.endswith('.json') and "F_morpheme" in f]

# 파일 이름을 번호 순서대로 정렬하기
file_names.sort(key=lambda x: int(x.split('_')[2][4:]))

for filename in file_names:
    file_path = os.path.join(folder_path, filename)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
        # 'data' 키 안의 요소들 순회
        for item in data['data']:
            for attribute in item['attributes']:
                word_list.append(attribute['name'])

# 결과 출력 
print(len(word_list))

# 필요한 경우 리스트를 파일로 저장할 수 있습니다.
with open('output_words.txt', 'w', encoding='utf-8') as output_file:
    for word in word_list:
        output_file.write(word + '\n')
