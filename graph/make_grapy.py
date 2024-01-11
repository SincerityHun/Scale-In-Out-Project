import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 엑셀 파일을 읽어오는 코드 (파일 경로는 예시입니다)
# df = pd.read_excel('path_to_your_file.xlsx')

# 가상의 데이터를 생성합니다 (실제 데이터로 대체될 예정)
data = {
    'Model': ['ResNet50', 'ResNet50', 'ResNet50', 'ResNet50'],
    'Batch Size': [32, 32, 64, 64],
    'GPU at tb0': [1, 2, 1, 2],
    'GPU at tb1': [1, 1, 1, 1],
    'Time per Iterate (sec)': [0.5, 0.3, 0.4, 0.2]
}
df = pd.DataFrame(data)

# 모델과 배치 사이즈에 따른 'Time per Iterate' 그래프
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Time per Iterate (sec)', hue='Batch Size', data=df)
plt.title('Time per Iterate by Model and Batch Size')
plt.xlabel('Model')
plt.ylabel('Time per Iterate (sec)')
plt.show()

# GPU 할당 구성에 따른 'Time per Iterate' 그래프
df['GPU Configuration'] = df['GPU at tb0'].astype(str) + ' at tb0, ' + df['GPU at tb1'].astype(str) + ' at tb1'
plt.figure(figsize=(10, 6))
sns.barplot(x='GPU Configuration', y='Time per Iterate (sec)', data=df)
plt.title('Time per Iterate by GPU Configuration')
plt.xlabel('GPU Configuration')
plt.ylabel('Time per Iterate (sec)')
plt.xticks(rotation=45)
plt.show()
