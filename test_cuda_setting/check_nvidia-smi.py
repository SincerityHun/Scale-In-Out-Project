import subprocess

# nvidia-smi 명령어를 실행하여 GPU 상태 확인
try:
    result = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
    print(result)
except subprocess.CalledProcessError as e:
    print("CUDA가 제대로 설정되지 않았습니다: ", e)
