# 전체 학습 시뮬레이터
import subprocess
import argparse
# Training settings

# 학습 스크립트 실행
training_process = subprocess.Popen(["horovodrun", "-np", "1", "python", "pytorch_mnist_elastic.py", "--batch-size", "128"])

# 모니터 스크립트 실행
monitor_process = subprocess.Popen(["python", "monitor.py"])

# 두 프로세스가 종료될 때까지 대기
training_process.wait()
monitor_process.wait()
