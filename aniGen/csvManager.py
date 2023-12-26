import csv

def clear_csv_file(file_path = 'dataset/pose-record.csv'):
    try:
        # 파일을 쓰기 모드로 열어 빈 내용으로 덮어쓰기
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 아무 내용도 쓰지 않음
        print(f"{file_path} 내용을 대체합니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

def add_csv(lst, file_path = 'dataset/pose-record.csv'):
    f = open(file_path, 'a', newline='')
    wr = csv.writer(f)
    for frame in lst:
        tmp = [frame[0]]
        for pos in frame[1]:
            if pos:
                tmp.append(pos[0])
                tmp.append(pos[1])
            else:
                tmp.append('none')
                tmp.append('none')

        wr.writerow(tmp)

def add_line(lst, file_path):
    f = open(file_path, 'a', newline='')
    wr = csv.writer(f)
    wr.writerow(lst)
