import csv
import os

""" convert .csv content to python list """
def csv_to_list(path):
    res = []
    with open(path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        # 以迴圈輸出每一列
        for row in rows:
            res.append(row)
    return res
        
if __name__=="__main__":
    
    path = './tasks/data/mask/test/mask/result.csv'
    results = csv_to_list(path)
    for res in results:
        file_path, classes, prob = res
        if float(prob) >= 0.99:
            print(file_path, classes, prob)

    # 開啟 CSV 檔案
    # with open('./tasks/data/mask/test/mask/result.csv', newline='') as csvfile:

    #     # 讀取 CSV 檔案內容
    #     rows = csv.reader(csvfile)

    #     # 以迴圈輸出每一列
    #     for row in rows:
    #         full_path, classes, prob = row[:]
    #         file_name = os.path.basename(full_path) 
    #         print(file_name, classes, prob)  
        