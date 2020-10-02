import os 
def getpath(path):
    fileList = os.listdir()
    with open("filePate",'r',encoding='utf-8') as ftp:
        for i in fileList:
            ftp.writelines(i + '  ' + path+ '\\' + i)

if __name__ == "__main__":
    path = r""
    getpath(path)