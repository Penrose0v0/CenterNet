import os

prj = 'cat'
names = ['image', 'annotation']
lbls = ['train', 'val']
# 要处理的文件夹路径

for lbl in lbls:
    for name in names:
        folder_path = f'.\\{prj}\\{lbl}\\{name}\\'
        file_names = os.listdir(folder_path)

        prefix = f'.\\dataset\\{prj}\\{lbl}\\{name}\\'

        # 打开一个txt文件用于写入文件名
        with open(f'./{prj}/{lbl}/{name}.txt', 'w', encoding='utf-8') as file:
            for filename in file_names:
                file.write(prefix + filename + '\n')

    print(f"文件名已写入 {lbl}")
