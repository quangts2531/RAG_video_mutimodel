import os

folder_path = "extract_media/video"

# 2. Khai báo tên file txt sẽ lưu kết quả
output_file = 'video_ids.txt'

# 3. Quét và lưu tên file
with open(output_file, 'w', encoding='utf-8') as f:
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            f.write(filename.replace(".mp4" , "") + '\n')

print("Đã lưu danh sách tên file thành công!")
