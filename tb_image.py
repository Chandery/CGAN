from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 初始化EventAccumulator，并加载事件文件
size_guidance = {'images': 20}
event_acc = EventAccumulator('tb_logs/my_model/version_16/events.out.tfevents.1730451332.ubuntu.1947576.0', size_guidance=size_guidance)
event_acc.Reload()

print("开始")

# 获取图像数据
image_data = event_acc.Images('Generated Images')

# 使用PIL或其他库将图像数据转换为图片
from PIL import Image
import io

for index, data in enumerate(image_data):
    print('Step: ', data.step)
    image = Image.open(io.BytesIO(data.encoded_image_string))
    image.save(f'image/image_{index}.png')