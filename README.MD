# StyleTTS: A NEW APPROACH TO TTS (Text To Speech) THAT CAN GENERATE MORE NATURAL AND DIVERSE SPEECH

File báo cáo, Present và Script của code này được để trong Folder "document". Các bước cài đặt bằng hình ảnh được để trong thư mục assets gồm nhiều thư mục con với mỗi thư mục con được đặt tên từng vấn đề

Để có thể chạy được code trên vui lòng thực hiện theo từng bước sau:

## Cài đặt thư viện và môi trường
Project này sử dụng phiên bản Python 3.9 vui lòng cài đặt [tại đây](https://www.python.org/downloads/release/python-390/#:~:text=Full%20Changelog-,Files,-Version). Sẽ tốt hơn nếu sử dụng môi trường conda để để bắt đầu với project này, hướng dẫn cài đặt conda và tạo môi trường python3.9 được giải quyết [tại đây](RS2023/asset/install_conda_setup_python)

Sau khi đã giải quyết xong vấn đề về phiên bản python thì tiếp là vấn đề về các phiên bản thư viện được sử dụng trong Project. Để giải quyết vấn đề này cần xác định rõ đường dẫn chính xác của Project (Thường có dạng cây thự mục như hình)
![Hình thư mục](asset/other/Screenshot 2023-11-03 at 12.44.42.png)

Khi đã đảm bảo xác định được đường dẫn thì cần mở terminal và thực hiện lệnh sau:
```bash
cd  /Users/Momo_employeee/Desktop/project/RS2023
pip install -r requirement.txt
```
Ngoài ra, còn phải cài thêm 1 thư viện ngoài là [espeak-ng](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md#windows). Nếu sử dụng HĐH Window chỉ cần nhấp vào [đây](https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X64.msi). Sau đó cài đặt bình thường.

Nếu sử dụng HĐH Linux thì cần chạy lệnh sau:
```bash
sudo apt-get install espeak-ng
```


P/s: Chú ý tên các folder nếu chứa dấu cách sẽ có thể gây ra 1 số trường hợp lỗi về đường dẫn có thể khắc phục bằng các thay đổi tên hoặc đưa path vào trong cặp dấu ""
```bash
cd  "/Users/Desktop/Nghiên cứu khoa học/RS2023"
```

Chương trình có sẵn Demo và API. Các bước triển khai sẽ được hướng dẫn trong phần hướng dẫn sử dụng. Phía trên là tổng quan về các vấn đề có thể gặp khi triển khai chương trình có thể tuỳ vào cấu hình môi trường của từng máy sẽ gặp những lỗi đặc biệt hơn. Nếu có bất cứ vấn đề nào có thể thử tìm kiếm các nguồn tài liệu có sẵn trên Internet, nếu đặc biệt không tìm thấy giải pháp cho vấn đề trên có thể gmail đính kèm hình ảnh issues qua gmail được để ở phần Contact (Lưu ý nhớ chụp rõ lỗi và check mail phản hồi thường xuyên và kiểm tra thùng thư rác)
## Hướng dẫn sử dụng
### Cách chạy chương trình API (Application Programming Interface)
Việc chạy chương trình thông qua API có thể được sử dụng cho các Web service. Nếu muốn xây dựng chương trình Web Application thì cần phát triển giao diện Web rồi gọi API để chạy chương trình. 

Có thể test API bằng thư viện fastAPI để test. Dưới đây là hướng dẫn về việc chạy và kiểm thử API bằng fastAPI trên terminal.

Đầu tiên cần cấu trúc của source code có 2 file chính là [main_API.py](main_API.py) và [main_program.py](main_program.py). Và đây là hướng dẫn về phần chạy API nên mục tiêu chính là file [main_API.py](main_API.py). Kiểm tra tổng quan file [main_API.py](main_API.py) có thể thấy bố cục của code được viết khá ổn định và cũng khá dễ hiểu. Vậy để có thể triển khai và chạy chương trình đầu tiên cần mở [terminal](asset/open_terminal_(DONE)) và thực hiện tuần tự các lệnh sau:

```bash
#This way used for Linux OS or Window OS

uvicorn main_API:app --host 0.0.0.0 --port 8040

#This way used for MacOS

python3 -m uvicorn main_API:app --host 0.0.0.0 --port 8040

# Chổ param --host và --port có thể thay đổi để phù hợp với máy
# Ví dụ: (Nếu máy cá nhân triển khai project không có sẵn host hay port có thể đổi thành như sau)

uvicorn main_API:app --host 0.0.0.0 --port 5050
```
Nếu không có bất kỳ lỗi gì xuất hiện thì terminal sẽ như này
![Image success run FastAPI](asset/run_api_fastAPI_(DONE)/step_2.png "Success Implement API")

Nếu để ý có thể thấy được 1 URL xuất hiện URL. Cụ thể là http://0.0.0.0:8040. Để có thể test API hãy truy cập URL đó. Đây là màn hình của Website khi truy cập vào URL đó
![Image success run FastAPI](asset/run_api_fastAPI_(DONE)/step_3.png "Success Implement API")

Cần thay đổi URL thành http://0.0.0.0:8040/docs màn hình sẽ chuyển thành như sau
![Image success run FastAPI](asset/run_api_fastAPI_(DONE)/step_4.png "Success Implement API")

Thấy có 2 component, Hãy show component <b> POST /CloneAudio </b> sẽ thấy được như màn hình sau khi đó cần chọn vào nút <b> Try it out </b> bên góc gần phía bên phải 
![Image success run FastAPI](asset/run_api_fastAPI_(DONE)/step_5.png "Success Implement API")

Sau khi hoàn thành các bước trên bấy giờ sẽ thấy xuất hiện 1 ô nhập đoạn text và 1 ô để chọn file. Ô nhập text chính là ô để nhập văn bản cần xử lý, còn ô chọn file thì cần chọn 1 file audio để sao chép giọng nói. Có thể xem ví dụ minh hoạ như trong hình
![Image success run FastAPI](asset/run_api_fastAPI_(DONE)/step_7.png "Success Implement API")

Sau khi hoàn thành các bước trên nhấn vào nút Execute để tiến hành tổng hợp và sao chép giọng nói. Quá trình trên có thể mất 1 khoảng thời gian để thực thi và tiêu hao 1 lượng tài nguyên nhất định. Màn hình hoàn thành như sau và kết quả có được kiểm tra trong thư mục [result](result) hoặc chọn <b><u>Download file</u></b> để tải audio về máy
![Image success run FastAPI](asset/run_api_fastAPI_(DONE)/step_8.png "Success Implement API")

Như vậy phía trên là tổng quan về phần khởi chạy Nghiên cứu trên qua API mọi document về mọi vấn đề đều được lưu trong [asset](asset) có thể vào đó để kiểm tra thêm. Hiện tại việc hướng dẫn khởi chạy API bằng hình ảnh cũng được lưu [ở đây](asset/run_api_fastAPI)

### Cách chạy chương trình với dữ liệu mẫu trên terminal (NO User Interface)

Tổng quan về phần này thì đây là phần khởi chạy chương trình trên dữ liệu tương đối sạch nên việc xử lý dữ liệu là không cần thiết. Những dữ liệu này được lưu bên trong thư mục [asset](asset/audio_to_test_(DONE)) là những dữ liệu hoàn toàn có thể chạy được trong phần này và cho ra những kết quả khá ổn định. Như vậy, thống nhất lại rằng việc chạy thử chương trình bằng file [main_program.py](main_program.py) sẽ dựa trên những file audio được lưu trữ trong folder [asset](asset/audio_to_test_(DONE)).

Để chạy test chương trình thì vẫn cần phải dùng terminal, về cách để mở terminal thì cần coi lại những hướng dẫn trên. Tiếp theo cần chạy lệnh sau để có thể chạy được chương trình

```bash
# Với MacOS
python3 main_program.py

# Với Linux OS, Window
python main_program.py
```
Nếu chạy thành công chương trình sẽ hiện như sau:
![success run program](asset/run_on_terminal_asset_(DONE)/step_1.png "Success Implement API")
Bây giờ giống như trên đọc kỹ yêu cầu. Đầu tiên là nhập vào đoạn text cần được nói sau đó nhấn Enter, tiếp theo nhập vào đường dẫn của audio trong folder [asset](asset/audio_to_test_(DONE)) màn hình sau khi xong sẽ như này.
![Image success program](asset/run_on_terminal_asset_(DONE)/step_3.png "Success Implement API")

Tương tự chờ trong khoảng 30 - 60 tuỳ độ dài đoạn text và độ dài của audio mà thời gian chờ có thể cao hơn dự kiến. Khi màn hình xuất hiện như sau là chương trình đã xong. Hãy kiểm tra thư mục [result](result) để xem kết quả.

### Cách chạy trên dữ liệu người dùng (File người dùng đưa vào)

Tương tự như cách trên chỉ khác biệt đây là 1 phiên bản ổn định và đầy đủ nhất với bộ xử lý được pretrain hoàn chỉnh. Ví dụ trong chương trình này được tích hợp bộ Denoiser để khử nhiễu và lọc âm thanh để đầu vào là một audio chuẩn single speaker. Kiến trúc Denoiser đã được được viết rõ trong bài báo cáo có thể xem qua [tại đây](document/Report_for_Research_Science.docx). Đây phần triển khai cuối cùng của nghiên cứu lần này và chi tiết của nó sẽ được mô tả sau đây

Đầu tiên, cần phải bật terminal lên giống như các ví dụ trước đây, Hướng dẫn về phần bật terminal có thể [tại đây](asset/open_terminal_(DONE)). Sau khi đã có terminal cần phải đảm bảo là đang ở trong cấu trúc thư mục của project và phải hoàn thành việc cài đặt thư viện [FFmpeg](https://ffmpeg.org/download.html) hướng dẫn về phần cài đặt này có thể xem ở trên các website hướng dẫn hoặc [tại đây](asset/Install_FFmpeg_(DONE)) (Vì việc cài đặt tương đối dễ dàng với các hệ điều hành khác nhưng lại tương đối khó khăn với window nên, ở đây chỉ có hướng dẫn cho cài đặt trên hệ thống của window). Thư viện [FFmpeg](https://ffmpeg.org/download.html) trên dùng để chuyển đổi qua lại giữa các định dạng file người dùng, nghĩa là mô hình chỉ hoạt động tương đối với các loại file có định dạng mp3 nhưng lại có chất lượng tốt hơn nếu đưa về định dạng file là wav. Vậy nên thư viện [FFmpeg](https://ffmpeg.org/download.html) sẽ giúp chuyển định dạng các file.

Sau khi đã hoàn thành các bước trên thì sẽ tới bước khởi chạy chương trình. Đầu tiên cần gõ lệnh sau trong terminal
```bash
# Với MacOS
python3 inference.py

# Với Linux OS, Window
python inference.py
```
Màn hình terminal sẽ hiện như thế này nếu được khởi động thành công.

![Image success run FastAPI](asset/run_inference_user_(DONE)/step_1.png "Success Implement API")

Tại phần nhập đầu tiên cần truyền vào audio mà người thực hiện muốn test. Lưu ý nên chọn audio có độ dài ngắn khoảng chừng từ 20s - 40s để tránh bị đứng máy vì toàn bộ code này được thực hiện trên CPU nên việc tính toán cũng được thực hiện trên CPU dẫn tới việc tính toán không song song nên việc dùng 1 audio có độ dài lớn sẽ dẫn đến việc chậm và đứng máy. Tuy nhiên, nếu chọn audio ngắn thì việc tính toán sẽ vẫn tiêu hao tài nguyên máy nhưng trong khoảng thời gian ngắn nên vô hại.
Sau khi chọn được video thì cần nhập đoạn text cần kiểm thử.
![Image success run FastAPI](asset/run_inference_user_(DONE)/step_2.png "Success Implement API")

Sau khi nhập đoạn text xong mọi thứ tiếp theo đều được làm tự động và chỉ cần chờ 1 vài phút kết quả sẽ được save tại thư mục [result](result)
Khi màn hình hiện như này thì nghĩa là code đã được thực thi xong và có thể nghe thử kết quả
![Image success run FastAPI](asset/run_inference_user_(DONE)/step_3.png "Success Implement API")
![Image success run FastAPI](asset/run_inference_user_(DONE)/step_4.png "Success Implement API")



### Nếu cần tăng tốc độ chạy chương trình thì có 1 vài bước trong code cần chuyển đổi

#### *****************************************************************************************************
<p>UPDATE </p>
Chương trình vừa hoàn thành việc setup GPU/CUDA nên để tăng tốc độ thực thi có thể tận dụng GPU/CUDA nếu máy có sẵn. Khi bắt đầu chạy chương trình hệ thống sẽ thông báo rằng có hãy chọn thiết bị muốn sử dụng là (cpu hoặc gpu). Lúc này chỉ cần nhập vào device muốn dùng phần còn lại hệ thống sẽ tự động sử lý.

![Image success run FastAPI](asset/use_GPU_CUDA/step_1.png "Success Implement API")

hoặc có thể fix trực tiếp trong code như sau

```python
import denoiser
import os
from pydub import AudioSegment
from scipy.io import wavfile
import threading
import torch
import torchaudio
from denoiser.dsp import convert_audio
import json
from bin.component import *
from hifi_gan.vocoder import Generator
from attrdict import AttrDict
import phonemizer
from bin.models import load_ASR_models, load_F0_models, build_model
from munch import Munch
import yaml
import glob
from denoiser import pretrained


model_denoiser = pretrained.dns64().cpu()
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

# load StyleTTS
model_path = "Models/LibriTTS/epoch_2nd_00050.pth"
model_config_path = "Models/LibriTTS/config.yml"
# device = input('Please enter your device (cpu/cuda): ')
# if device == 'cpu':
#     model_denoiser = pretrained.dns64().cpu()
# elif device == 'cuda':
#     model_denoiser = pretrained.dns64().cuda()

device = 'cuda'

# OR

device = 'cpu'
```

## Contact for help

If you have any question which can't found in my asset. You can send email to follow mail. I will reply it around 1 day

Mail to: nhanht.18@grad.uit.edu.vn

## License

[MIT](https://choosealicense.com/licenses/mit/)