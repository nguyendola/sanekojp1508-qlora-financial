## Ứng dụng nhận diện cảm xúc:

- positive
- negative
- neutral

Model đang dùng:
- `sanekojp1508/qlora-financial`

## Cấu trúc
- `app.py`: giao diện Streamlit và logic suy luận
- `requirements.txt`: thư viện cần cài
- `run.sh`: script chạy nhanh

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

https://skilrekscojfderhajurvy.streamlit.app/

## Cách dùng
1. Nhập nội dung tiếng Việt vào ô text.
2. Bấm **Submit**.
3. Màn hình sẽ hiển thị `positive`, `negative` hoặc `neutral`.

## Ghi chú 
Do train với số lượng epochs và batch_size thấp do hạn chế GPU nên đôi khi nhận diện không được tốt
