## Ứng dụng nhận diện cảm xúc:

- positive
- negative
- neutral

Model đang dùng:
- `sanekojp1508/cyber_abte`

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

## Cách dùng
1. Nhập nội dung tiếng Việt vào ô text.
2. Bấm **Submit**.
3. Màn hình sẽ hiển thị `positive`, `negative` hoặc `neutral`.

## Ghi chú
Nếu repo model private, bạn cần đăng nhập Hugging Face trước khi chạy.
