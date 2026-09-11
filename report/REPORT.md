# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không 

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id: 468`, `class_name: cab`, `rank: 1`, `score: 0.510915`, `taxonomy_name: ImageNet-1K`):
- Record này mô tả toàn ảnh là nhãn có xác suất cao nhất được mô hình dự đoán cho toàn bộ khung hình (xe taxi).
- Class list do cộng đồng ImageNet và nhóm tác giả mô hình định nghĩa.
- Cần giữ ID để máy xử lý, tên lớp cho người đọc và taxonomy để tránh nhầm lẫn giữa các bộ quy tắc nhãn khác nhau.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định chọn đối tượng chính, chiếm diện tích lớn hoặc nằm trung tâm.
- Model score là xác suất thống kê của máy(luôn luôn có tỉ lệ sai sót), không phải sự thật khách quan đã được con người xác nhận (ground truth).

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name: person`, `score: 0.825852`, `bbox_xyxy: [419.86, 172.6, 640.0, 427.0]`, `width: 220.14`, `height: 254.4`):
- Diễn giải vị trí box bằng lời: Nằm ở phía dưới bên phải của ảnh.
    Theo chiều ngang (Trục X - Trái sang Phải): Vật thể nằm ở nửa bên phải của bức ảnh. Cạnh trái của hộp bắt đầu ở vị trí 385.33 pixel (đã vượt qua điểm chính giữa ảnh là 320) và cạnh phải kết thúc ở 498.92 pixel.

    Theo chiều dọc (Trục Y - Trên xuống Dưới): Vật thể trải dài gần hết chiều cao của ảnh. Phần đỉnh (cạnh trên) bắt đầu gần sát mép trên ở tọa độ 69.24, và phần đáy (cạnh dưới) kéo dài xuống gần sát mép dưới ở tọa độ 348.92 (trên tổng số 427 pixel chiều cao).

    Hình dáng: Hộp giới hạn này là một hình chữ nhật đứng, khá hẹp về chiều ngang (rộng 113.58 pixel) nhưng rất cao (cao 279.68 pixel).
- So sánh số prediction ở hai threshold: Ngưỡng 0.20 có 17 vật thể, ngưỡng 0.60 có 6 vật thể.
- Khi ngưỡng thấp, độ bao phủ tăng (ít sót) nhưng Reviewer phải xem nhiều dự đoán sai hơn. Ngưỡng cao thì ngược lại.
- Đề xuất một quy tắc box chặt: Box phải ôm sát các điểm ngoài cùng của vật thể, sai số không quá 2 pixel.
- Với vật thể bị che khuất, cần guideline quy định có vẽ box cho phần bị ẩn hay không, hoặc báo cáo (escalation) nếu không thể xác định loại vật thể.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id: kitchen-001`, `class_name: person`, `score: 0.826`, `polygon_point_count: 57`): 
- Polygon bổ sung chi tiết về hình dạng (shape) và ranh giới chính xác giữa vật thể và nền so với box chỉ là hình chữ nhật.
- `instance_id` dùng để phân biệt các thực thể riêng biệt của cùng một lớp, không phải là Class ID.
- Đề xuất một quy tắc biên mask: Đường biên phải bám sát theo viền vật thể, không để lẹm vào trong hoặc thừa ra ngoài quá nhiều.
- Vùng mờ/che khuất cần quy định rõ việc ước lượng biên hoặc chỉ gán nhãn phần nhìn thấy rõ.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Lớp (String/ID) | Nhầm lẫn xe bus/taxi | Chọn 1 nhãn đúng nhất | Kiểm tra nhãn đúng chủ thể |
| Phát hiện vật thể | Bounding Box (xyxy) | Box bị lỏng/thiếu vật thể | Vẽ hộp bao sát vật thể | Kiểm tra độ chặt và sót lỗi |
| Instance segmentation | Polygon (xy list) | Mask răng cưa/lẹm | Vẽ biên chi tiết | Kiểm tra độ khớp của biên |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Không sao chép dữ liệu ra thiết bị cá nhân.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Quản lý dự án hoặc Mentor.

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
