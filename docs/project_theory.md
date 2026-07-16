# Lý thuyết dự án — IT Job Market Intelligence

## Mục lục
1. [Giới thiệu 3 bài toán ML](#1-giới-thiệu-3-bài-toán-ml)
2. [Dataset và Feature Engineering](#2-dataset-và-feature-engineering)
3. [Tiền xử lý dữ liệu](#3-tiền-xử-lý-dữ-liệu)
4. [Pipeline huấn luyện](#4-pipeline-huấn-luyện)
5. [Thuật toán chi tiết](#5-thuật-toán-chi-tiết)
6. [Đánh giá mô hình](#6-đánh-giá-mô-hình)
7. [Real-time Architecture](#7-real-time-architecture)
8. [Thuật ngữ quan trọng](#8-thuật-ngữ-quan-trọng)

---

## 1. Giới thiệu 3 bài toán ML

### 1.1 Bài toán hồi quy (Regression)

**Định nghĩa**: Hồi quy là bài toán dự đoán một giá trị số liên tục (continuous value) dựa trên các đặc trưng đầu vào.

Trong dự án có 2 bài toán hồi quy:

| Bài toán | Đầu vào (Input) | Đầu ra (Output) | Ý nghĩa |
|----------|----------------|-----------------|---------|
| Salary Prediction | 17 features (kỹ năng, ngành, cấp bậc, vị trí) | salary_annual (USD) | Dự đoán mức lương hàng năm |
| Demand Scoring | 4 categoricals (domain, state, seniority, job_type) | demand_score (0-100) | Chấm điểm nhu cầu tuyển dụng |

### 1.2 Bài toán phân cụm (Clustering)

**Định nghĩa**: Phân cụm là bài toán học không giám sát (unsupervised learning) — không có nhãn trước. Mục tiêu: nhóm các điểm dữ liệu có đặc điểm tương đồng vào cùng một cụm.

**Tại sao dùng Clustering?** Để phân khúc thị trường việc làm IT thành các nhóm có đặc điểm riêng (vd: nhóm Senior lương cao, nhóm Junior mới nổi, nhóm Manager, v.v.) — giúp người tìm việc định vị bản thân.

### 1.3 Tại sao chọn các thuật toán này?

| Thuật toán | Loại | Phù hợp với |
|------------|------|-------------|
| RandomForest | Ensemble (bagging) | Dữ liệu tabular, non-linear, ít preprocessing |
| XGBoost | Ensemble (boosting) | Dữ liệu tabular, cần regularization mạnh |
| KMeans | Distance-based | Dữ liệu liên tục, cần interpretability |
| PCA | Dimensionality reduction | Giảm nhiễu, trực quan hóa, tăng tốc KMeans |

**So sánh Bagging vs Boosting:**
- **Bagging** (RF): train nhiều cây song song, mỗi cây trên bootstrap sample, kết quả = average. Giảm variance, chống overfit.
- **Boosting** (XGBoost): train tuần tự, cây sau sửa lỗi cây trước. Giảm bias tốt hơn nhưng dễ overfit nếu không tuning.

---

## 2. Dataset và Feature Engineering

### 2.1 Dữ liệu gốc (Kaggle LinkedIn Job Postings)

3 file CSV, liên kết qua khóa `job_id`:

```
linkedin_job_postings.csv (1.3M rows)
├── job_id: khóa chính
├── job_title: chức danh (text) — feature gốc
├── salary_annual: lương hàng năm (float) — OUTPUT của salary model
├── job_level: cấp bậc từ LinkedIn (text: "Mid-Senior", "Associate") — KHÔNG dùng trực tiếp
├── job_type: Remote / Hybrid / On-site — feature gốc
├── location: địa điểm "City, State" — feature gốc
├── company_size: quy mô công ty — KHÔNG dùng (nhiều null)
├── industry: ngành công ty — KHÔNG dùng (nhiều null)
│
job_skills.csv (1.3M rows)
├── job_id: khóa ngoại
├── job_skills: mảng kỹ năng (text array) — feature gốc quan trọng nhất
│
job_summary.csv (1.3M rows)
├── job_id: khóa ngoại
├── job_summary: mô tả công việc (text dài) — TIỀM NĂNG (chưa dùng)
├── raw_criteria: text chứa seniority + employment type — fallback cho seniority
```

### 2.2 Feature Engineering — Biến raw thành features

Feature Engineering là quá trình biến dữ liệu thô thành vector đặc trưng mà ML model có thể học.

**Từ job_title → it_domain (8 ngành IT):**

Dùng **keyword mapping** — dictionary ánh xạ từ khóa trong title sang domain:

```
"Software Engineer", "Backend Developer", "Full Stack" → Software Engineering
"Data Scientist", "Data Analyst", "ML Engineer" → Data Science
"DevOps Engineer", "Site Reliability" → DevOps
"Security Engineer", "Cybersecurity" → Cybersecurity
"Cloud Architect", "Cloud Engineer" → Cloud
"iOS Developer", "Android Developer" → Mobile
"QA Engineer", "Test Automation" → QA
"IT Manager", "Technical Lead" → IT Management
```

**Tại sao dùng keyword mapping thay vì train một classifier?** Đơn giản, nhanh, đủ chính xác với các từ khóa IT rõ ràng.

**Từ job_title → seniority_level (4 cấp bậc):**

Dùng **regex pattern matching**:
- `/Senior|Lead|Staff|Principal|Sr/i` → Senior (53.6K)
- `/Junior|Entry|Associate|Graduate/i` → Junior (12.4K)
- `/Manager|Director|Head/i` → Manager (13.2K)
- else → Mid (49.1K)

**Tại sao ưu tiên title hơn cột job_level gốc?** Cột job_level của LinkedIn chỉ có 2 giá trị: "Mid-Senior" (1.2M) và "Associate" (144K) — không phân biệt được Junior vs Senior. Regex trên title cho 4 levels chi tiết hơn.

**Từ job_skills → 9 binary skill features:**

Nguyên tắc: mỗi kỹ năng trong mảng gốc được map vào một bucket. Nếu bucket có ≥1 skill → giá trị 1.

```
job_skills = ["Python", "Docker", "AWS", "Machine Learning"]
  → skill_programming = 1 (Python)
  → skill_devops = 1 (Docker)
  → skill_cloud = 1 (AWS)
  → skill_ai_ml = 1 (Machine Learning)
  → các skill khác = 0
```

**Tại sao dùng bucket thay vì one-hot từng skill?** Có hàng ngàn kỹ năng riêng lẻ, one-hot sẽ tạo feature vector quá thưa (sparse) và lớn. Gom vào 9 bucket giảm chiều dữ liệu, tăng generalization.

**Từ job_skills → num_skills, skill_diversity:**
- `num_skills = len(job_skills)` — tổng số kỹ năng
- `skill_diversity = count_nonzero(buckets)` — số bucket khác nhau

**Từ location → state:**
Trích 2 ký tự cuối từ "City, ST" → state code. Tại sao chỉ lấy state, không lấy city? City có quá nhiều giá trị (hàng nghìn), gây sparse features. State chỉ 50 giá trị, dễ xử lý hơn.

**Interaction Features (domain_seniority, state_seniority):**

Interaction features = kết hợp 2 hoặc nhiều features để model học được mối quan hệ giữa chúng.

Ví dụ: "Software Engineering + Senior" cho lương khác với "Software Engineering + Junior" — interaction feature `domain_seniority = "Software Engineering_Senior"` giúp model học trực tiếp điều này, thay vì phải tự suy ra từ 2 features riêng.

### 2.3 Feature Vector cuối cùng (17 chiều)

```
11 numeric:  [num_skills, skill_diversity, skill_programming, skill_cloud, 
               skill_ai_ml, skill_database, skill_devops, skill_framework,
               skill_data_engineering, skill_security, skill_soft_skills]

6 categorical: [seniority_level, job_type, state, it_domain, 
                domain_seniority, state_seniority]
```

Sau OneHotEncoder: 11 numeric + ~110 categorical binary columns = ~121 features.

---

## 3. Tiền xử lý dữ liệu

### 3.1 Gộp dữ liệu (Joining)

Dùng `pd.merge()` (inner join) trên `job_id` để kết hợp 3 CSV. Inner join chỉ giữ lại các dòng có mặt ở cả 3 file → đảm bảo mỗi dòng có đủ job_title, skills, summary.

### 3.2 Lọc IT jobs

Giữ lại các dòng có `it_domain` thuộc 8 domain IT. Tại sao? Dataset gốc có nhiều ngành non-IT (Finance, Healthcare, Legal...) — không liên quan đến mục tiêu phân tích thị trường IT, gây nhiễu cho model.

### 3.3 Xử lý outlier (IQR method)

**Ý tưởng trực quan:**

Hãy tưởng tượng bạn có 100 người xếp hàng theo thứ tự lương từ thấp đến cao. Bạn cắt hàng này làm 4 phần bằng nhau:

```
Người thứ 1 → 25:   nhóm thấp nhất  (Q0 → Q1)
Người thứ 26 → 50:  nhóm trung bình thấp (Q1 → Q2 = median)
Người thứ 51 → 75:  nhóm trung bình cao (Q2 → Q3)
Người thứ 76 → 100: nhóm cao nhất  (Q3 → Q4)
```

**Ý tưởng của IQR:**
- Q1 (25th percentile) là mức lương của người thứ 25 — 25% người có lương thấp hơn mốc này
- Q3 (75th percentile) là mức lương của người thứ 75 — 75% người có lương thấp hơn mốc này
- IQR = Q3 - Q1 = khoảng cách giữa người thứ 75 và người thứ 25 — nó đo độ trải rộng của **50% dân số ở giữa** (bỏ qua 25% thấp nhất và 25% cao nhất)

Ý tưởng là: nếu một mức lương cách xa "khu vực đám đông" (Q1 đến Q3) quá 1.5 lần độ rộng của khu vực đó, thì nó đáng nghi là outlier.

```
Ví dụ cụ thể với dữ liệu lương IT:
Q1 = $80,000   (25% lập trình viên có lương ≤ $80K)
Q3 = $150,000  (75% lập trình viên có lương ≤ $150K)
IQR = $150K - $80K = $70K  (50% dân số ở giữa có lương trong khoảng $70K)

Lower bound = $80K - 1.5 × $70K = -$25K → bị chặn dưới $15K
Upper bound = $150K + 1.5 × $70K = $255K → bị chặn trên $500K

→ Giữ lại: $15,000 ≤ lương ≤ $255,000 (sau chặn: $15K ≤ lương ≤ $255K)
→ Loại bỏ: lương < $15K (thực tập, bán thời gian) và lương > $255K (C-suite, VP)
```

Các mức lương $15K-255K đại diện cho 96% dân số IT — phần còn lại (thực tập sinh lương rất thấp, hay CEO/CTO lương rất cao) không đại diện cho thị trường lao động IT phổ thông.

**Công thức IQR:**
```
Q1 = 25th percentile của salary_annual
Q3 = 75th percentile của salary_annual
IQR = Q3 - Q1
Lower bound = max(Q1 - 1.5×IQR, 15,000)   # chặn dưới $15K
Upper bound = min(Q3 + 1.5×IQR, 500,000)   # chặn trên $500K
Giữ lại: lower ≤ salary_annual ≤ upper
```

**Tại sao chọn IQR thay vì Z-score?**
- IQR không giả định phân phối chuẩn (normal distribution) — lương thường phân phối lệch phải (right-skewed), không đối xứng
- Z-score giả định dữ liệu hình chuông (bell curve) → không phù hợp với lương: có nhiều người lương thấp-trung bình và ít người lương rất cao
- Threshold 1.5 là quy tắc của nhà thống kê học John Tukey — kinh nghiệm cho thấy 1.5×IQR cân bằng tốt giữa giữ lại dữ liệu và loại outlier
- IQR robust hơn Z-score: Z-score dùng mean và std deviation — cả hai đều bị ảnh hưởng bởi outlier (vì outlier kéo mean lệch). IQR dùng median và percentile — không bị ảnh hưởng bởi outlier

**Kết quả:** ~50K rows có salary → ~30K rows sau lọc (loại ~40%). Các mức lương cực thấp (<$15K, internship/part-time) và cực cao (>$500K, C-suite) được loại bỏ vì không đại diện cho thị trường IT chung.

---

## 4. Pipeline huấn luyện — Máy rửa xe tự động cho dữ liệu

Hãy tưởng tượng bạn có một chiếc xe bẩn (dữ liệu thô) và bạn muốn đưa nó vào tiệm rửa xe (pipeline). Chiếc xe đi qua một dây chuyền tự động: xịt nước → chà xà phòng → xịt lại → sấy khô → ra xe sạch. Bạn không cần tự làm từng bước — chỉ cần đưa xe vào đầu dây chuyền và nhận xe sạch ở cuối dây chuyền.

Pipeline ML cũng vậy. Bạn đưa dữ liệu thô vào đầu pipeline, pipeline tự động làm mọi bước (scale, encode, predict) và trả kết quả ở cuối.

### 4.1 ColumnTransformer — Bàn phân loại, chia hàng về đúng ngăn

Cột dữ liệu của chúng ta có **2 loại** — giống như 2 loại hàng hóa khác nhau:

| Loại cột | Ví dụ | Vấn đề | Cách xử lý |
|----------|-------|--------|------------|
| **Số (numeric)** | num_skills=5, skill_programming=1 | Đơn vị khác nhau: num_skills (0→30) khác skill_programming (0 hoặc 1). Nếu không xử lý, model nghĩ num_skills quan trọng hơn chỉ vì nó là số to, không phải vì nó thực sự quan trọng | **StandardScaler**: đưa về thang đo chung |
| **Danh mục (categorical)** | state=CA, seniority=Senior | Máy tính không hiểu chữ "CA" hay "Senior" — chỉ hiểu số | **OneHotEncoder**: biến chữ thành số |

**StandardScaler — ví dụ dễ hiểu:**

Bạn có 2 bài kiểm tra: Toán thang 100 và Văn thang 10. Nếu cộng điểm trung bình cộng, bạn 9đ Văn sẽ chỉ được 9, trong khi bạn 50đ Toán chỉ được 50 — điểm Toán lấn át Văn. Scale đưa cả 2 về thang z-score: "bạn đứng trên trung bình bao nhiêu độ lệch chuẩn?" — công bằng.

```
z = (giá_trị - trung_bình) / độ_lệch_chuẩn

num_skills=30 (cao bất thường) → z=2.5 (cao hơn trung bình 2.5 độ lệch)
skill_programming=1 → z=0.8 (cao hơn trung bình 0.8 độ lệch)
```

→ Cả 2 feature được đưa về cùng scale, model không bị "lừa" bởi đơn vị khác nhau.

**OneHotEncoder — ví dụ dễ hiểu:**

State có 50 giá trị (CA, NY, TX, FL...). Nếu gán số: CA=0, NY=1, TX=2 → model sẽ nghĩ "CA < NY < TX" — một quan hệ "lớn hơn, bé hơn" hoàn toàn không có thật. (California không "bé hơn" New York.)

OneHotEncoder thay vì gán 1 số, tạo 50 cột, mỗi cột là 1 state:

```
State=CA → [1, 0, 0, 0, ..., 0]  (50 cột, cột CA = 1, còn lại = 0)
State=NY → [0, 1, 0, 0, ..., 0]  (cột NY = 1)
```

→ Máy tính hiểu: CA và NY là 2 giá trị khác nhau, không có thứ tự. Giống như bạn đánh dấu √ vào ô tương ứng trên phiếu khảo sát.

### 4.2 Pipeline — Dây chuyền tự động

Pipeline = đóng gói tất cả vào 1 object. Bạn chỉ cần gọi `pipeline.predict(dữ_liệu_mới)` và pipeline tự động:

1. **Phân loại cột**: cột số → vào StandardScaler, cột danh mục → vào OneHotEncoder
2. **Scale + Encode**: tự động transform dữ liệu
3. **Predict**: đưa ma trận đã xử lý vào model (RF / XGBoost)
4. **Trả kết quả**: lương dự đoán / điểm nhu cầu / cluster ID

**Ví dụ: Dự đoán lương cho "Senior, Python, AWS, California"**

```
Đầu vào (raw):
  job_title="Senior Software Engineer", state="CA", skill=["Python","AWS","SQL"]
  
Bước 1 - Feature engineering (tự động từ code xử lý):
  it_domain="Software Engineering", seniority="Senior", num_skills=3,
  skill_programming=1, skill_cloud=1, skill_database=1, các skill khác=0

Bước 2 - ColumnTransformer:
  Số: [3, 3, 1, 1, 0, 1, 0, 0, 0, 0, 0] ← sau StandardScaler
  Danh mục: [Senior, Remote, CA, Software Engineering, ...] ← sau OneHotEncoder

Bước 3 - Model predict:
  Kết quả: $165,230/năm
  
Đầu ra: {"predicted_salary": 165230.00, "currency": "USD", "period": "Annual"}
```

**Lợi ích của Pipeline:**
- **Không cần nhớ**: bạn không cần nhớ phải scale trước, encode sau — pipeline làm hết
- **Chống gian lận dữ liệu**: pipeline chỉ "học" cách scale/encode từ dữ liệu train, không từ dữ liệu test. Nếu bạn tự scale bằng tay, bạn có thể vô tình dùng thông tin từ test set để scale → kết quả đánh giá sẽ ảo, cao hơn thực tế
- **Dễ mang đi**: dump 1 file .joblib duy nhất chứa toàn bộ pipeline, deploy lên server chỉ cần 1 dòng lệnh

### 4.3 Train/Test Split — Chia bài thi ra làm 2

Bạn không thể vừa học bài vừa tự chấm điểm cho mình — sẽ gian lận. ML cũng vậy.

- **Train set (80%)**: model học từ dữ liệu này — giống như bạn học giáo trình
- **Test set (20%)**: model KHÔNG BAO GIỜ nhìn thấy — giống như đề thi thật, dùng để chấm điểm thực sự

**Cụ thể với dự án:**
- 30,039 dòng có lương → chia làm 2:
  - Train: ~24,031 dòng — model học từ đây
  - Test: ~6,008 dòng — model chưa thấy bao giờ, dùng để tính R², MAE

**random_state=42 — chìa khóa tái tạo kết quả:**
Khi chia ngẫu nhiên, máy tính dùng "số random". random_state=42 là hạt giống (seed) — nói với máy: "hãy chia theo cách này, và lần sau cũng chia y hệt". Nếu không có seed, mỗi lần chạy bạn có một bộ train/test khác nhau → không thể so sánh kết quả giữa các lần chạy.

**Tại sao 80/20?** Giống quy tắc Pareto: 80% là đủ để model học tốt, 20% là đủ để đánh giá tin cậy. Với 30K mẫu, 80% = 24K là dư để RF hội tụ (không cần thêm dữ liệu cũng không cải thiện đáng kể).

### 4.4 GridSearchCV — Nếm thử 12 công thức phở để chọn ngon nhất

Mỗi model có nhiều **tham số (hyperparameters)** — bạn phải đặt trước khi nấu. Giống như nấu phở: bao nhiêu bánh phở? lửa to hay nhỏ? nước dùng đậm hay nhạt?

Thay vì đoán mò, GridSearchCV tự động thử tất cả tổ hợp:

```
Các tham số cần thử:
- n_estimators (số cây): 200 hay 400 cây?
- max_depth (độ sâu cây): 15, 25, hay không giới hạn?
- min_samples_leaf (mẫu tối thiểu ở lá): 1 hay 3?

→ 2 × 3 × 2 = 12 tổ hợp (12 công thức phở)

Với mỗi tổ hợp, GridSearchCV:
1. Chia train set ra làm 3 phần (3-fold cross-validation)
2. Lần lượt lấy 2 phần học, 1 phần kiểm tra — xoay vòng 3 lần
3. Tính điểm R² trung bình của 3 lần
4. Chọn tổ hợp có điểm cao nhất
```

**Ví dụ kết quả thực tế trong dự án:**
- 400 cây tốt hơn 200 cây (R² 0.531 vs 0.528)
- max_depth=25 tốt hơn 15, không thua kém None (không giới hạn) — nhưng nhanh hơn
- min_samples_leaf=3 tốt hơn 1 (chống overfit)
- Kết luận: RF tuned (max_depth=25, min_samples_leaf=3, n_estimators=400)

**Cross-validation (xoay vòng) là gì?** Thay vì chỉ chia 1 lần (may rủi), cross-validation chia nhiều lần, mỗi lần lấy phần khác nhau làm validation. Giống như bạn học thuộc lòng 3 đề và thi thử cả 3 đề — nếu chỉ thi 1 đề có thể hôm đó bạn không may gặp đúng tủ.

**Tại sao chỉ 3 lần (cv=3)?** 12 tổ hợp × 3 lần = 36 lần train. Mỗi lần train RF trên 24K dòng tốn ~30 giây → 36 × 30 = 18 phút. Nếu làm 5 lần (60 lần train) → 30 phút. 3 lần là dung hòa giữa độ tin cậy và thời gian chờ.

---

## 5. Thuật toán chi tiết

### 5.1 RandomForestRegressor — 400 chuyên gia bỏ phiếu, không ai bị lái

**Decision Tree — cây quyết định là gì? (nhắc lại)**

Trước khi hiểu RandomForest, phải hiểu Decision Tree. Nó giống như trò chơi "20 câu hỏi":

Bạn muốn đoán lương của một lập trình viên. Bạn hỏi từng câu:
- Câu 1: "Có biết Python không?" → Có → đi nhánh trái, Không → nhánh phải
- Câu 2: "Cấp bậc Senior?" → Có → nhánh trái, Không → nhánh phải
- Câu 3: "Ở California?" → Có → nhánh trái, Không → nhánh phải
- ... tiếp tục cho đến khi tới 1 cái "lá" — một nhóm nhỏ mà tất cả có mức lương gần nhau → lấy trung bình làm dự đoán

Mỗi câu hỏi là một **node**, mỗi câu trả lời là một **nhánh**, điểm cuối là **lá (leaf)**. Decision Tree tự động học các câu hỏi nào tốt nhất (câu nào giúp phân chia dữ liệu hiệu quả nhất).

**Vấn đề của 1 cây duy nhất:** Nó dễ bị overfit — học quá kỹ dữ liệu train. Giống như bạn học thuộc lòng đáp án mà không hiểu bản chất, ra đề thi mới là sai.

**RandomForest — giải pháp: nhiều cây thay vì một cây**

Thay vì dùng 1 chuyên gia (1 cây), RF dùng 400 chuyên gia (400 cây). Mỗi chuyên gia:
- Chỉ được xem một phần dữ liệu (bootstrap: lấy 24K mẫu từ 24K mẫu, nhưng có hoàn lại — có mẫu được lấy 2 lần, có mẫu không được lấy)
- Chỉ được dùng một phần câu hỏi (mỗi node chỉ được xét subset ngẫu nhiên các features)
- Học với dữ liệu khác nhau → mỗi cây có "góc nhìn" khác nhau

Khi cần dự đoán:
- 400 chuyên gia bỏ phiếu (mỗi cây cho 1 con số)
- Kết quả cuối = trung bình 400 phiếu

**Ví dụ:** Dự đoán lương cho "Senior, Python, AWS, California":
- Cây 1 nhìn thấy nhiều Senior ở California → dự đoán $170K
- Cây 2 nhìn thấy ít Python+Cloud → dự đoán $155K
- Cây 3 nhìn thấy nhiều Senior Cloud → dự đoán $175K
- ...
- Kết quả cuối: trung bình ≈ $165K

**Tại sao RF tốt hơn 1 cây?** Sai số của model đến từ 2 nguồn: **bias** (sai do giả định đơn giản hóa) và **variance** (sai do nhạy cảm với dữ liệu). Decision Tree có variance cao (thay đổi 1 chút dữ liệu → cây hoàn toàn khác). RF lấy trung bình nhiều cây → variance giảm mạnh, không tăng bias.

**Tham số sau tuning:**
- `n_estimators=400`: 400 chuyên gia. Càng nhiều càng ổn định, nhưng sau 400 thì lợi ích thêm không đáng kể (diminishing returns)
- `max_depth=25`: mỗi chuyên gia chỉ được hỏi tối đa 25 câu. Giới hạn này chống việc học thuộc lòng (overfit). Nếu không giới hạn, cây có thể hỏi 100+ câu → quá chi tiết → nhớ từng mẫu train
- `min_samples_leaf=3`: mỗi lá phải có ít nhất 3 người. Nếu lá chỉ có 1 người → quá đặc thù → overfit

### 5.2 XGBoost

**Gradient Boosting:**

Khác với bagging (train song song), boosting train tuần tự:
1. Cây đầu tiên dự đoán y
2. Tính residual (lỗi): r = y - f(x)
3. Cây tiếp theo học để dự đoán residual
4. Lặp đến khi đủ số cây hoặc lỗi đủ nhỏ
5. Kết quả = tổng có trọng số của tất cả cây

**Ký hiệu toán học:**
```
f(x) = Σ_{m=1}^{M} γ_m · h_m(x)
```
Trong đó: h_m là cây thứ m, γ_m là learning rate, M là số cây

**Cơ chế regularization của XGBoost:**
- **L1 (Lasso) regularization**: phạt số lượng leaf nodes → cây đơn giản hơn
- **L2 (Ridge) regularization**: phạt trọng số leaf → tránh leaves quá lớn
- **Shrinkage (learning_rate)**: nhân kết quả mỗi cây với η < 1 → chậm hội tụ nhưng generalize tốt hơn
- **Column subsampling**: mỗi cây chỉ xem subset columns → giống RF, giảm correlation giữa các cây
- **Row subsampling**: mỗi cây chỉ xem subset rows → stochastic gradient boosting

**Tham số:**
- `n_estimators=500`: số cây. Cao hơn RF vì learning_rate thấp cần nhiều cây hơn
- `max_depth=8`: XGBoost thường dùng depth nhỏ hơn RF (6-10) vì boosting dễ overfit
- `learning_rate=0.05`: shrinkage — nhỏ (0.01-0.1) để generalization tốt. Nếu cao (0.3) → nhanh hội tụ nhưng dễ overfit
- `subsample=0.8`: lấy 80% mẫu mỗi cây. 1.0 → không subsample, dễ overfit
- `colsample_bytree=0.8`: lấy 80% features mỗi cây. Cơ chế tương tự RF's max_features

**So sánh RF vs XGBoost cho dataset này:**
- RF tuned thắng XGBoost (0.5314 vs 0.5284) — lý do: data chỉ 24K train samples, RF với bagging ít overfit hơn boosting
- Trên dataset lớn hơn (>100K), XGBoost thường outperform RF

### 5.3 PCA (Principal Component Analysis) — Chụp ảnh 3D từ 120 góc nhìn, chỉ giữ 5 góc quan trọng nhất

**Vấn đề: dữ liệu nhiều chiều**

Sau OneHotEncoder, mỗi dòng dữ liệu là một điểm trong không gian **120 chiều** (120 features). Bạn có thể tưởng tượng không gian 2D (trục X, Y), 3D (X, Y, Z), nhưng không ai có thể tưởng tượng không gian 120 chiều.

Vấn đề: trong không gian nhiều chiều, **mọi thứ đều xa nhau** — khoảng cách Euclidean giữa 2 điểm bất kỳ gần như bằng nhau → KMeans (dùng khoảng cách) không phân biệt được cụm nào với cụm nào. Đây gọi là "curse of dimensionality" (lời nguyền chiều cao).

**Ý tưởng trực quan của PCA:**

Hãy tưởng tượng bạn đứng giữa một căn phòng và có 120 người bạn rải rác xung quanh. Bạn muốn mô tả vị trí của họ, nhưng 120 con số (tọa độ X₁, X₂, ..., X₁₂₀) là quá nhiều.

PCA nói: "Nhiều chiều trong số đó thực ra rất giống nhau, hoặc không quan trọng. Hãy tìm ra **hướng nào có sự khác biệt nhất** giữa mọi người."

Cụ thể:
1. PCA tìm ra **hướng thứ nhất** (gọi là PC1) — đây là hướng mà dữ liệu trải rộng nhất, khác biệt nhất. Nếu bạn chỉ được nhìn dữ liệu theo 1 hướng, hãy nhìn theo hướng này.
2. Sau đó PCA tìm **hướng thứ hai** (PC2) — vuông góc với hướng thứ nhất, là hướng khác biệt thứ hai.
3. Tiếp tục đến PC3, PC4, PC5...
4. Mỗi hướng có một "điểm số quan trọng" gọi là **eigenvalue** — eigenvalue càng lớn, hướng đó càng chứa nhiều thông tin

**Ví dụ cụ thể với 2 chiều (dễ hình dung):**

Giả sử bạn có 100 lập trình viên, mỗi người được đo 2 feature: **số năm kinh nghiệm (X)** và **lương (Y)**. 2 chiều này có tương quan (nhiều năm → lương cao), nên dữ liệu tạo thành một đường chéo dài.

- PC1 = hướng dọc theo đường chéo (giải thích ~95% variance) — đây là hướng quan trọng
- PC2 = hướng vuông góc với đường chéo (chỉ ~5% variance) — có thể bỏ qua

→ Bạn giảm từ 2 chiều xuống 1 chiều (PC1) mà vẫn giữ được 95% thông tin.

**Trong dự án: 120 chiều → 5 chiều**

120 features sau OneHotEncoder có rất nhiều feature tương quan nhau (vd: state_CA và state_NY hiếm khi cùng xảy ra). PCA tìm ra 5 hướng quan trọng nhất, chứa ~79% tổng variance.

Kết quả: thay vì 120 con số, mỗi dòng dữ liệu chỉ còn 5 con số — đủ để KMeans hoạt động tốt.

**Tại sao chọn 5?**
- Vẽ biểu đồ explained variance vs số components → chọn điểm "khuỷu tay" (elbow): sau 5 thì thêm component không giúp ích nhiều
- 5 components giữ ~79% thông tin — mất 21% nhưng giảm được 96% số chiều (120→5)

### 5.4 KMeans

**Nguyên lý:**
1. Chọn K centroids ngẫu nhiên
2. Gán mỗi điểm vào centroid gần nhất
3. Cập nhật centroid = trung bình các điểm trong cụm
4. Lặp (2)-(3) đến khi hội tụ (centroid không đổi)

**Công thức:**
```
minimize: Σ_{i=1}^{n} Σ_{k=1}^{K} ||x_i - μ_k||²
```
Trong đó: μ_k là centroid thứ k, ||·||² là Euclidean distance squared

**Tại sao K=5?**
- Dùng Elbow method + Silhouette analysis
- Elbow: vẽ inertia (within-cluster sum of squares) vs K, chọn điểm elbow (góc giảm chậm lại)
- Silhouette: đo độ tách biệt giữa các cụm, range [-1,1]. K=5 cho Silhouette=0.524 (>0.5 là tốt)
- 5 cụm cho interpretation rõ ràng: Senior Data, Junior Systems, Management, v.v.

**Thuật toán KMeans++ (n_init=10):**
- Chạy 10 lần với centroid initialization khác nhau
- Mỗi lần dùng KMeans++: centroid đầu tiên random, các centroid sau chọn xa centroid đã chọn (tránh hội tụ local minimum)
- Chọn lần có inertia thấp nhất

---

## 6. Đánh giá mô hình

### 6.1 R² (Coefficient of Determination)

**Công thức:**
```
R² = 1 - (Σ(y_i - ŷ_i)²) / (Σ(y_i - ȳ)²)
```
- y_i: giá trị thực
- ŷ_i: giá trị dự đoán
- ȳ: trung bình của y

**Ý nghĩa:**
- R² = 1: dự đoán hoàn hảo
- R² = 0: model không tốt hơn dự đoán bằng trung bình
- R² < 0: model tệ hơn dự đoán bằng trung bình
- R² = 0.531: model giải thích được 53.1% phương sai lương

**Tại sao R² không cao hơn?** Giới hạn đến từ feature:
- Không có: years of experience, education level, company_size, industry cụ thể
- Binary skills: 1/0 không phân biệt "biết sơ" vs "thành thạo"
- Không có yếu tố thị trường: chi phí sống, thời điểm tuyển dụng

### 6.2 MAE (Mean Absolute Error)

**Công thức:**
```
MAE = (1/n) * Σ|y_i - ŷ_i|
```

**Ý nghĩa:** Sai số trung bình là $23,088/năm. Nghĩa là dự đoán của model thường lệch ~23K so với lương thực tế.

**So sánh MAE vs RMSE:**
- MAE: dễ diễn giải (đơn vị giống y)
- RMSE: phạt outlier nặng hơn (bình phương lỗi)
- Dùng MAE vì dễ giải thích cho người dùng

### 6.3 Silhouette Score — Đo xem các cụm có "tách biệt" tốt không

**Mục tiêu của metric này là gì?**

Sau khi KMeans phân 30,000 job thành 5 cụm, ta cần một con số để đánh giá: **liệu 5 cụm này có thực sự tách biệt nhau không?** Hay chúng chỉ là 5 đám mây chồng lên nhau?

Silhouette trả lời câu hỏi đó. Nó đo cho từng điểm dữ liệu: "Điểm này có thuộc đúng cụm không?"

**Cách đo — ví dụ với 1 điểm:**

Lấy một job bất kỳ (giả sử: "Senior Data Engineer ở California"). Silhouette tính 2 khoảng cách:

- **a(i)** = khoảng cách trung bình từ job này đến **các job khác trong cùng cụm**. Hỏi: "Nó có gần với đồng đội không?"
- **b(i)** = khoảng cách trung bình từ job này đến **các job ở cụm gần nhất**. Hỏi: "Nó có xa với cụm lân cận không?"

**Công thức:**
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

**Ý nghĩa từng trường hợp:**

- **s(i) ≈ 1**: b(i) >> a(i) — job này rất gần cụm của nó và rất xa cụm khác → phân cụm hoàn hảo
- **s(i) ≈ 0**: b(i) ≈ a(i) — job này nằm đúng ranh giới giữa 2 cụm, có thể thuộc cụm nào cũng được → không rõ ràng
- **s(i) < 0**: b(i) < a(i) — job này gần cụm khác hơn cụm của nó → bị phân sai cụm

**Giá trị trung bình của toàn bộ dữ liệu (mean silhouette):**
- < 0.25: cấu trúc cụm yếu, không có ý nghĩa
- 0.25 - 0.50: có cấu trúc nhưng chồng lấn nhiều
- **0.50 - 0.70: cấu trúc tốt, các cụm tách biệt rõ** ← dự án đạt 0.524
- > 0.70: rất tốt (hiếm trên dữ liệu thực tế)

**Kết luận:** Silhouette = 0.524 chứng tỏ 5 cụm KMeans tách biệt tốt, phân khúc thị trường IT có ý nghĩa thực tế.

### 6.4 Confusion Matrix vs Regression Metrics

**Lưu ý:** Bài toán hồi quy (salary, demand) không dùng confusion matrix, precision, recall — những metrics này dành cho classification. Đây là lỗi thường gặp khi trình bày.

---

## 7. Real-time Architecture — Lấy dữ liệu thị trường "bây giờ" như thế nào?

### 7.0 Thu thập dữ liệu realtime — API chạy mỗi khi user click

**Cơ chế:**

Không có worker chạy nền định kỳ (không phải "5h/lần"). Thay vào đó, dữ liệu realtime được lấy **theo yêu cầu** (on-demand):

1. User mở dashboard và click nút **"Lấy xu hướng thị trường Real-time"** hoặc chuyển sang tab Real-time
2. Backend gọi API freehire.dev để lấy job đang tuyển dụng ở Mỹ
3. Kết quả được cache trong 1 giờ — nếu user click lại trong vòng 1h, lấy từ cache (không gọi API lại)
4. Nếu API fail (hết rate, mất mạng), dùng cache cũ hoặc snapshot

**Tần suất thực tế:**
- Mỗi user click → 1 lần gọi API (nếu cache hết hạn)
- Cache 1h → tối đa 24 lần/ngày nếu có người dùng liên tục
- Không có scheduler riêng — tiết kiệm tài nguyên, không gọi API vô ích khi không ai xem dashboard

### 7.1 Ý tưởng — Tại sao cần 2 nguồn dữ liệu?

Dự án có **2 loại dữ liệu**:

| Loại | Nguồn | Dùng để làm gì? |
|------|-------|-----------------|
| **Lịch sử (Historical)** | Kaggle dataset 2024 — 1.3M bài đăng cũ | **Train ML model** (cần nhiều dữ liệu để học), vẽ trend chart 30 tháng |
| **Thời gian thực (Realtime)** | freehire.dev hôm nay — job đang tuyển | **Phân tích hiện tại** (skill nào đang hot, công ty nào đang tuyển, mức lương hiện nay) |

**Tại sao không chỉ dùng 1 nguồn?**
- Chỉ Kaggle: dữ liệu 2024 → không biết thị trường hôm nay ra sao
- Chỉ API: không có lịch sử 30 tháng → không train được ML, không vẽ được biểu đồ xu hướng

→ Kết hợp: ML học từ quá khứ, API cho biết hiện tại, dashboard hiển thị cả hai.

### 7.2 Provider Chain — Kế hoạch A, B, C, D

Khi user click "Lấy xu hướng", backend không chỉ gọi 1 API rồi bỏ cuộc nếu fail. Nó có 4 lớp dự phòng:

```
Bước 1: freehire.dev (miễn phí, 2.9 triệu jobs Mỹ)
  → Nếu API chết (timeout, hết lượt)
Bước 2: RemoteOK (miễn phí, job remote toàn cầu)
  → Nếu cũng chết
Bước 3: Cache trên ổ cứng (dữ liệu của lần gọi thành công gần nhất, còn dưới 1 giờ)
  → Nếu cache cũ quá hoặc không có
Bước 4: Snapshot (15 job mẫu có sẵn trong code — luôn chạy được)
```

**Tại sao phải phức tạp vậy?** API bên ngoài có thể bị block, rate-limit, hay đơn giản là server chết. Dự phòng đảm bảo dashboard **luôn có dữ liệu để hiển thị**, dù là dữ liệu cũ hay mẫu.

**freehire.dev làm việc thế nào?**

Là API free, không cần key. Gửi request với tham số:
- category: backend, frontend, devops, data, mobile, security...
- location: US

API trả về danh sách job với: title, company, location, description, tags (list kỹ năng), date.

**Vấn đề:** API trả về mọi ngành, không chỉ IT. Có cả "security officer" (bảo vệ tòa nhà), "data entry" (nhập liệu), "finance analyst" — những job này không liên quan đến IT.

→ **IT Filter — bộ lọc 3 lớp:**

```
Với mỗi job:

Lớp 1 - Loại trừ: title có chứa từ khóa non-IT?
  ["security officer", "data entry", "finance", "nurse", "teacher", ...] (50+ từ)
  → CÓ → loại bỏ ngay

Lớp 2 - Bao gồm: title có chứa từ khóa IT?
  ["engineer", "developer", "scientist", "architect", "analyst", "admin", ...] (40+ từ)
  → CÓ → chấp nhận

Lớp 3 - Dự phòng: nếu title không rõ ràng, đếm tech tags
  tags chứa ["Python", "AWS", "Docker", "React", "SQL", ...] ?
  Nếu ≥ 2 tech tags → chấp nhận (vd: "Technical Lead" không có title IT nhưng có Python+AWS)
  Nếu không → loại bỏ
```

→ Kết quả: từ 30 job API trả về, chỉ giữ 15-18 job thực sự IT (~95% purity).

### 7.3 Cache — Tại sao không gọi API mỗi lần?

Mỗi lần fetch thành công, dữ liệu được ghi vào file `data/realtime_cache.json` kèm timestamp.

```
Lần 1: User A click → gọi freehire.dev (2 giây) → lưu cache
Lần 2: User A click lại sau 10 phút → đọc từ cache (<1 mili giây) → không gọi API
Lần 3: User B click sau 2 tiếng → cache hết hạn → gọi API mới → lưu cache mới
```

**Lợi ích:**
1. **Tôn trọng rate-limit**: freehire.dev giới hạn số request/phút. Cache giảm tải
2. **Nhanh hơn**: đọc file local mất micro giây, gọi API mất 2-5 giây
3. **Offline**: nếu mất internet, cache vẫn còn → dashboard vẫn hoạt động

### 7.4 Trend Forecast — Dự báo 2 tháng tới

**Ý tưởng:** Có số liệu 30 tháng quá khứ, muốn vẽ đường cong để đoán 2 tháng tiếp theo.

**Cách làm — Polynomial Regression degree 2:**

Giả sử dữ liệu 30 tháng cho thấy số lượng job tăng dần, nhưng tốc độ tăng chậm lại gần đây (thị trường chững). Ta cần một đường cong:

- **Đường thẳng (degree 1)**: quá đơn giản, không bắt được chững lại → dự báo sẽ tăng đều, sai
- **Đường cong bậc 2 (degree 2)**: hình parabol — cong 1 lần → phù hợp: tăng nhanh → chậm lại → phẳng
- **Bậc 3+**: cong nhiều lần → có thể uốn éo bất kỳ → dễ overfit, dự báo phi thực tế

→ Chọn bậc 2 là phù hợp nhất.

**Dữ liệu đầu vào cho forecast:**
- `backup_trends.csv`: 30 dòng, mỗi dòng = 1 tháng, cột job_count
- Baseline: 128,307 jobs (tổng số job trong Kaggle)
- Mỗi tháng thêm ~0.29% growth + nhiễu ngẫu nhiên ±2.5% (giả lập biến động thị trường thực tế)
- Khi có API count hiện tại: scale con số API để khớp với magnitude historical data

→ Kết quả: biểu đồ đường 2 màu (xanh = lịch sử, vàng nét đứt = dự báo)

---

## 8. Thuật ngữ quan trọng

| Thuật ngữ | Giải thích | Ví dụ trong dự án |
|-----------|-----------|-------------------|
| **Feature** | Đặc trưng đầu vào cho ML model | `num_skills`, `seniority_level` |
| **Label / Target** | Biến mục tiêu cần dự đoán | `salary_annual` (cho Salary model) |
| **Pipeline** | Chuỗi các bước preprocessing + model | ColumnTransformer → RF |
| **Hyperparameter** | Tham số của thuật toán, đặt trước khi train | `n_estimators=400`, `max_depth=25` |
| **Parameter** | Tham số model học được từ dữ liệu | Feature weights trong linear regression |
| **Overfitting** | Model học quá kỹ train set → kém trên test | RF depth=100 → nhớ từng mẫu |
| **Underfitting** | Model học chưa đủ → kém cả train + test | Linear regression trên dữ liệu non-linear |
| **Bias** | Sai số do giả định đơn giản hóa | Giả định lương là hàm tuyến tính |
| **Variance** | Sai số do nhạy cảm với dữ liệu train | Cây sâu thay đổi nhiều khi thay data |
| **Bias-Variance Tradeoff** | Cân bằng giữa 2 loại sai số | RF giảm variance bằng ensemble |
| **Cross-validation** | Chia train thành k folds, đánh giá chéo | GridSearchCV(cv=3) |
| **Encoding** | Biến categorical thành số | OneHotEncoder biến "CA"→[0,0,1,0...] |
| **Scaling** | Chuẩn hóa feature về cùng scale | StandardScaler: z=(x-μ)/σ |
| **Ensemble** | Kết hợp nhiều model yếu → model mạnh | RF: 400 cây, XGBoost: 500 cây |
| **Bagging** | Ensemble parallel, giảm variance | RF (Bootstrap + Aggregating) |
| **Boosting** | Ensemble sequential, giảm bias | XGBoost (học từ residual) |
| **Dimensionality Reduction** | Giảm số chiều dữ liệu | PCA: 120→5 components |
| **Silhouette** | Đo độ tách biệt cụm | Cluster: 0.524 (>0.5 tốt) |
| **R²** | Tỉ lệ phương sai được giải thích | Salary: 0.531 (53.1%) |
| **MAE** | Sai số tuyệt đối trung bình | Salary: $23,088 |
| **IQR** | Khoảng tứ phân vị, dùng detect outlier | IQR method: Q1-1.5×IQR đến Q3+1.5×IQR |
| **Interaction Feature** | Feature kết hợp 2+ features khác | domain_seniority = domain + "_" + seniority |
| **One-Hot Encoding** | Biến category → vector binary | seniority=Senior → [0,0,1,0] |
| **StandardScaler** | Chuẩn hóa z-score | z = (x - mean) / std |
