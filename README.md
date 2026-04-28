# 🎬 MovieRec — Personalized Movie Recommendation System

Hệ thống gợi ý phim cá nhân hóa sử dụng các mô hình Deep Learning hiện đại, được xây dựng với **FastAPI** (backend) và **React + Vite** (frontend).

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────┐
│                  React + Vite (Port 5173)            │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Sidebar  │  │  HomePage    │  │ MovieInfoPage  │ │
│  │ New User │  │ New / Exist  │  │  Detail + Sim  │ │
│  │ Existing │  │ User flows   │  │  ilar movies   │ │
│  └──────────┘  └──────────────┘  └────────────────┘ │
└───────────────────────┬─────────────────────────────┘
                        │ REST API (fetch)
┌───────────────────────▼─────────────────────────────┐
│               FastAPI Backend (Port 8000)            │
│  /movies/search  /movies/trending  /movies/{id}      │
│  /movies/{id}/similar                                │
│  /recommend/new-user   /recommend/existing-user      │
│  /users/sample                                       │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│                    ML Models (src/)                  │
│  BERT4Rec (new user)  │  NeuMF + Hybrid (exist user) │
│  Content Similarity   │  TMDB API (metadata/poster)  │
└─────────────────────────────────────────────────────┘
```

---

## Các mô hình ML

| Mô hình | Vai trò | Khi nào dùng |
|---|---|---|
| **BERT4Rec** | Sequence-based collaborative filtering | New user (cold-start) |
| **NeuMF** | Neural Matrix Factorization | Existing user |
| **Hybrid** | α × BERT4Rec + (1-α) × NeuMF | Existing user có click history |
| **Content Similarity** | Cosine similarity trên feature vector | Phim tương tự (Similar Movies) |

---

## Cấu trúc dự án

```
Movie_recomendation/
├── backend/
│   ├── main.py          # FastAPI server — 8 endpoints
│   └── .env             # Config (TMDB key, TOP_K, ...)
├── frontend/
│   └── src/
│       ├── api/client.js          # Fetch wrapper
│       ├── context/AppContext.jsx # Global state
│       ├── components/
│       │   ├── MovieCard.jsx      # Card phim
│       │   ├── Carousel.jsx       # Carousel cuộn ngang
│       │   └── Sidebar.jsx        # Điều hướng mode
│       └── pages/
│           ├── HomePage.jsx       # New User / Existing User
│           └── MovieInfoPage.jsx  # Chi tiết + Similar Movies
├── src/
│   ├── recommenders/    # BERT4Rec, NeuMF, Hybrid
│   └── services/        # TMDB, MovieCatalog, ContentSimilarity, IdMapper
├── model/               # File model đã train (.pt, .keras, .pkl, .csv, ...)
├── train/               # Script training
└── scripts/             # Script tiện ích (build topk_similar, ...)
```

---

## Cài đặt & Chạy

### Yêu cầu

- Python 3.10+ với virtual environment (`.venv`)
- Node.js 18+

### 1. Cài dependencies

```bash
# Backend (trong .venv)
.venv\Scripts\pip install fastapi uvicorn python-dotenv

# Frontend
cd frontend
npm install
```

### 2. Cấu hình backend

Tạo / kiểm tra file `backend/.env`:

```env
TMDB_API_KEY=your_tmdb_api_key_here
TOP_K=10
MIN_CLICKS_FOR_COLD_START=3
TMDB_TIMEOUT_SECONDS=10
```

> Lấy TMDB API key miễn phí tại: https://www.themoviedb.org/settings/api

### 3. Chạy ứng dụng

Mở **2 terminal** song song:

**Terminal 1 — Backend:**
```bash
cd e:\Movie_recomendation
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd e:\Movie_recomendation\frontend
npm run dev
```

### 4. Truy cập

| URL | Mô tả |
|---|---|
| http://127.0.0.1:5173 | Giao diện React |
| http://127.0.0.1:8000/docs | Swagger API docs |

---

## Luồng hoạt động

### 🆕 New User (cold-start)
1. Browse / search phim trong catalog
2. Click **View** trên ít nhất **3 phim** để tạo sequence
3. BERT4Rec dự đoán top-10 phim phù hợp với taste của bạn

### 👤 Existing User
1. Chọn User ID (có trong MovieLens 1M dataset)
2. Hệ thống hybrid tổng hợp: `α × BERT4Rec + (1-α) × NeuMF`
3. Giá trị α tăng dần theo số phim click trong session:
   - 0 click → α = 0.0 (chỉ NeuMF)
   - 1-2 click → α = 0.3
   - 3-4 click → α = 0.5
   - 5+ click → α = 0.7

### 🎬 Movie Detail
- Xem metadata đầy đủ (poster, thể loại, điểm, nội dung)
- Top-10 phim tương tự dựa trên Content Similarity

---

## Lưu ý

- Lần khởi động đầu tiên backend cần **~10-15 giây** để load các model ML vào RAM
- Trending movies sẽ tự động retry sau 8 giây nếu BERT4Rec chưa load xong
- Poster phim cần TMDB API key hợp lệ trong `backend/.env`
