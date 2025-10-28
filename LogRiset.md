## Catatan Riset: Evolusi Desain Agen Booking Bioskop Tiketa

### Latar Belakang & Tujuan Awal
Proyek ini dimulai dengan tujuan membangun agen percakapan yang *robust* untuk menangani pemesanan tiket bioskop. Arsitektur awal yang dipilih adalah **State Machine** (via LangGraph) yang dikombinasikan dengan **Heuristic Matching**.

Pendekatan ini dipilih untuk memberikan "pagar pembatas" (guardrails) yang kaku pada LLM, memastikan alur transaksi (Film -> Jadwal -> Kursi -> Konfirmasi) diikuti dengan benar dan mencegah halusinasi.

### Metode Awal: Heuristic Matching (State Machine Kaku)

Arsitektur awal sangat bergantung pada logika Python yang eksplisit untuk memandu LLM:

1.  **`TicketAgentState` Kaku:** *State* melacak tidak hanya data (`movie_id`), tetapi juga alur percakapan (`current_question: "ask_movie"`).
2.  **Klasifikasi Generik:** Satu node `classify_intent` (dengan tool `extract_intent_and_entities`) digunakan untuk menebak niat dan entitas user secara umum.
3.  **Logika Heuristik Berat:** Fungsi-fungsi inti seperti `_match_movie_from_text` dan `_match_showtime_from_text` dibuat untuk mem-parsing input mentah user ("yang ketiga", "jam 7 malam", "aot") dan mencocokkannya dengan data yang ada di *state* (`candidate_movies`, `available_showtimes`) menggunakan `regex` dan logika `if-else`.

### Permasalahan yang Ditemukan ("Heuristic Hell")

Setelah implementasi dan pengujian, pendekatan Heuristic Matching terbukti sangat rapuh dan tidak skalabel. Masalah yang muncul bersifat fundamental:

1.  **Beban Kognitif yang Salah:** Arsitektur ini memaksa *developer* (kita) untuk mengantisipasi *setiap* variasi linguistik pengguna. Kita pada dasarnya membangun NLU (Natural Language Understanding) engine yang buruk dari nol, sementara kita memiliki LLM yang sangat mampu diabaikan.
2.  **Kode Rapuh & Kompleks:** Setiap *edge case* baru ("3" vs "jam 3", "anime" vs "animation", "kimi no nawa" vs "your name") membutuhkan penambalan `regex` atau `if-else` baru. Ini mengarah ke kode yang panjang, sulit dipelihara, dan penuh bug tersembunyi.
3.  **Kebocoran State (State Leaks):** Ketika heuristik gagal (misal, `_match_showtime_from_text` mengembalikan `None`), *state* tidak diperbarui. `main_router` kemudian bingung dan sering kali salah merutekan alur (misal, "bocor" ke `browsing_agent` di tengah alur booking), yang menyebabkan pengalaman pengguna yang rusak (amnesia, looping, halusinasi).
4.  **Kegagalan Fleksibilitas:** Pendekatan ini sangat buruk dalam menangani perubahan pikiran atau kueri "satu tembakan" (*one-shot*) di mana pengguna memberikan semua informasi sekaligus.

Singkatnya, kita berakhir di **"neraka heuristik" (heuristic hell)**, di mana kita lebih banyak menghabiskan waktu untuk menambal logika Python daripada memanfaatkan kekuatan kognitif LLM.

### Perubahan Pendekatan: Contextual Selector Pattern

Untuk mengatasi masalah ini, kami mempensiunkan pendekatan Heuristic Matching dan beralih ke **Contextual Selector Pattern**.

**Konsep Inti:**
Daripada menggunakan logika Python untuk *menebak* maksud user, kita **meminta LLM untuk memilih** dari daftar opsi yang valid.

**Implementasi Baru:**
1.  **Hapus Heuristik:** Fungsi `_match_movie_from_text` dan `_match_showtime_from_text` (dan regex kompleksnya) **dihapus seluruhnya**.
2.  **Node Klasifikasi Cerdas:** `node_classify_intent` menjadi jauh lebih cerdas dan sadar konteks.
    * Jika `current_question == "ask_movie"`, node ini **secara dinamis membangun prompt baru** yang berisi daftar `candidate_movies`.
    * LLM kemudian diinstruksikan: "Ini input user: 'yang ketiga'. Ini daftarnya: [1. Akira (ID: 1), 2. Gundam (ID: 6), 3. Your Name (ID: 2)]. Kembalikan HANYA ID yang benar."
    * LLM, dengan kemampuan kognitifnya, akan dengan mudah mencocokkan "yang ketiga" atau "your name" ke `ID: 2`.
3.  **Peran Dibalik:**
    * **Sebelumnya:** Python bekerja keras (regex), LLM menebak secara generik.
    * **Sekarang:** LLM bekerja keras (pencocokan kontekstual), Python hanya memvalidasi output (misal, `if result.isdigit()`).

### Keuntungan Pendekatan Baru

1.  **Memanfaatkan Kognisi LLM:** Kita menggunakan LLM untuk tugas yang paling cocok: pemahaman bahasa alami yang bernuansa.
2.  **Kode Lebih Bersih & Sederhana:** Menghapus ratusan baris kode heuristik yang rapuh.
3.  **Anti-Halusinasi:** LLM tetap terkendali. Ia *hanya bisa* memilih dari ID valid yang kita berikan di dalam *prompt* dinamis, tidak bisa mengarang ID atau judul film sendiri.
4.  **Lebih Robust:** Jauh lebih baik dalam menangani variasi linguistik ("yang rabu", "jam 4 sore", "no. 3") tanpa perlu kode tambahan.
5.  **Perbaikan State yang Jelas:** Karena LLM mengembalikan ID yang pasti, *state* (`current_movie_id`, `current_showtime_id`) diperbarui secara andal, mencegah "kebocoran" alur.