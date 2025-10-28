import re
from typing import Iterable, List, Sequence
from sqlalchemy import select, insert
from sqlalchemy.exc import IntegrityError
from langchain_core.tools import tool

from db.schema import engine, movies_table, movie_genres_table, genres_table, showtimes_table, bookings_table
from data.seats import SEAT_MAP, ALL_VALID_SEATS


@tool
def search_movies(title: str = None, genre_name: str = None, **kwargs) -> dict:
    """Cari film berdasarkan judul atau genre dan kembalikan hasil terstruktur."""
    title = title or kwargs.get("movie_title") or kwargs.get("movie")
    genre_name = genre_name or kwargs.get("genre") or kwargs.get("genreId")
    stmt = select(movies_table.c.id, movies_table.c.title, movies_table.c.description).select_from(movies_table)
    if genre_name:
        stmt = stmt.join(movie_genres_table).join(genres_table).where(genres_table.c.name.ilike(f"%{genre_name}%"))
    if title:
        stmt = stmt.where(movies_table.c.title.ilike(f"%{title}%"))
    with engine.connect() as conn:
        results = conn.execute(stmt).fetchall()
    if not results:
        return {
            "message": "Film tidak ditemukan. Coba cari dengan genre atau judul lain.",
            "movies": [],
        }
    movies = [
        {
            "id": row.id,
            "title": row.title,
            "description": row.description,
        }
        for row in results
    ]
    summary_lines = [
        f"{idx + 1}. {item['title']} — {(item['description'] or '')[:80]}..."
        for idx, item in enumerate(movies)
    ]
    return {
        "message": (
            "Ditemukan film berikut:\n"
            + "\n".join(summary_lines)
            + "\nPilih dengan menyebut judul atau nomor urutnya."
        ),
        "movies": movies,
    }


@tool
def get_showtimes(movie_id: int = None, **kwargs) -> dict:
    """Ambil jadwal tayang untuk film tertentu dalam format terstruktur."""
    movie_id = movie_id or kwargs.get("id") or kwargs.get("film_id") or kwargs.get("movie")
    movie_id = _coerce_int(movie_id)
    if movie_id is None:
        return {
            "message": "Silakan berikan film yang mau dicek jadwalnya dulu, ya.",
            "showtimes": [],
        }
    stmt = select(showtimes_table.c.id, showtimes_table.c.time).where(showtimes_table.c.movie_id == movie_id)
    with engine.connect() as conn:
        results = conn.execute(stmt).fetchall()
    if not results:
        return {
            "message": "Maaf, belum ada jadwal tayang untuk film ini.",
            "showtimes": [],
        }
    showtimes = [
        {
            "id": row.id,
            "movie_id": movie_id,
            "time": row.time,
            "time_display": row.time.strftime("%A, %d %B %Y %H:%M"),
        }
        for row in results
    ]
    lines = [f"{idx + 1}. {item['time_display']}" for idx, item in enumerate(showtimes)]
    return {
        "message": (
            "Jadwal tersedia:\n"
            + "\n".join(lines)
            + "\nSebut jam/tanggal atau nomor urut jadwal yang kamu mau."
        ),
        "showtimes": showtimes,
    }


@tool
def get_available_seats(showtime_id: int = None, **kwargs) -> dict:
    """Daftar kursi yang masih tersedia untuk suatu jadwal tayang."""
    showtime_id = showtime_id or kwargs.get("schedule_id") or kwargs.get("id")
    showtime_id = _coerce_int(showtime_id)
    if showtime_id is None:
        return {
            "message": "Silakan sebutkan jadwal mana yang mau dicek kursinya.",
            "available_seats": [],
        }
    stmt = select(bookings_table.c.seat).where(bookings_table.c.showtime_id == showtime_id)
    with engine.connect() as conn:
        booked = {r.seat for r in conn.execute(stmt).fetchall()}
    available_rows = []
    available_flat: List[str] = []
    for row_letter, row_seats in zip("ABCDEFGHIJKLM", SEAT_MAP):
        if not any(row_seats):
            continue
        available_in_row = [s for s in row_seats if s and s not in booked]
        if available_in_row:
            available_rows.append(f"Baris {row_letter}: {', '.join(available_in_row)}")
            available_flat.extend(available_in_row)
    if not available_rows:
        return {
            "message": "Maaf, kursi untuk jadwal ini sudah penuh.",
            "available_seats": [],
        }
    return {
        "message": (
            "Kursi yang tersedia:\n"
            + "\n".join(available_rows)
            + "\nPilih kursi dengan menyebut kode seperti D7 atau E3."
        ),
        "available_seats": available_flat,
        "showtime_id": showtime_id,
    }


@tool
def book_tickets(
    showtime_id: int = None,
    seats: List[str] | Sequence[str] | str | None = None,
    user_name: str | None = None,
    **kwargs,
) -> dict:
    """Pesan kursi untuk pengguna dengan validasi kapasitas dan konflik."""
    showtime_id = showtime_id or kwargs.get("schedule_id") or kwargs.get("id")
    showtime_id = _coerce_int(showtime_id)
    if showtime_id is None:
        return {
            "success": False,
            "message": "Masih belum tahu jadwal mana yang mau dibooking. Boleh ulangi?",
        }

    user_name = user_name or kwargs.get("name") or kwargs.get("customer")
    if not user_name:
        return {
            "success": False,
            "message": "Nama pemesan wajib diisi dulu, ya.",
        }

    seats = seats or kwargs.get("seat_codes") or kwargs.get("seat") or kwargs.get("seat_list")
    seats = _normalize_seat_list(seats)
    if not seats:
        return {
            "success": False,
            "message": "Daftar kursi tidak valid. Coba sebutkan lagi kursinya.",
        }
    if len(seats) > 5:
        return {
            "success": False,
            "message": "Maaf, maksimal pemesanan sekaligus adalah 5 kursi.",
        }
    invalid = [s for s in seats if s not in ALL_VALID_SEATS]
    if invalid:
        return {
            "success": False,
            "message": f"Kursi tidak valid: {', '.join(invalid)}. Coba pilih kursi lain.",
        }
    insert_data = [{"showtime_id": showtime_id, "seat": s, "user_name": user_name} for s in seats]
    with engine.connect() as conn:
        try:
            with conn.begin():
                conn.execute(insert(bookings_table), insert_data)
            return {
                "success": True,
                "message": f"Sukses! Tiket untuk {user_name} di kursi {', '.join(seats)} telah dikonfirmasi.",
                "seats": seats,
                "showtime_id": showtime_id,
            }
        except IntegrityError:
            return {
                "success": False,
                "message": f"Salah satu kursi ({', '.join(seats)}) sudah terisi. Pilih kursi lain, ya.",
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Gagal memproses pemesanan. {e}",
            }


def _normalize_seat_list(value: Iterable[str] | str | None) -> List[str]:
    if value is None:
        return []
    seats: List[str] = []
    raw: Iterable[str]
    if isinstance(value, str):
        raw = [item for item in re.split(r"[,\s]+", value) if item]
    elif isinstance(value, Sequence):
        raw = value  # type: ignore[assignment]
    else:
        raw = list(value)

    seen = set()
    for seat in raw:
        if not isinstance(seat, str):
            continue
        normalized = seat.strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        seats.append(normalized)
    return seats


def _coerce_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None