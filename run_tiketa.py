import os
import operator
import re
from copy import deepcopy
from datetime import date, datetime, timedelta
from typing import TypedDict, List, Optional, Literal, Annotated, Any

# LangChain & LangGraph
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# Database (SQLAlchemy)
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError  # Ini untuk error 'UniqueViolationError'

# Setup Lingkungan
from dotenv import load_dotenv

load_dotenv()

# Modul internal
from db.seed import seed_database
from db.schema import engine, movies_table, showtimes_table
from tools.bookings import (
    search_movies,
    get_showtimes,
    get_available_seats,
    book_tickets,
)
from data.seats import ALL_VALID_SEATS
from agent.workflow import compile_ticket_agent_workflow


def setup_environment():
    """Memuat semua environment variables, menyimpannya ke os.environ, dan menampilkan nilai TERMASKED."""

    def mask_value(val: str, visible_fraction: float = 0.5) -> str:
        if val is None:
            return ""
        s = str(val)
        n = len(s)
        if n <= 4:
            return "*" * n
        visible = max(1, int(n * visible_fraction))
        return s[:visible] + "*" * (n - visible)

    env_vars = [
        "GOOGLE_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGSMITH_TRACING",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_PROJECT",
    ]
    for var in env_vars:
        value = os.getenv(var)
        if not value:
            raise RuntimeError(
                f"{var} not found in environment. Set it in .env or export it."
            )
        os.environ[var] = value  # pastikan tersedia di sesi ini
        print(f"{var} Terload! Value: {mask_value(value)}")


setup_environment()

print(f"Total kursi valid yang dikenali: {len(ALL_VALID_SEATS)}")

# Fungsi untuk membuat dan mengisi database
# Jalankan seeder
seed_database()


class TicketAgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

    intent: Literal["browsing", "booking", "answering_question", "other"]

    movie_title: Optional[str]
    genre: Optional[str]
    current_movie_id: Optional[int]
    current_showtime_id: Optional[int]
    selected_seats: Optional[List[str]]
    user_name: Optional[str]
    candidate_movies: Optional[List[dict]]
    available_showtimes: Optional[List[dict]]
    available_seats: Optional[List[str]]

    current_question: Optional[
        Literal[
            "ask_movie", "ask_showtime", "ask_seats", "ask_confirmation", "ask_name"
        ]
    ]


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def _get_movie_title(movie_id: Optional[int]) -> Optional[str]:
    if not movie_id:
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                select(movies_table.c.title).where(movies_table.c.id == movie_id)
            ).fetchone()
        if row and row.title:
            return row.title
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"   > Peringatan: gagal mengambil judul film {movie_id}: {exc}")
    return None


def _get_showtime_info(showtime_id: Optional[int]) -> Optional[dict]:
    if not showtime_id:
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                select(
                    showtimes_table.c.id,
                    showtimes_table.c.movie_id,
                    showtimes_table.c.time,
                ).where(showtimes_table.c.id == showtime_id)
            ).fetchone()
        if row:
            return {
                "id": row.id,
                "movie_id": row.movie_id,
                "time": row.time,
                "time_display": row.time.strftime("%A, %d %B %Y %H:%M"),
            }
    except Exception as exc:  # pragma: no cover
        print(f"   > Peringatan: gagal mengambil jadwal {showtime_id}: {exc}")
    return None

# --- 4. Tool & Model untuk Classifier ---
@tool
def extract_intent_and_entities(
    intent: Literal["booking", "browsing", "answering_question", "other"],
    user_name: Optional[str] = None,
    movie_title: Optional[str] = None,
    genre: Optional[str] = None,
    movie_id: Optional[int] = None,
    showtime_id: Optional[int] = None,
    seats: Optional[List[str]] = None,
):
    """Mengekstrak niat dan entitas dari pesan pengguna."""
    return {
        "intent": intent,
        "user_name": user_name,
        "movie_title": movie_title,
        "genre": genre,
        "movie_id": movie_id,
        "showtime_id": showtime_id,
        "seats": seats,
    }


classifier_model = model.bind_tools([extract_intent_and_entities])

# --- 5. Kumpulan Tool untuk Agen ---
booking_tools = [search_movies, get_showtimes, get_available_seats, book_tickets]
booking_model = model.bind_tools(booking_tools)  # Model khusus untuk booking
browsing_tools = [search_movies, get_showtimes, get_available_seats]
browsing_model = model.bind_tools(browsing_tools)  # Model khusus untuk browsing


def _message_from_tool_result(result: Any) -> str:
    if isinstance(result, dict):
        message = result.get("message")
        if isinstance(message, str):
            return message
    return str(result)


DAY_ALIASES = {
    "monday": {"senin", "monday"},
    "tuesday": {"selasa", "tuesday"},
    "wednesday": {"rabu", "rabo", "wednesday"},
    "thursday": {"kamis", "kemis", "thursday"},
    "friday": {"jumat", "jum'at", "friday"},
    "saturday": {"sabtu", "saterday", "saturday"},
    "sunday": {"minggu", "ahad", "sunday"},
}

MONTH_ALIASES = {
    "january": {"januari", "jan", "january"},
    "february": {"februari", "feb", "february"},
    "march": {"maret", "mar", "march"},
    "april": {"april", "apr"},
    "may": {"mei", "may"},
    "june": {"juni", "jun", "june"},
    "july": {"juli", "jul", "july"},
    "august": {"agustus", "agust", "august", "aug"},
    "september": {"september", "sept", "sep"},
    "october": {"oktober", "okt", "october", "oct"},
    "november": {"november", "nov"},
    "december": {"desember", "des", "december", "dec"},
}

ORDINAL_WORDS = {
    "pertama": 0,
    "kesatu": 0,
    "kedua": 1,
    "keduanya": 1,
    "ketiga": 2,
    "keempat": 3,
    "kelima": 4,
    "keenam": 5,
    "ketujuh": 6,
}

YES_WORDS = {"ya", "iyah", "iya", "yes", "ok", "oke", "sip", "lanjut", "gas"}
NO_WORDS = {"tidak", "gak", "ga", "enggak", "no", "ntar", "nanti", "belum"}

STOPWORDS = {"the", "film", "movie", "saya", "aku", "mau", "dong", "lah", "yang", "itu", "itu", "itu", "ini"}


def _match_movie_from_text(text: str, candidates: Optional[List[dict]]) -> tuple[Optional[int], Optional[str]]:
    if not text or not candidates:
        return None, None
    text_lower = text.lower()
    digits = re.findall(r"\b\d+\b", text_lower)
    if digits:
        try:
            idx = int(digits[0])
            if 1 <= idx <= len(candidates):
                movie = candidates[idx - 1]
                return movie.get("id"), movie.get("title")
        except ValueError:
            pass

    text_tokens = set(
        tok for tok in re.findall(r"[a-z0-9]+", text_lower)
        if len(tok) > 1 and tok not in STOPWORDS
    )

    best_movie = None
    best_score = 0

    for movie in candidates:
        title = (movie.get("title") or "").strip()
        if not title:
            continue
        normalized_title = re.sub(r"\s+", " ", title).lower()
        if normalized_title in text_lower:
            score = 100
        else:
            title_tokens = [
                tok
                for tok in re.findall(r"[a-z0-9]+", normalized_title)
                if len(tok) > 1 and tok not in STOPWORDS
            ]
            if not title_tokens:
                continue
            token_matches = sum(1 for tok in title_tokens if tok in text_tokens)
            longest_overlap = max((len(tok) for tok in title_tokens if tok in text_tokens), default=0)
            score = token_matches * 10 + longest_overlap
            if normalized_title.split() and normalized_title.split()[0] in {"the", "a", "an"}:
                score += token_matches
            if title_tokens and token_matches == len(title_tokens):
                score += 15
            if token_matches == 1 and len(title_tokens) > 2:
                score -= 5

        if score > best_score:
            best_score = score
            best_movie = movie

    if best_movie and best_score >= 15:
        return best_movie.get("id"), best_movie.get("title")

    return None, None


def _match_showtime_from_text(text: str, showtimes: Optional[List[dict]]) -> Optional[dict]:
    if not text or not showtimes:
        return None
    text_lower = text.lower()

    id_matches = re.findall(r"(?:id|jadwal)\s*(\d+)", text_lower)
    for match in id_matches:
        try:
            candidate_id = int(match)
            for show in showtimes:
                if show.get("id") == candidate_id:
                    return show
        except ValueError:
            continue

    for word, index in ORDINAL_WORDS.items():
        if word in text_lower and 0 <= index < len(showtimes):
            return showtimes[index]

    numbers = [n for n in re.findall(r"\b\d{1,4}\b", text_lower) if n not in {"24"}]
    for number in numbers:
        try:
            candidate_id = int(number)
        except ValueError:
            continue
        if any(value in text_lower for value in (":", ".")):
            break
        for show in showtimes:
            if show.get("id") == candidate_id:
                return show

    best_match = None
    best_score = 0
    for show in showtimes:
        dt = show.get("time")
        if not isinstance(dt, datetime):
            continue
        score = 0
        hm = dt.strftime("%H:%M").lower()
        hm_alt = dt.strftime("%H.%M").lower()
        hour = dt.strftime("%H").lower()
        if hm in text_lower or hm_alt in text_lower:
            score += 5
        elif f"{hour}" in text_lower:
            score += 2
        if "malam" in text_lower and 18 <= int(hour) <= 23:
            score += 1
        if "siang" in text_lower and 12 <= int(hour) < 18:
            score += 1
        if "pagi" in text_lower and int(hour) < 12:
            score += 1
        day_tokens = DAY_ALIASES.get(dt.strftime("%A").lower(), {dt.strftime("%A").lower()})
        if any(token in text_lower for token in day_tokens):
            score += 3
        date_token = dt.strftime("%d").lstrip("0")
        if date_token and re.search(rf"\b{date_token}\b", text_lower):
            score += 2
        date_combo = dt.strftime("%d/%m")
        if date_combo in text_lower:
            score += 3
        date_combo_dash = dt.strftime("%d-%m")
        if date_combo_dash in text_lower:
            score += 3
        month_tokens = MONTH_ALIASES.get(dt.strftime("%B").lower(), {dt.strftime("%B").lower()})
        if any(token in text_lower for token in month_tokens):
            score += 2
        if score > best_score:
            best_score = score
            best_match = show

    if best_match and best_score > 0:
        return best_match

    return None


def _detect_confirmation(text: str) -> Optional[bool]:
    if not text:
        return None
    lowered = text.strip().lower()
    if lowered in YES_WORDS:
        return True
    if lowered in NO_WORDS:
        return False
    return None


def _format_showtime_label(show: Optional[dict]) -> str:
    if not show:
        return "(belum dipilih)"
    if isinstance(show, dict):
        if isinstance(show.get("time"), datetime):
            return show["time"].strftime("%A, %d %B %Y %H:%M")
        return show.get("time_display") or str(show.get("id"))
    return str(show)


def _format_seat_rows(seats: Optional[List[str]]) -> List[str]:
    if not seats:
        return []
    rows: dict[str, List[str]] = {}
    for seat in seats:
        if not seat:
            continue
        row_letter = seat[0]
        rows.setdefault(row_letter, []).append(seat)
    formatted = []
    for row_letter in sorted(rows.keys()):
        sorted_row = sorted(rows[row_letter], key=lambda x: (len(x), x))
        formatted.append(f"Baris {row_letter}: {', '.join(sorted_row)}")
    return formatted


def node_classify_intent(state: TicketAgentState):
    """Node pertama: Mengklasifikasikan niat DAN mengekstrak entitas."""
    print("--- NODE: Classify Intent ---")

    messages = state.get("messages", [])
    print(f"   > State keys: {list(state.keys())}")
    if messages:
        print(
            f"   > Pesan terakhir: {messages[-1].__class__.__name__} -> {getattr(messages[-1], 'content', '')}"
        )
    if not messages:
        print("   > Tidak ada pesan pada state, fallback 'other'")
        return {"intent": "other"}

    latest_message_raw = messages[-1].content
    latest_message = latest_message_raw.lower()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Anda adalah asisten resepsionis. Tugas Anda adalah menganalisis pesan terakhir pengguna dan mengekstrak niat serta informasi (entitas) yang relevan. Gunakan tool 'extract_intent_and_entities' untuk mengembalikan hasilnya.",
            ),
            ("user", "{input}"),
        ]
    )
    classifier_chain = prompt | classifier_model
    response = classifier_chain.invoke({"input": messages[-1].content})

    if not response.tool_calls:
        print("   > Classifier gagal, kembali ke 'other'")
        return {"intent": "other"}

    tool_call_args = response.tool_calls[0]["args"]
    updates = {"intent": tool_call_args.get("intent", "other")}

    key_mapping = {
        "movie_id": "current_movie_id",
        "current_movie_id": "current_movie_id",
        "showtime_id": "current_showtime_id",
        "current_showtime_id": "current_showtime_id",
        "seats": "selected_seats",
        "selected_seats": "selected_seats",
    }

    for source_key in [
        "user_name",
        "movie_title",
        "genre",
        "movie_id",
        "current_movie_id",
        "showtime_id",
        "current_showtime_id",
        "seats",
        "selected_seats",
    ]:
        target_key = key_mapping.get(source_key, source_key)
        value = tool_call_args.get(source_key)
        if value in (None, ""):
            continue

        normalized_value = value

        if target_key in {"current_movie_id", "current_showtime_id"}:
            try:
                normalized_value = int(value)
            except (TypeError, ValueError):
                pass
        elif target_key == "selected_seats":
            if isinstance(value, str):
                normalized_value = [
                    seat.strip().upper()
                    for seat in re.split(r"[,\s]+", value)
                    if seat.strip()
                ]
            elif isinstance(value, (set, tuple)):
                normalized_value = [
                    seat.strip().upper()
                    for seat in value
                    if isinstance(seat, str) and seat.strip()
                ]
            elif isinstance(value, list):
                normalized_value = [
                    seat.strip().upper()
                    for seat in value
                    if isinstance(seat, str) and seat.strip()
                ]
        elif target_key == "movie_title" and isinstance(normalized_value, str):
            if not latest_message_raw.strip():
                continue
            title_tokens = [
                tok
                for tok in re.findall(r"[a-z0-9]+", normalized_value.lower())
                if len(tok) > 2 and tok not in STOPWORDS
            ]
            if title_tokens and not any(tok in latest_message for tok in title_tokens):
                continue

        if not normalized_value:
            continue

        if state.get(target_key) != normalized_value:
            updates[target_key] = normalized_value
            print(f"   > Info '{target_key}' ditangkap: {updates[target_key]}")

    tentative_intent = updates.get("intent", state.get("intent", "other"))
    booking_keywords = [
        "pesan",
        "booking",
        "kursi",
        "seat",
        "kursinya",
        "kursi nya",
        "cek seat",
        "cek kursi",
        "ambil",
        "mau",
    ]
    if tentative_intent in {"other", "browsing"} and any(
        kw in latest_message for kw in booking_keywords
    ):
        updates["intent"] = "booking"
        if "current_question" not in updates:
            if any(word in latest_message for word in ["kursi", "seat"]):
                updates["current_question"] = "ask_seats"
            elif any(word in latest_message for word in ["jam", ":", "jadwal", "hari"]):
                updates["current_question"] = "ask_showtime"

    if (
        not updates.get("current_movie_id")
        and not state.get("current_movie_id")
        and updates.get("movie_title")
    ):
        updates.setdefault("current_question", "ask_movie")

    # --- TAMBAHAN LOGIKA ---
    if (
        state.get("current_movie_id")
        and not updates.get("current_showtime_id")
        and any(kw in latest_message for kw in ["kapan", "jadwal", "tayang", "jam", "showtime"])
    ):
        print("    > Heuristik: Terdeteksi pertanyaan jadwal, memaksa intent 'booking'.")
        updates["intent"] = "booking"
        available_showtimes = state.get("available_showtimes")
        if available_showtimes:
            match = _match_showtime_from_text(latest_message_raw, available_showtimes)
            if match:
                updates["current_showtime_id"] = match.get("id")
                updates["intent"] = "answering_question"
                print(f"    > Heuristik: Jadwal {match.get('id')} ditangkap.")
    elif (
        state.get("current_showtime_id")
        and not updates.get("selected_seats")
        and any(kw in latest_message for kw in ["kursi", "seat", "bangku"])
    ):
        print("    > Heuristik: Terdeteksi pertanyaan kursi, memaksa intent 'booking'.")
        updates["intent"] = "booking"
        seat_candidates = [
            seat
            for seat in re.findall(r"[A-Za-z][0-9]{1,2}", latest_message_raw.upper())
            if seat in ALL_VALID_SEATS
        ]
        if seat_candidates:
            updates["selected_seats"] = seat_candidates
            updates["intent"] = "answering_question"
            print(f"    > Heuristik: Kursi {seat_candidates} ditangkap.")
    # --- AKHIR TAMBAHAN ---

    current_question = state.get("current_question")
    candidate_movies = updates.get("candidate_movies") or state.get("candidate_movies")
    available_showtimes = updates.get("available_showtimes") or state.get("available_showtimes")

    if current_question == "ask_movie" and not updates.get("current_movie_id"):
        movie_id, movie_name = _match_movie_from_text(latest_message_raw, candidate_movies)
        if movie_id:
            updates["current_movie_id"] = movie_id
            if movie_name:
                updates["movie_title"] = movie_name
            else:
                title_lookup = _get_movie_title(movie_id)
                if title_lookup:
                    updates["movie_title"] = title_lookup
            updates["intent"] = "answering_question"

    if current_question == "ask_showtime" and not updates.get("current_showtime_id"):
        match = _match_showtime_from_text(latest_message_raw, available_showtimes)
        if match:
            updates["current_showtime_id"] = match.get("id")
            movie_id_from_show = match.get("movie_id")
            if movie_id_from_show and not updates.get("current_movie_id"):
                updates["current_movie_id"] = movie_id_from_show
            updates["intent"] = "answering_question"

    if current_question == "ask_seats" and not updates.get("selected_seats"):
        seat_candidates = [
            seat
            for seat in re.findall(r"[A-Za-z][0-9]{1,2}", latest_message_raw.upper())
            if seat in ALL_VALID_SEATS
        ]
        if seat_candidates:
            updates["selected_seats"] = seat_candidates
            updates["intent"] = "answering_question"

    if current_question == "ask_seats" and updates.get("selected_seats"):
        updates["intent"] = "answering_question"

    if current_question == "ask_name" and updates.get("user_name"):
        updates["intent"] = "answering_question"

    if current_question and updates.get("intent") == "other":
        prior_intent = state.get("intent")
        updates["intent"] = prior_intent if prior_intent and prior_intent != "other" else "booking"

    return updates


def node_browsing_agent(state: TicketAgentState):
    """Agen ReAct loop sederhana untuk Q&A (tapi diimplementasikan sbg 1 langkah)."""
    print("--- NODE: Browsing Agent ---")

    response = browsing_model.invoke(state["messages"])
    tool_calls = getattr(response, "tool_calls", []) or []
    if not tool_calls:
        return {"messages": [response]}

    tool_outputs: List[ToolMessage] = []
    resolved_calls: List[tuple[str, dict, Any]] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id")
        else:
            tool_name = getattr(tool_call, "name", None)
            tool_args = getattr(tool_call, "args", {})
            tool_id = getattr(tool_call, "id", None)

        if not tool_name:
            print("   > Peringatan: tool call tanpa nama diabaikan.")
            continue

        tool_impl = next((t for t in browsing_tools if t.name == tool_name), None)
        if tool_impl is None:
            print(f"   > Peringatan: tool '{tool_name}' tidak dikenal, diabaikan.")
            continue

        output = tool_impl.invoke(tool_args)
        message_text = _message_from_tool_result(output)
        tool_outputs.append(
            ToolMessage(
                content=message_text,
                name=tool_name,
                tool_call_id=tool_id or f"{tool_name}__manual",
            )
        )
        resolved_calls.append((tool_name, tool_args, output))

    if not tool_outputs:
        return {"messages": [response]}

    messages_with_tool_results = state.get("messages", []) + [response] + tool_outputs
    final_response = model.invoke(messages_with_tool_results)
    state_updates: dict = {"messages": [response] + tool_outputs + [final_response]}

    for tool_name, tool_args, tool_output in resolved_calls:
        if tool_name == "search_movies":
            state_updates.setdefault("intent", "browsing")
            state_updates.setdefault("current_question", "ask_movie")
            if isinstance(tool_output, dict):
                movies = tool_output.get("movies")
                if movies is not None:
                    state_updates["candidate_movies"] = movies
                    if len(movies) == 1:
                        only_movie = movies[0]
                        state_updates["current_movie_id"] = only_movie.get("id")
                        state_updates["movie_title"] = only_movie.get("title")
        elif tool_name == "get_showtimes":
            state_updates["intent"] = "booking"
            state_updates["current_question"] = "ask_showtime"
            movie_id = tool_args.get("movie_id") or tool_args.get("id")
            try:
                if movie_id is not None:
                    state_updates["current_movie_id"] = int(movie_id)
            except (TypeError, ValueError):
                pass
            if isinstance(tool_output, dict):
                showtimes = tool_output.get("showtimes")
                if showtimes is not None:
                    state_updates["available_showtimes"] = showtimes
                    if len(showtimes) == 1:
                        state_updates["current_showtime_id"] = showtimes[0].get("id")
                    if not state_updates.get("movie_title"):
                        title = _get_movie_title(state_updates.get("current_movie_id"))
                        if title:
                            state_updates["movie_title"] = title
        elif tool_name == "get_available_seats":
            state_updates["intent"] = "booking"
            state_updates["current_question"] = "ask_seats"
            showtime_id = (
                tool_args.get("showtime_id")
                or tool_args.get("schedule_id")
                or tool_args.get("id")
            )
            try:
                if showtime_id is not None:
                    state_updates["current_showtime_id"] = int(showtime_id)
            except (TypeError, ValueError):
                pass
            if isinstance(tool_output, dict):
                avail = tool_output.get("available_seats")
                if avail is not None:
                    state_updates["available_seats"] = avail
                if tool_output.get("showtime_id") and not state_updates.get("current_showtime_id"):
                    state_updates["current_showtime_id"] = tool_output.get("showtime_id")

    return state_updates


def node_find_movie(state: TicketAgentState):
    """Langkah 1 Pemesanan: Mengidentifikasi film."""
    print("--- NODE: Find Movie ---")

    title = state.get("movie_title")
    genre = state.get("genre")

    tool_args = {}
    if title:
        tool_args["title"] = title
    if genre:
        tool_args["genre_name"] = genre

    if tool_args:
        print(f"    > Mencari film dengan args: {tool_args}")
        result = search_movies.invoke(tool_args)
        message_text = _message_from_tool_result(result)
        movies = result.get("movies") if isinstance(result, dict) else None
        tool_msg = ToolMessage(
            content=message_text,
            name="search_movies",
            tool_call_id="search_movies__manual",
        )

        if movies:
            lines = []
            for idx, movie in enumerate(movies, start=1):
                snippet = (movie.get("description") or "").strip()
                if snippet:
                    snippet = snippet[:120] + ("..." if len(snippet) > 120 else "")
                lines.append(f"{idx}. {movie.get('title', 'Tanpa Judul')} — {snippet or 'Deskripsi tidak tersedia'}")
            response_text = (
                "Berikut daftar film yang cocok:\n"
                + "\n".join(lines)
                + "\nSilakan pilih dengan menyebut judul atau nomor urutnya."
            )
        else:
            response_text = message_text + " Mau coba cari genre atau judul lain?"

        ai_response = AIMessage(content=response_text)

        updates: dict[str, Any] = {
            "messages": [tool_msg, ai_response],
            "current_question": "ask_movie",
        }
        if movies is not None:
            updates["candidate_movies"] = movies
            if len(movies) == 1:
                only_movie = movies[0]
                updates["current_movie_id"] = only_movie.get("id")
                updates["movie_title"] = only_movie.get("title")

        if not updates.get("movie_title") and state.get("current_movie_id"):
            title_lookup = _get_movie_title(state.get("current_movie_id"))
            if title_lookup:
                updates["movie_title"] = title_lookup

        return updates

    return {
        "messages": [AIMessage(content="Tentu, mau cari film apa atau genre apa?")],
        "current_question": "ask_movie",
    }


def node_find_showtime(state: TicketAgentState):
    """Langkah 2 Pemesanan: Mengidentifikasi jadwal."""
    print("--- NODE: Find Showtime ---")
    movie_id = state.get("current_movie_id")

    # 1. Penanganan Error: Jika user entah bagaimana sampai di sini tanpa ID film
    if not movie_id:
        return {
            "messages": [
                AIMessage(
                    content="Ups, sepertinya saya belum tahu Anda mau film apa. Bisa sebutkan judul filmnya?"
                )
            ],
            "intent": "booking",  # Paksa kembali ke booking (meskipun seharusnya sudah)
            "current_question": "ask_movie",  # Set pertanyaan kembali ke film
        }

    # 2. Panggil Tool untuk mendapatkan data mentah
    result = get_showtimes.invoke({"movie_id": movie_id})
    message_text = _message_from_tool_result(result)
    showtimes = result.get("showtimes") if isinstance(result, dict) else None
    tool_msg = ToolMessage(
        content=message_text,
        name="get_showtimes",
        tool_call_id="get_showtimes__manual",
    )

    movie_title = state.get("movie_title") or _get_movie_title(movie_id)
    if showtimes:
        lines = [
            f"{idx}. {item.get('time_display')}"
            for idx, item in enumerate(showtimes, start=1)
        ]
        movie_label = movie_title or "film ini"
        response_text = (
            f"Jadwal tayang untuk {movie_label} yang tersedia:\n"
            + "\n".join(lines)
            + "\nSebutkan nomor urut atau jam/tanggal yang kamu inginkan."
        )
    else:
        response_text = message_text + " Mau cek film lain?"

    ai_response = AIMessage(content=response_text)

    # 4. Kembalikan state
    updates: dict[str, Any] = {
        "messages": [tool_msg, ai_response],
        "current_question": "ask_showtime",
    }
    if showtimes is not None:
        updates["available_showtimes"] = showtimes
        if len(showtimes) == 1:
            updates["current_showtime_id"] = showtimes[0].get("id")
    if movie_title and movie_title != state.get("movie_title"):
        updates["movie_title"] = movie_title

    return updates


def node_select_seats(state: TicketAgentState):
    """Langkah 3 Pemesanan: Mengidentifikasi kursi."""
    print("--- NODE: Select Seats ---")
    showtime_id = state.get("current_showtime_id")

    # 1. Penanganan Error: Jika user sampai di sini tanpa ID jadwal
    if not showtime_id:
        return {
            "messages": [
                AIMessage(
                    content="Ups, sepertinya saya belum tahu Anda mau jadwal jam berapa. Bisa pilih salah satu jadwalnya?"
                )
            ],
            "intent": "booking",
            "current_question": "ask_showtime",  # Kembalikan ke pertanyaan jadwal
        }

    # 2. Panggil Tool untuk mendapatkan data mentah
    result = get_available_seats.invoke({"showtime_id": showtime_id})
    message_text = _message_from_tool_result(result)
    available_seats = result.get("available_seats") if isinstance(result, dict) else None
    tool_msg = ToolMessage(
        content=message_text,
        name="get_available_seats",
        tool_call_id="get_available_seats__manual",
    )

    if available_seats:
        seat_rows = _format_seat_rows(available_seats)
        if seat_rows:
            display_rows = seat_rows[:8]
            remaining = max(len(seat_rows) - len(display_rows), 0)
            rows_text = "\n".join(display_rows)
            if remaining:
                rows_text += f"\n... dan {remaining} baris lainnya masih kosong."
        else:
            rows_text = message_text
        response_text = (
            "Kursi yang masih tersedia:\n"
            + rows_text
            + "\nSebutkan kursi yang kamu mau, contoh D7 atau E3."
        )
    else:
        response_text = message_text

    ai_response = AIMessage(content=response_text)

    # 4. Kembalikan state
    updates: dict[str, Any] = {
        "messages": [tool_msg, ai_response],
        "current_question": "ask_seats",
    }
    if available_seats is not None:
        updates["available_seats"] = available_seats

    return updates


def node_confirm_booking(state: TicketAgentState):
    """Langkah 4 Pemesanan: Meminta konfirmasi nama & detail."""
    print("--- NODE: Confirm Booking ---")

    if not state.get("user_name"):
        return {
            "messages": [AIMessage(content="Baik, pemesanan ini atas nama siapa?")],
            "current_question": "ask_name",
        }

    seats = state.get("selected_seats") or []
    seat_list = ", ".join(seats) if seats else "(belum dipilih)"

    movie_id = state.get("current_movie_id")
    movie_title = state.get("movie_title") or _get_movie_title(movie_id)
    movie_label = movie_title or (f"Film ID {movie_id}" if movie_id else "(belum dipilih)")

    showtime_id = state.get("current_showtime_id")
    showtimes = state.get("available_showtimes") or []
    matched_show = None
    if isinstance(showtimes, list) and showtime_id is not None:
        matched_show = next((s for s in showtimes if s.get("id") == showtime_id), None)
    if matched_show is None and showtime_id is not None:
        matched_show = _get_showtime_info(showtime_id)
    showtime_label = _format_showtime_label(matched_show)
    if showtime_label == "(belum dipilih)" and showtime_id is not None:
        showtime_label = f"Jadwal ID {showtime_id}"
    summary = (
        "Konfirmasi pesanan:\n"
        f"- Film: {movie_label}\n"
        f"- Jadwal: {showtime_label}\n"
        f"- Kursi: {seat_list}\n"
        f"- Atas Nama: {state['user_name']}\n\n"
        "Apakah sudah benar? (ya/tidak)"
    )
    return {
        "messages": [AIMessage(content=summary)],
        "current_question": "ask_confirmation",
    }


def node_execute_booking(state: TicketAgentState):
    """
    Langkah 5 Pemesanan: Menjalankan tool 'book_tickets'.
    Fungsi ini sekarang mencakup:
    1. Validasi data tidak lengkap.
    2. Validasi bahwa kursi yang dipilih ada di SEAT_MAP.
    3. Reset state setelah pemesanan (sukses atau gagal) agar alur bisa dimulai dari awal.
    """
    print("--- NODE: Execute Booking (FINAL) ---")

    showtime_id = state.get("current_showtime_id")
    seats = state.get("selected_seats")
    user_name = state.get("user_name")

    # --- PERBAIKAN: Validasi 1 (Data Tidak Lengkap) ---
    if not all([showtime_id, seats, user_name]):
        print("   > Error: Data tidak lengkap.")
        return {
            "messages": [
                AIMessage(
                    content="Maaf, terjadi kesalahan. Data pemesanan tidak lengkap. Mari kita ulangi dari awal."
                )
            ],
            # Reset state di sini juga, untuk menghindari loop error
            "intent": "other",
            "movie_title": None,
            "genre": None,
            "current_movie_id": None,
            "current_showtime_id": None,
            "selected_seats": None,
            "candidate_movies": None,
            "available_showtimes": None,
            "available_seats": None,
            "current_question": None,
        }

    # --- PERBAIKAN: Validasi 2 (Kursi Tidak Valid / Z99) ---
    # Gunakan ALL_VALID_SEATS yang sudah kita definisikan secara global
    invalid_seats = [s for s in seats if s not in ALL_VALID_SEATS]

    if invalid_seats:
        print(f"   > Error: Kursi tidak valid: {invalid_seats}")
        error_msg = f"Maaf, kursi {', '.join(invalid_seats)} tidak ada dalam denah. Silakan pilih ulang kursi yang tersedia."

        # Kembalikan pesan error & minta user pilih ulang
        return {
            "messages": [AIMessage(content=error_msg)],
            "selected_seats": None,  # Hapus kursi yang salah
            "current_question": "ask_seats",  # Arahkan user untuk pilih ulang
            # Kita JANGAN reset state lain (movie_id, etc.)
            # Kita hanya ingin user mengulang langkah pemilihan kursi.
        }

    # --- Jika lolos semua validasi, baru panggil tool ---
    print(f"   > Mencoba memesan: {user_name} | Jadwal {showtime_id} | Kursi {seats}")
    result = book_tickets.invoke(
        {
            "showtime_id": showtime_id,
            "seats": seats,
            "user_name": user_name,
        }
    )
    print(f"   > Hasil Tool: {result}")
    message_text = _message_from_tool_result(result)

    # --- PERBAIKAN: 3 (Reset State) ---
    # Kembalikan hasil tool DAN reset state agar alur selesai
    return {
        "messages": [
            ToolMessage(
                content=message_text,
                name="book_tickets",
                tool_call_id="book_tickets__manual",  # ID manual
            )
        ],
        # Reset semua state yang berhubungan dengan alur booking ini
        "intent": "other",  # Setel ulang niat ke default
        "movie_title": None,
        "genre": None,
        "current_movie_id": None,
        "current_showtime_id": None,
        "selected_seats": None,
        "candidate_movies": None,
        "available_showtimes": None,
        "available_seats": None,
        # Kita biarkan user_name agar agen tetap ingat namanya
        "current_question": None,  # Ini yang paling penting
    }


def node_final_response(state: TicketAgentState):
    """Node terakhir. Memberi pesan sukses atau gagal ke user."""
    print("--- NODE: Final Response ---")

    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage):
        response_content = f"Status Pemesanan: {last_message.content}"
        return {"messages": [AIMessage(content=response_content)]}

    # Jika pesan terakhir sudah berupa AIMessage (mis. error atau klarifikasi), biarkan saja.
    return {}


# Router utama (si resepsionis cerdas)
def main_router(state: TicketAgentState):
    """Router cerdas yang memprioritaskan alur booking."""
    print("--- ROUTER: Main Router ---")
    intent = state.get("intent", "other")
    question = state.get("current_question")
    messages = state.get("messages", [])
    print(f"    > Niat: {intent}, Pertanyaan Terakhir: {question}")

    # --- Prioritas 1: Cek Konfirmasi Booking ---
    if question == "ask_confirmation" and messages and isinstance(messages[-1], HumanMessage):
        confirmation = _detect_confirmation(messages[-1].content)
        if confirmation is True:
            print("    > Router: Konfirmasi 'ya' terdeteksi. Lanjut eksekusi.")
            return "execute_booking"
        if confirmation is False:
            print("    > Router: Konfirmasi 'tidak' terdeteksi. Batal.")
            return "__end__"

        print("    > Router: Jawaban konfirmasi tidak jelas. Tanya lagi.")
        return "confirm_booking"

    # --- Prioritas 2: Cek jika AI baru saja merespons ---
    if messages and isinstance(messages[-1], AIMessage):
        return "__end__"

    # --- Prioritas 3: Cek 'Happy Path' (User baru menjawab pertanyaan) ---
    if intent == "answering_question":
        if question == "ask_movie":
            return "find_showtime"
        if question == "ask_showtime":
            return "select_seats"
        if question == "ask_seats":
            return "confirm_booking"
        if question == "ask_name":
            return "confirm_booking"

    # --- Prioritas 4: Cek alur booking (jika state belum lengkap) ---
    if intent == "booking" or (question and intent == "other"):
        if question == "ask_showtime" and not state.get("current_showtime_id"):
            print(f"    > Router: Memaksa kembali ke 'find_showtime' (pertanyaan: {question})")
            return "find_showtime"

        if not state.get("current_movie_id"):
            return "find_movie"
        if not state.get("current_showtime_id"):
            return "find_showtime"
        if not state.get("selected_seats"):
            return "select_seats"

        return "confirm_booking"

    # --- Prioritas 5: Fallback ke Q&A (Browsing) ---
    print("    > Router: Niat 'other'/'browsing' terdeteksi, merutekan ke browsing_agent.")
    return "browsing_agent"


# Kompilasi graph menggunakan helper modularada film
app = compile_ticket_agent_workflow(
    state_type=TicketAgentState,
    router=main_router,
    classify_intent=node_classify_intent,
    browsing_agent=node_browsing_agent,
    find_movie=node_find_movie,
    find_showtime=node_find_showtime,
    select_seats=node_select_seats,
    confirm_booking=node_confirm_booking,
    execute_booking=node_execute_booking,
    final_response=node_final_response,
)

# Tampilkan graph (sekarang seharusnya sudah terhubung)
print("Graph berhasil di-compile. Menampilkan visualisasi...")

# Kode untuk buat gambar mermaid PNG
try:
    print("Mencoba membuat visualisasi graph...")
    png_data = app.get_graph().draw_mermaid_png()
    with open("workflow_graph.png", "wb") as f:
        f.write(png_data)
    print("Visualisasi graph berhasil disimpan ke 'workflow_graph.png'")
except Exception as e:
    print(f"Gagal membuat visualisasi graph. Error: {e}")
    print(
        "(Ini tidak menghentikan agen, tapi Anda mungkin perlu 'pip install pygraphviz' atau 'playwright' untuk visualisasi.)"
    )


INITIAL_STATE_TEMPLATE: TicketAgentState = {
    "messages": [],
    "intent": "other",
    "movie_title": None,
    "genre": None,
    "current_movie_id": None,
    "current_showtime_id": None,
    "selected_seats": None,
    "user_name": None,
    "candidate_movies": None,
    "available_showtimes": None,
    "available_seats": None,
    "current_question": None,
}

session_states: dict[str, TicketAgentState] = {}


def hydrate_state(state: Optional[TicketAgentState]) -> TicketAgentState:
    base_state = deepcopy(INITIAL_STATE_TEMPLATE)
    if state:
        # Gunakan .get untuk menangani state parsial dari graph
        for key, value in state.items():
            if key == "messages":
                base_state["messages"] = list(value or [])
            else:
                base_state[key] = value
    return base_state


# Jalankan Chat Loop
print("\n--- Agen Bioskop Siap! ---")
print("Ketik 'exit' untuk keluar.")
print(
    "Contoh: 'Film action apa yang ada?', 'Saya Rafi, mau pesan tiket The Dark Knight', 'Pilih ID 101', 'Kursi D1, D2', 'ya'"
)

SESSION_ID = "user_123_notebook"

while True:
    try:
        user_input = input("\nAnda: ")
        if user_input.lower() == "exit":
            break

        config = {"configurable": {"session_id": SESSION_ID}}
        input_message = HumanMessage(content=user_input)

        current_state = hydrate_state(session_states.get(SESSION_ID))
        existing_messages = list(current_state.get("messages", []))
        existing_messages.append(input_message)
        current_state["messages"] = existing_messages

        messages_before_run = len(existing_messages)

        print("\nAgen:")
        result_state = app.invoke(current_state, config=config)
        session_states[SESSION_ID] = hydrate_state(result_state)

        all_messages = list(result_state.get("messages", []))
        new_messages = all_messages[messages_before_run:]
        ai_responses = [m for m in new_messages if isinstance(m, AIMessage)]

        if not ai_responses:
            print("(Tidak ada respons dari agen)")
        else:
            for message in ai_responses:
                print(message.content)

    except KeyboardInterrupt:
        print("\nBerhenti...")
        break
    except Exception as e:
        print(f"\nTerjadi error: {e}")
        break


