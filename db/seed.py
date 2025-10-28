from datetime import datetime, timedelta
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError

from db.schema import (
    engine,
    metadata,
    genres_table,
    movies_table,
    movie_genres_table,
    showtimes_table,
)
from data.movies import SAMPLE_MOVIES


def seed_database():
    """Buat tabel dan isi SAMPLE_MOVIES -> genres, movies, showtimes (sama logika seperti run_tiketa)."""
    metadata.create_all(engine)

    with engine.connect() as conn:
        all_genres = set()
        for movie in SAMPLE_MOVIES:
            all_genres.update(movie.get("genres", []))

        genre_map = {}
        for genre_name in all_genres:
            result = conn.execute(insert(genres_table).values(name=genre_name).returning(genres_table.c.id))
            genre_map[genre_name] = result.fetchone()[0]

        today = datetime.now().date()
        movie_id_map = {}
        showtime_data = []
        movie_genre_data = []

        for movie in SAMPLE_MOVIES:
            result = conn.execute(
                insert(movies_table)
                .values(
                    title=movie["title"],
                    description=movie.get("description"),
                    studio_number=movie["studio_number"],
                    release_date=movie.get("release_date"),
                )
                .returning(movies_table.c.id)
            )
            movie_id = result.fetchone()[0]
            movie_id_map[movie["title"]] = movie_id

            showtime_data.extend(
                [
                    {"movie_id": movie_id, "time": datetime(today.year, today.month, today.day, 19, 0)},
                    {"movie_id": movie_id, "time": datetime(today.year, today.month, today.day, 21, 30)},
                    {"movie_id": movie_id, "time": datetime(today.year, today.month, today.day, 16, 0) + timedelta(days=1)},
                ]
            )

            for g in movie.get("genres", []):
                movie_genre_data.append({"movie_id": movie_id, "genre_id": genre_map[g]})

        if showtime_data:
            conn.execute(insert(showtimes_table), showtime_data)
        if movie_genre_data:
            conn.execute(insert(movie_genres_table), movie_genre_data)

        conn.commit()

    print("Database seeded.")