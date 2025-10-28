-- Database Schema for Tiketa RDS:
-- This file is for reference and documentation purposes.

-- DBeaver DDL Script
-- Tables
-- public.genres definition

-- Drop table
-- DROP TABLE public.genres;

CREATE TABLE public.genres (
    id serial4 NOT NULL,
    "name" varchar(120) NOT NULL,
    created_at timestamp NULL,
    CONSTRAINT genres_name_key UNIQUE (name),
    CONSTRAINT genres_pkey PRIMARY KEY (id)
);


-- public.movies definition

-- Drop table
-- DROP TABLE public.movies;

CREATE TABLE public.movies (
    id serial4 NOT NULL,
    title varchar(255) NOT NULL,
    description text NULL,
    studio_number int4 NOT NULL,
    poster_path varchar(255) NULL,
    backdrop_path varchar(255) NULL,
    release_date date NULL,
    trailer_youtube_id varchar(20) NULL,
    created_at timestamp NULL,
    CONSTRAINT movies_pkey PRIMARY KEY (id),
    CONSTRAINT movies_studio_number_key UNIQUE (studio_number)
);


-- public.movie_genres definition

-- Drop table
-- DROP TABLE public.movie_genres;

CREATE TABLE public.movie_genres (
    movie_id int4 NOT NULL,
    genre_id int4 NOT NULL,
    CONSTRAINT movie_genres_pkey PRIMARY KEY (movie_id, genre_id),
    CONSTRAINT movie_genres_genre_id_fkey FOREIGN KEY (genre_id) REFERENCES public.genres(id),
    CONSTRAINT movie_genres_movie_id_fkey FOREIGN KEY (movie_id) REFERENCES public.movies(id)
);


-- public.showtimes definition

-- Drop table
-- DROP TABLE public.showtimes;

CREATE TABLE public.showtimes (
    id serial4 NOT NULL,
    movie_id int4 NOT NULL,
    "time" timestamp NOT NULL,
    is_archived bool NOT NULL,
    created_at timestamp NULL,
    CONSTRAINT showtimes_pkey PRIMARY KEY (id),
    CONSTRAINT showtimes_movie_id_fkey FOREIGN KEY (movie_id) REFERENCES public.movies(id)
);


-- public.bookings definition

-- Drop table
-- DROP TABLE public.bookings;

CREATE TABLE public.bookings (
    id serial4 NOT NULL,
    "user" varchar(255) NOT NULL,
    seat varchar(10) NOT NULL,
    showtime_id int4 NOT NULL,
    created_at timestamp NULL,
    CONSTRAINT bookings_pkey PRIMARY KEY (id),
    CONSTRAINT uq_booking_showtime_seat UNIQUE (showtime_id, seat),
    CONSTRAINT bookings_showtime_id_fkey FOREIGN KEY (showtime_id) REFERENCES public.showtimes(id)
);
