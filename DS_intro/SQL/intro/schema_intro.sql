Table "casting_info" {
  "id" integer [pk, not null]
  "people_info_id" integer [not null]
  "general_info_id" text [not null]
  "role" text [not null]

  Indexes {
    (people_info_id, general_info_id, role) [unique, name: "uniq_cast"]
  }
}

Table "general_info" {
  "imdb_title_id" text [pk, not null]
  "title" text
  "original_title" text
  "date_published" date
  "duration" integer
  "country" text
  "language" text
  "production_company" text
  "description" text
}

Table "genre_info" {
  "id" integer [pk, not null]
  "imdb_title_id" text [not null]
  "genre" text
}

Table "people_info" {
  "id" integer [pk, not null]
  "name" text [unique, not null]
}

Table "rating_info" {
  "imdb_title_id" text [pk, not null]
  "avg_vote" real
  "votes" integer
  "metascore" integer
  "reviews_from_users" integer
  "reviews_from_critics" integer
}

Ref "casting_info_general_info_fk":"general_info"."imdb_title_id" < "casting_info"."general_info_id"

Ref "casting_info_people_info_fk":"people_info"."id" < "casting_info"."people_info_id"

Ref "genre_info_imdb_title_id_fkey":"general_info"."imdb_title_id" < "genre_info"."imdb_title_id"

Ref "rating_info_imdb_title_id_fkey":"general_info"."imdb_title_id" < "rating_info"."imdb_title_id"
