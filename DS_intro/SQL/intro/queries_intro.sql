-- Show All Movies in the table general_info:
SELECT * FROM general_info

-- Display Specific Columns for All Movies (title, date_published, duration)
SELECT title, date_published, duration FROM general_info

-- FILTERING AND SORTING
-- find all movies with the genre "Action"
SELECT * FROM genre_info WHERE genre = 'Action';

-- list the top 5 longest movies
SELECT * FROM general_info ORDER BY duration DESC LIMIT 5;

-- find all movies produced in the us
SELECT * FROM general_info WHERE country = 'USA';

-- count the total number of movies
SELECT COUNT(*) FROM general_info;

-- find average, minimun, maximum duration of movies
SELECT AVG(duration), MIN(duration), MAX(duration) from general_info;

-- cout the number of movies by language
SELECT language, COUNT(*) AS movie_count
FROM general_info 
GROUP BY language
ORDER BY movie_count DESC;

-- find the average vote for movies by country
-- rating_info table thas column: 'imdb_title_id', "avg_vote"
-- general_info has column: 'conutry', 'title', 'imdb_title_id'
SELECT g.country, AVG(r.avg_vote) as avergae_vote
from general_info g 
INNER JOIN rating_info r
ON g.imdb_title_id = r.imdb_title_id
GROUP BY country
ORDER BY average_vote DESC;

-- NULL VALUES
-- find imdb_title_id missing metascore
-- metascore from rating_info
SELECT imdb_title_id, metascore FROM rating_info where metascore IS NULL;

-- find movies with empty description
SELECT imdb_title_id, description 
FROM general_info 
WHERE description = '';

-- JOIN ACROSS TABLES
-- find all movies along with their ratings
-- 'title' from ginfo, 'avg_rate' and votes froom rinfo
SELECT g.title, r.avg_vote, r.votes
from general_info g
INNER JOIN rating_info r
ON g.imdb_title_id = r.imdb_title_id
order by r.avg_vote desc;

-- find all genres of a movie (for just one movie)
SELECT g.title, l.genre
from general_info g
INNER JOIN genre_info l
ON g.imdb_title_id = l.imdb_title_id
WHERE g.title = 'Nome in codice: Nina';

-- find all movies along with their ratings and their genres
-- title from general_info, 'avg_vote
SELECT g.title, r.avg_vote, l.genre
FROM general_info g
JOIN rating_info r 
ON r.imdb_title_id = g.imdb_title_id
JOIN genre_info l 
ON g.imdb_title_id = l.imdb_title_id;

-- all genres along with their means ratings
SELECT l.genre, AVG(r.avg_vote) as average_rating
FROM genre_info l
INNER JOIN general_info g 
ON l.imdb_title_id = g.imdb_title_id
INNER JOIN rating_info r
ON r.imdb_title_id = g.imdb_title_id
GROUP BY l.genre
ORDER BY average_rating DESC;

-- list all actors and movies they appeared in 
-- 'title', 'imdb...' from general_info
-- 'name', 'id' from people_info
-- 'people_info_id','general_info_id' from casting_info
SELECT g.title, p.name 
FROM casting_info c
INNER JOIN general_info g
ON c.general_info_if = g.imdb_title_id
INNER JOIN people_info p 
ON c.people_info_id = p.id
ORDER BY p.name, g.title;


-- GROUPING AND AGGREGATION

-- count movies by genre
-- id, imdb.., genre in genre_info
SELECT l.genre, COUNT(DISTINCT l.imdb_title_id) AS movie_count  -- select the column of interest, and count on each id
FROM genre_info l                                               
GROUP BY l.genre                                                -- group by genre
ORDER BY movie_count DESC;

-- count movies by known production company
-- production_company from general_info
SELECT production_company, COUNT(distinct imdb_title_id) AS qty
from general_info
group by production_company
order by qty desc;

-- count the number of movies by each actor
-- casting info : id, people_info_id, general_info_id
-- general_info : 'title', 
-- people_info : name
-- i need the column name (actor name) from p; 
-- i count the number using general_info_id from c and 
-- save the grouped amount in movie_qty;
-- i gorup by name o the actor
SELECT p.name, COUNT(distinct c.general_info_id) as movie_qty
FROM casting_info c
INNER JOIN people_info p 
ON c.people_info_id = p.id 
GROUP BY p.name
ORDER BY movie_qty DESC;

-- find the average rate by director
-- 'name' in people info
-- 'director' is one of the possible value in casting_info.role
-- avg_vote is column of rating_info
-- join: rating_info.imdb_id <-> casting_info.general_info_id
SELECT p.name, AVG(r.avg_vote) as avgr
FROM casting_info c
INNER JOIN people_info p ON c.people_info_id = p.id
INNER JOIN rating_info r ON c.general_info_id = r.imdb_title_id
WHERE c.role = 'director'
GROUP BY p.name
ORDER BY avgr desc;