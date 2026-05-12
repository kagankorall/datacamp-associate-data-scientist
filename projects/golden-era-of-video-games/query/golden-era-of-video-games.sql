-- best_selling_game
WITH best_selling_games AS (
SELECT gs.name, gs.platform, gs.publisher, gs.developer, gs.games_sold, gs.year
FROM public.game_sales gs
ORDER BY gs.games_sold DESC
LIMIT 10)
SELECT * FROM best_selling_games

-- critics_top_ten_years
WITH years AS (
SELECT gs.year, COUNT(gs.name) as GameCount
FROM public.game_sales gs
JOIN public.reviews r ON r.name = gs.name
GROUP BY gs.year
HAVING COUNT(gs.name) >= 4),
critics_top_ten_years AS (
	SELECT gs.year, COUNT(gs.name) as num_games, ROUND(AVG(r.critic_score),2) as avg_critic_score
	FROM public.game_sales gs
	JOIN public.reviews r ON r.name = gs.name
	JOIN years y ON y.year = gs.year
	GROUP BY gs.year
	ORDER BY ROUND(AVG(r.critic_score),2) DESC)
SELECT *
FROM critics_top_ten_years
LIMIT 10

-- golden_years
WITH golden_years AS (
	SELECT u.year, u.num_games, u.avg_user_score, c.avg_critic_score, 
	(c.avg_critic_score - u.avg_user_score) as diff
	FROM public.users_avg_year_rating u
	LEFT JOIN public.critics_avg_year_rating c ON c.year = u.year
	WHERE u.avg_user_score > 9 OR c.avg_critic_score > 9
	ORDER BY u.year
)
SELECT *
FROM golden_years