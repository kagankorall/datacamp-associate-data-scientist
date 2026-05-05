-- Run this code to view the data in students
SELECT * 
FROM students;

-- Project Code
SELECT s.stay, 
	COUNT(*) as count_int,
	ROUND(AVG(s.todep),2) as average_phq,
	ROUND(AVG(s.tosc),2) as average_scs,
	ROUND(AVG(s.toas),2) as average_as
FROM students s
WHERE s.stay IS NOT NULL AND s.inter_dom = 'Inter'
GROUP BY s.stay
ORDER BY s.stay DESC