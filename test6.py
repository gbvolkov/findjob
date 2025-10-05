from job_agent.rerankers import VacancyEmbeddingReranker
import json

if __name__ == "__main__":  # pragma: no cover - manual smoke test helper
    sample_vacancies = [
        {
            "id": "1",
            "title": "Senior Python Developer",
            "company": "Tech Corp",
            "location": "Remote",
            "description": "Develop backend services with FastAPI and PostgreSQL.",
            "skills": ["Python", "FastAPI", "PostgreSQL"],
            "experience": "3-5 years",
            "salary": {"min": 250000, "max": 300000, "currency": "RUB"},
            "url": "https://example.com/jobs/1",
        },
        {
            "id": "2",
            "title": "Data Scientist",
            "company": "AI Labs",
            "location": "Moscow",
            "description": "Build ML models for customer analytics and reporting.",
            "skills": ["Python", "Pandas", "Machine Learning"],
            "experience": "2+ years",
            "salary": {"min": 220000, "max": 280000, "currency": "RUB"},
            "url": "https://example.com/jobs/2",
        },
    ]

    sample_features = {
        "positions": ["Python Developer", "Machine Learning Engineer"],
        "skills": ["Python", "FastAPI", "ML"],
        "locations": ["Remote", "Moscow"],
    }

    reranker = VacancyEmbeddingReranker()
    results = reranker.rerank(
        sample_vacancies,
        resume_text="Experienced Python developer with ML background and FastAPI expertise.",
        features=sample_features,
        use_cross_encoder=False,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))