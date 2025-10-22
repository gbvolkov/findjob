from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def _truncate(text: str) -> str:
    return (text or "").strip()


def format_salary(salary: Optional[Dict[str, Any]]) -> str:
    if not isinstance(salary, dict):
        return "не указана"
    minimum = salary.get("min")
    maximum = salary.get("max")
    currency = salary.get("currency") or "RUB"
    if minimum is None and maximum is None:
        return "не указана"
    if minimum is None:
        return f"до {maximum} {currency}"
    if maximum is None:
        return f"от {minimum} {currency}"
    if minimum == maximum:
        return f"{minimum} {currency}"
    return f"{minimum}-{maximum} {currency}"


def _non_empty(parts: Iterable[str]) -> List[str]:
    return [part for part in parts if part]


def format_vacancy(job: Dict[str, Any], index: int | None = None) -> str:
    title = _truncate(job.get("title") or "Вакансия")
    company = _truncate(job.get("company"))
    location = _truncate(job.get("location"))
    salary = format_salary(job.get("salary"))
    experience = _truncate(job.get("experience"))
    source = _truncate(job.get("source"))
    published_at = _truncate(job.get("published_at"))
    url = _truncate(job.get("url"))
    description = _truncate(job.get("description"))
    match_score = job.get("match_score")
    rank_score = job.get("rank_score")

    skills = [
        skill.strip()
        for skill in job.get("skills") or []
        if isinstance(skill, str) and skill.strip()
    ]

    header_prefix = f"{index}. " if index is not None else ""
    header = f"{header_prefix}{title}"
    if company:
        header += f" — {company}"

    parts: List[str] = [header]
    details: List[str] = []
    if location:
        details.append(f"Локация: {location}")
    if salary:
        details.append(f"Зарплата: {salary}")
    if experience:
        details.append(f"Опыт: {experience}")
    if match_score is not None:
        details.append(f"Оценка соответствия: {match_score}")
    if rank_score is not None and match_score is None:
        details.append(f"Ранг: {rank_score}")
    if published_at:
        details.append(f"Опубликовано: {published_at}")
    if source:
        details.append(f"Источник: {source}")
    if skills:
        details.append("Навыки: " + ", ".join(skills[:12]))
    if description:
        details.append("Описание: " + description)
    if url:
        details.append(f"Ссылка: {url}")

    return "\n".join(parts + _non_empty(details))


def format_agent_response(payload: Dict[str, Any]) -> str:
    summary = _truncate(payload.get("summary"))
    vacancies = [vac for vac in payload.get("vacancies") or [] if isinstance(vac, dict)]

    blocks: List[str] = []
    if summary:
        blocks.append(summary)

    if vacancies:
        vacancy_blocks = [format_vacancy(vacancy, index=i + 1) for i, vacancy in enumerate(vacancies)]
        blocks.append("\n\n".join(vacancy_blocks))

    formatted = "\n\n".join(blocks).strip()
    return formatted or payload.get("summary") or ""
