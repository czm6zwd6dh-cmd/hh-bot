import os
import sqlite3
import asyncio
import logging
import re
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from openai import AsyncOpenAI
import httpx
import aiohttp
import random
from fastapi import FastAPI, Request
import uvicorn
import yaml
import urllib.parse

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
CHAT_ID = os.getenv("CHAT_ID")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан")

MAX_PUSH_PER_CYCLE = 5
SEEN_VACANCIES_TTL_DAYS = 7
MAX_SEARCH_TIME = 300
SEARCH_COOLDOWN_SECONDS = 30

deepseek_available = False
client = None
search_lock = asyncio.Lock()
last_search_time = 0

PROFILE_YAML_PATH = "profile.yaml"

DEFAULT_PROFILE = {
    "candidate": {
        "name": "Зинченко Виктор Александрович",
        "age": 42,
        "city": "Волгоград",
        "relocation_ready": ["Астана", "Баку", "Казань", "Минск", "Нижний Новгород", "Уфа"],
        "business_trips": True,
        "desired_positions": ["коммерческий директор", "руководитель отдела продаж", "директор по продажам", "директор филиала"],
        "specialization": "Нефтепродукты, ГСМ, B2B, опт",
        "employment_type": "Полная занятость",
        "format": "Офис/производство",
        "salary_min": 200000,
        "experience_years": 20,
        "management_years": 11,
        "key_skills": [
            "управление отделом продаж ГСМ",
            "закупки СПбМТСБ",
            "контракты с НПЗ",
            "B2B продажи",
            "нефтебаза",
            "перевалка нефтепродуктов",
            "ж/д и автотранспорт",
            "логистика",
            "дебиторская задолженность",
            "1С: Управление предприятием"
        ]
    },
    "filters": {
        "salary_min": 200000,
        "cities": {"Волгоград": "24", "Москва": "1", "Казань": "88", "Нижний Новгород": "66",
                   "Уфа": "99", "Астана": "160", "Баку": "100", "Минск": "1002"},
        "keywords": ["коммерческий директор", "руководитель отдела продаж", "директор по продажам",
                     "директор филиала", "руководитель направления"],
        "industry_keywords": ["нефтепродукты", "ГСМ", "топливо", "бензин", "дизель", "мазут",
                              "нефть", "нефтетрейдинг", "нефтебаза", "АЗС", "СПбМТСБ", "НПЗ",
                              "нефтепереработка", "моторное топливо", "нефтяной", "oil", "petroleum", "fuel"],
        "exclude_words": ["стажёр", "intern", "junior", "1С", "QA", "тестировщик", "розница",
                          "продавец-консультант", "FMCG", "продукты питания", "одежда", "обувь",
                          "строительные материалы", "электроника", "IT", "финансы", "банки",
                          "страхование", "недвижимость", "маркетинг", "реклама", "HR", "медицина", "образование"],
        "company_blacklist": [],
        "title_blacklist": []
    },
    "scoring": {
        "weights": {
            "role_fit": 0.30,
            "industry_match": 0.25,
            "salary_match": 0.15,
            "location_match": 0.10,
            "experience_match": 0.10,
            "skills_match": 0.10
        },
        "min_score": 60,
        "strict_requirements": {
            "min_salary": 200000,
            "required_industries": ["нефтепродукты", "ГСМ", "топливо", "нефть", "oil", "petroleum", "fuel"],
            "exclude_any": ["стажёр", "intern", "junior"]
        }
    },
    "notifications": {
        "auto_push": True,
        "max_per_cycle": 5,
        "digest_mode": False,
        "quiet_hours": {"start": 23, "end": 8}
    }
}

def load_profile() -> dict:
    if os.path.exists(PROFILE_YAML_PATH):
        try:
            with open(PROFILE_YAML_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки profile.yaml: {e}, использую defaults")
    with open(PROFILE_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(DEFAULT_PROFILE, f, allow_unicode=True, sort_keys=False)
    return DEFAULT_PROFILE

def save_profile(profile: dict):
    with open(PROFILE_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(profile, f, allow_unicode=True, sort_keys=False)

PROFILE = load_profile()

def reload_profile():
    global PROFILE
    PROFILE = load_profile()
    logger.info("Profile reloaded from YAML")

def init_deepseek_client():
    global client
    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY не задан")
        return
    try:
        http_client = httpx.AsyncClient(timeout=10.0, follow_redirects=True)
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1", http_client=http_client)
        logger.info("DeepSeek клиент создан")
    except Exception as e:
        logger.warning(f"DeepSeek init error: {e}")
        client = None

async def check_deepseek_connection():
    global deepseek_available, client
    if not client or not DEEPSEEK_API_KEY:
        deepseek_available = False
        logger.warning("DeepSeek: клиент не создан или API ключ отсутствует")
        return False
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "Привет"}],
                max_tokens=5
            ),
            timeout=15.0
        )
        if response and response.choices:
            deepseek_available = True
            logger.info("DeepSeek API доступен")
            return True
        else:
            deepseek_available = False
            logger.warning("DeepSeek: пустой ответ от API")
            return False
    except asyncio.TimeoutError:
        deepseek_available = False
        logger.error("DeepSeek: таймаут при проверке подключения")
        return False
    except Exception as e:
        deepseek_available = False
        logger.error(f"DeepSeek: ошибка проверки подключения: {e}")
        return False

init_deepseek_client()

DB_PATH = "vacancies.db"

def _init_db_sync():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS sent_vacancies (
        id TEXT PRIMARY KEY, title TEXT, company TEXT, score REAL, sent_at TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS seen_vacancies (
        id TEXT PRIMARY KEY, title TEXT, company TEXT, score REAL, reason TEXT, seen_at TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS search_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, found_count INTEGER, sent_count INTEGER, 
        avg_score REAL, searched_at TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT, vacancy_id TEXT, title TEXT, company TEXT,
        score REAL, status TEXT DEFAULT 'new', cover_letter TEXT, notes TEXT, created_at TIMESTAMP
    )""")
    c.execute("""CREATE INDEX IF NOT EXISTS idx_seen_at ON seen_vacancies(seen_at)""")
    c.execute("""CREATE INDEX IF NOT EXISTS idx_sent_at ON sent_vacancies(sent_at)""")
    c.execute("""CREATE TABLE IF NOT EXISTS oauth_tokens (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        access_token TEXT,
        refresh_token TEXT,
        expires_at TIMESTAMP
    )""")
    conn.commit()
    conn.close()

def _is_vacancy_sent_sync(vacancy_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT 1 FROM sent_vacancies WHERE id = ?", (vacancy_id,))
    result = c.fetchone() is not None
    conn.close()
    return result

def _is_vacancy_seen_sync(vacancy_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cutoff = datetime.now() - timedelta(days=SEEN_VACANCIES_TTL_DAYS)
    c.execute("SELECT 1 FROM seen_vacancies WHERE id = ? AND seen_at > ?", (vacancy_id, cutoff))
    result = c.fetchone() is not None
    conn.close()
    return result

def _mark_vacancy_sent_sync(vacancy_id, title, company, score=0):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO sent_vacancies VALUES (?, ?, ?, ?, ?)",
              (vacancy_id, title, company, score, datetime.now()))
    conn.commit()
    conn.close()

def _mark_vacancy_seen_sync(vacancy_id, title, company, score, reason=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO seen_vacancies VALUES (?, ?, ?, ?, ?, ?)",
              (vacancy_id, title, company, score, reason, datetime.now()))
    conn.commit()
    conn.close()

def _log_search_sync(found, sent, avg_score):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO search_log (found_count, sent_count, avg_score, searched_at) VALUES (?, ?, ?, ?)",
              (found, sent, avg_score, datetime.now()))
    conn.commit()
    conn.close()

def _get_stats_sync():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM sent_vacancies")
    total_sent = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM seen_vacancies")
    total_seen = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM search_log")
    total_searches = c.fetchone()[0]
    c.execute("SELECT found_count, sent_count, avg_score, searched_at FROM search_log ORDER BY searched_at DESC LIMIT 5")
    recent = c.fetchall()
    c.execute("SELECT status, COUNT(*) FROM applications GROUP BY status")
    app_stats = dict(c.fetchall())
    conn.close()
    return total_sent, total_seen, total_searches, recent, app_stats

def _get_top_vacancies_sync(limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title, company, score, sent_at FROM sent_vacancies ORDER BY score DESC, sent_at DESC LIMIT ?", (limit,))
    result = c.fetchall()
    conn.close()
    return result

def _add_application_sync(vacancy_id, title, company, score):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO applications (vacancy_id, title, company, score, created_at) VALUES (?, ?, ?, ?, ?)",
              (vacancy_id, title, company, score, datetime.now()))
    conn.commit()
    conn.close()

def _update_application_status_sync(vacancy_id, status, notes=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE applications SET status = ?, notes = ? WHERE vacancy_id = ?",
              (status, notes, vacancy_id))
    conn.commit()
    conn.close()

def _get_applications_sync(status=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if status:
        c.execute("SELECT * FROM applications WHERE status = ? ORDER BY created_at DESC", (status,))
    else:
        c.execute("SELECT * FROM applications ORDER BY created_at DESC")
    result = c.fetchall()
    conn.close()
    return result

def _cleanup_sync(days=30):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cutoff = datetime.now() - timedelta(days=days)
    c.execute("DELETE FROM sent_vacancies WHERE sent_at < ?", (cutoff,))
    deleted_sent = c.rowcount
    c.execute("DELETE FROM seen_vacancies WHERE seen_at < ?", (cutoff,))
    deleted_seen = c.rowcount
    conn.commit()
    conn.close()
    return deleted_sent + deleted_seen

async def init_db():
    await asyncio.to_thread(_init_db_sync)

async def is_vacancy_sent(vacancy_id):
    return await asyncio.to_thread(_is_vacancy_sent_sync, vacancy_id)

async def is_vacancy_seen(vacancy_id):
    return await asyncio.to_thread(_is_vacancy_seen_sync, vacancy_id)

async def mark_vacancy_sent(vacancy_id, title, company, score=0):
    await asyncio.to_thread(_mark_vacancy_sent_sync, vacancy_id, title, company, score)

async def mark_vacancy_seen(vacancy_id, title, company, score, reason=""):
    await asyncio.to_thread(_mark_vacancy_seen_sync, vacancy_id, title, company, score, reason)

async def log_search(found, sent, avg_score):
    await asyncio.to_thread(_log_search_sync, found, sent, avg_score)

async def get_stats():
    return await asyncio.to_thread(_get_stats_sync)

async def get_top_vacancies(limit=10):
    return await asyncio.to_thread(_get_top_vacancies_sync, limit)

async def add_application(vacancy_id, title, company, score):
    await asyncio.to_thread(_add_application_sync, vacancy_id, title, company, score)

async def update_application_status(vacancy_id, status, notes=""):
    await asyncio.to_thread(_update_application_status_sync, vacancy_id, status, notes)

async def get_applications(status=None):
    return await asyncio.to_thread(_get_applications_sync, status)

async def cleanup_old_vacancies(days=30):
    return await asyncio.to_thread(_cleanup_sync, days)

# ========== SCORING SYSTEM ==========
@dataclass
class VacancyScore:
    total: float
    role_fit: float
    industry_match: float
    salary_match: float
    location_match: float
    experience_match: float
    skills_match: float
    verdict: str
    reasoning: str

def calculate_keyword_score(text: str, keywords: List[str]) -> float:
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return min(matches / max(len(keywords) * 0.3, 1), 1.0) * 10

def calculate_salary_score(vacancy_salary: dict) -> float:
    if not vacancy_salary:
        return 5.0
    salary_from = vacancy_salary.get("from")
    salary_to = vacancy_salary.get("to")
    min_required = PROFILE["scoring"]["strict_requirements"]["min_salary"]
    if salary_from and salary_from >= min_required:
        return 10.0
    elif salary_to and salary_to >= min_required:
        return 8.0
    elif salary_from and salary_from >= min_required * 0.8:
        return 6.0
    elif salary_to and salary_to >= min_required * 0.8:
        return 4.0
    return 2.0

def calculate_location_score(city: str) -> float:
    allowed_cities = list(PROFILE["filters"]["cities"].keys())
    if any(c.lower() in city.lower() for c in allowed_cities):
        return 10.0
    return 0.0

def calculate_experience_score(text: str) -> float:
    text_lower = text.lower()
    exp_keywords = ["опыт", "лет", "years", "senior", "руководитель", "директор", "manager"]
    score = sum(2 for kw in exp_keywords if kw in text_lower)
    if "6" in text or "5" in text or "10" in text or "20" in text:
        score += 3
    return min(score, 10)

def calculate_skills_score(text: str) -> float:
    key_skills = PROFILE["candidate"]["key_skills"]
    return calculate_keyword_score(text, key_skills)

def score_vacancy_heuristic(vacancy: dict) -> VacancyScore:
    text = (vacancy.get("name", "") + " " + vacancy.get("description", "") +
            " " + vacancy.get("snippet", {}).get("requirement", "")).lower()
    weights = PROFILE["scoring"]["weights"]
    role_fit = calculate_keyword_score(text, PROFILE["filters"]["keywords"])
    industry_match = calculate_keyword_score(text, PROFILE["filters"]["industry_keywords"])
    salary_match = calculate_salary_score(vacancy.get("salary"))
    location_match = calculate_location_score(vacancy.get("area", {}).get("name", ""))
    experience_match = calculate_experience_score(text)
    skills_match = calculate_skills_score(text)
    total = (role_fit * weights["role_fit"] +
             industry_match * weights["industry_match"] +
             salary_match * weights["salary_match"] +
             location_match * weights["location_match"] +
             experience_match * weights["experience_match"] +
             skills_match * weights["skills_match"])
    strict = PROFILE["scoring"]["strict_requirements"]
    for ex in strict["exclude_any"]:
        if ex.lower() in text:
            total = 0
            verdict = "SKIP"
            reasoning = f"Стоп-слово: {ex}"
            break
    else:
        if total >= 80:
            verdict = "STRONG_MATCH"
            reasoning = "Отличное соответствие"
        elif total >= PROFILE["scoring"]["min_score"]:
            verdict = "MATCH"
            reasoning = "Хорошее соответствие"
        elif total >= 40:
            verdict = "WEAK_MATCH"
            reasoning = "Слабое соответствие"
        else:
            verdict = "SKIP"
            reasoning = "Низкий скор"
    return VacancyScore(
        total=round(total, 1),
        role_fit=round(role_fit, 1),
        industry_match=round(industry_match, 1),
        salary_match=round(salary_match, 1),
        location_match=round(location_match, 1),
        experience_match=round(experience_match, 1),
        skills_match=round(skills_match, 1),
        verdict=verdict,
        reasoning=reasoning
    )

# ========== AI SCORING PROMPT ==========
SCORING_PROMPT = """Ты — профессиональный рекрутер-аналитик с 20-летним опытом в подборе топ-менеджеров в нефтяную отрасль.

Профиль кандидата:
{profile}

Оцени вакансию по шкале 0-100 по 6 критериям:
1. role_fit (0-10) — соответствие должности (коммерческий директор, директор продаж и т.д.)
2. industry_match (0-10) — соответствие индустрии (нефтепродукты, ГСМ, топливо)
3. salary_match (0-10) — зарплата от 200K
4. location_match (0-10) — город из списка: Волгоград, Москва, Казань, НН, Уфа, Астана, Баку, Минск
5. experience_match (0-10) — требуемый опыт управления, B2B, НПЗ
6. skills_match (0-10) — ключевые навыки: СПбМТСБ, нефтебаза, логистика, 1С

Важно: если в тексте есть слова "стажёр", "intern", "junior", "1С" (как программист), "QA" — total = 0.

Верни СТРОГО JSON без markdown:
{"total": 75, "role_fit": 8, "industry_match": 9, "salary_match": 7, "location_match": 10, "experience_match": 8, "skills_match": 7, "verdict": "MATCH", "reasoning": "Краткое обоснование на русском"}"""

async def ask_deepseek_scoring(vacancy: dict) -> Optional[VacancyScore]:
    global deepseek_available, client
    if not deepseek_available:
        logger.info("DeepSeek недоступен, пробуем переподключиться...")
        ok = await check_deepseek_connection()
        if not ok:
            return None
    if not client:
        return None
    profile_text = yaml.dump(PROFILE["candidate"], allow_unicode=True)
    vacancy_text = "Название: " + vacancy.get("name", "") + "\n"
    vacancy_text += "Компания: " + vacancy.get("employer", {}).get("name", "") + "\n"
    vacancy_text += "Город: " + vacancy.get("area", {}).get("name", "") + "\n"
    salary = vacancy.get("salary", {})
    vacancy_text += "Зарплата: " + str(salary.get("from", "")) + " - " + str(salary.get("to", "")) + " " + str(salary.get("currency", "")) + "\n"
    vacancy_text += "Требования: " + vacancy.get("snippet", {}).get("requirement", "") + "\n"
    vacancy_text += "Обязанности: " + vacancy.get("snippet", {}).get("responsibility", "") + "\n"
    desc = vacancy.get("description", "")
    vacancy_text += "Описание: " + desc[:1500]
    prompt = SCORING_PROMPT.format(profile=profile_text) + "\n\n=== ВАКАНСИЯ ===\n" + vacancy_text
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            ),
            timeout=20.0
        )
        answer = response.choices[0].message.content.strip()
        answer = re.sub(r"```json\s*", "", answer)
        answer = re.sub(r"```\s*", "", answer)
        data = json.loads(answer)
        return VacancyScore(
            total=float(data.get("total", 0)),
            role_fit=float(data.get("role_fit", 0)),
            industry_match=float(data.get("industry_match", 0)),
            salary_match=float(data.get("salary_match", 0)),
            location_match=float(data.get("location_match", 0)),
            experience_match=float(data.get("experience_match", 0)),
            skills_match=float(data.get("skills_match", 0)),
            verdict=data.get("verdict", "SKIP"),
            reasoning=data.get("reasoning", "")
        )
    except asyncio.TimeoutError:
        logger.error("DeepSeek: таймаут скоринга")
        deepseek_available = False
        return None
    except Exception as e:
        logger.error(f"DeepSeek scoring error: {e}")
        deepseek_available = False
        return None

# ========== COVER LETTER GENERATION ==========
COVER_LETTER_PROMPT = """Ты — профессиональный карьерный консультант. Напиши сопроводительное письмо от имени кандидата для отклика на вакансию.

Профиль кандидата:
{profile}

Вакансия:
{vacancy}

Требования:
1. Письмо на русском языке, деловой стиль
2. 150-200 слов
3. Акцент на релевантный опыт (управление продажами ГСМ, СПбМТСБ, нефтебаза)
4. Упомяни готовность к переезду/командировкам если релевантно
5. Заверши призывом к действию (собеседование)
6. Без шаблонных фраз типа "уважаемый работодатель"

Верни ТОЛЬКО текст письма, без JSON, без markdown."""

async def generate_cover_letter(vacancy: dict) -> Optional[str]:
    global deepseek_available, client
    if not deepseek_available:
        logger.info("DeepSeek недоступен, пробуем переподключиться для генерации письма...")
        ok = await check_deepseek_connection()
        if not ok:
            return None
    if not client:
        return None
    profile_text = yaml.dump(PROFILE["candidate"], allow_unicode=True)
    vacancy_text = vacancy.get("name", "") + " в " + vacancy.get("employer", {}).get("name", "") + ". " + vacancy.get("description", "")[:1000]
    prompt = COVER_LETTER_PROMPT.format(profile=profile_text, vacancy=vacancy_text)
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            ),
            timeout=20.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Cover letter error: {e}")
        deepseek_available = False
        return None

# ========== RAG Q&A ==========
RAG_PROMPT = """Ты — карьерный советник. Ответь на вопрос пользователя о вакансии, используя профиль кандидата и описание вакансии.

Профиль кандидата:
{profile}

Вакансия: {vacancy_name} в {company}
Описание: {description}

Вопрос: {question}

Дай краткий, конкретный ответ на русском. Если вопрос о том, подходит ли кандидат — объясни почему да или нет с цитатами из профиля."""

async def ask_rag_about_vacancy(vacancy: dict, question: str) -> Optional[str]:
    global deepseek_available, client
    if not deepseek_available:
        logger.info("DeepSeek недоступен, пробуем переподключиться для RAG...")
        ok = await check_deepseek_connection()
        if not ok:
            return "🤖 AI временно недоступен. Попробуйте позже."
    if not client:
        return "🤖 AI не инициализирован."
    profile_text = yaml.dump(PROFILE["candidate"], allow_unicode=True)
    prompt = RAG_PROMPT.format(
        profile=profile_text,
        vacancy_name=vacancy.get("name", ""),
        company=vacancy.get("employer", {}).get("name", ""),
        description=vacancy.get("description", "")[:1500],
        question=question
    )
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            ),
            timeout=15.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"RAG error: {e}")
        deepseek_available = False
        return "⚠️ Ошибка при обработке вопроса"

# ========== SCRAPING ==========
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

class RateLimiter:
    def __init__(self, min_delay=2.0, max_delay=5.0):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_request_time = 0
        self.lock = asyncio.Lock()
        self.consecutive_429s = 0
        self.cooldown_until = 0

    async def wait(self):
        async with self.lock:
            now = asyncio.get_event_loop().time()
            if now < self.cooldown_until:
                wait_time = self.cooldown_until - now
                logger.info(f"Rate limiter: cooldown ещё {wait_time:.1f}с")
                await asyncio.sleep(wait_time)
                now = asyncio.get_event_loop().time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_delay:
                wait_time = self.min_delay - elapsed + random.uniform(0, 1)
                await asyncio.sleep(wait_time)
            self.last_request_time = asyncio.get_event_loop().time()

    def on_429(self):
        self.consecutive_429s += 1
        cooldown = min(10 * (2 ** self.consecutive_429s), 120)
        jitter = random.uniform(5, 15)
        self.cooldown_until = asyncio.get_event_loop().time() + cooldown + jitter
        logger.warning(f"Rate limiter: 429 #{self.consecutive_429s}, cooldown {cooldown + jitter:.0f}с")

    def on_success(self):
        if self.consecutive_429s > 0:
            self.consecutive_429s = max(0, self.consecutive_429s - 1)

hh_rate_limiter = RateLimiter(min_delay=2.0, max_delay=5.0)

# ========== HH.RU OAUTH CLIENT ==========
HH_CLIENT_ID = os.getenv("HH_CLIENT_ID")
HH_CLIENT_SECRET = os.getenv("HH_CLIENT_SECRET")
HH_REDIRECT_URI = os.getenv("HH_REDIRECT_URI", "")
HH_ACCESS_TOKEN = os.getenv("HH_ACCESS_TOKEN")
HH_REFRESH_TOKEN = os.getenv("HH_REFRESH_TOKEN")

class HHOAuthClient:
    """Клиент OAuth 2.0 для HH.ru с автообновлением токена."""

    TOKEN_URL = "https://hh.ru/oauth/token"
    AUTH_URL = "https://hh.ru/oauth/authorize"

    def __init__(self):
        self.access_token = HH_ACCESS_TOKEN
        self.refresh_token = HH_REFRESH_TOKEN
        self.client_id = HH_CLIENT_ID
        self.client_secret = HH_CLIENT_SECRET
        self.token_expires_at = None
        self.load_tokens()

    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def has_token(self) -> bool:
        return bool(self.access_token)

    def get_auth_url(self) -> str:
        """URL для авторизации пользователя (открыть в браузере)."""
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": HH_REDIRECT_URI,
        }
        return f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_code(self, code: str) -> dict:
        """Обменять authorization code на access + refresh токены."""
        data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": HH_REDIRECT_URI,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.TOKEN_URL, data=data) as resp:
                    result = await resp.json()
                    logger.info(f"HH OAuth: ответ API status={resp.status}, keys={list(result.keys())}")
                    if resp.status == 200:
                        self.access_token = result.get("access_token")
                        self.refresh_token = result.get("refresh_token")
                        expires_in = result.get("expires_in", 1209600)
                        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                        logger.info(f"HH OAuth: токен получен, access={self.access_token[:15]}..." if self.access_token else "HH OAuth: токен пустой!")
                        # Сохраняем синхронно, т.к. async может не успеть
                        saved = self.save_tokens_sync()
                        if saved:
                            logger.info("HH OAuth: токены сохранены успешно")
                        else:
                            logger.error("HH OAuth: НЕ УДАЛОСЬ сохранить токены")
                        return {
                            "success": True,
                            "access_token": self.access_token,
                            "refresh_token": self.refresh_token,
                            "expires_in": expires_in,
                        }
                    else:
                        logger.error(f"HH OAuth: ошибка обмена code: {result}")
                        return {"success": False, "error": result}
        except Exception as e:
            logger.error(f"HH OAuth: исключение при exchange_code: {e}")
            return {"success": False, "error": str(e)}

    async def refresh_access_token(self) -> bool:
        """Обновить access token через refresh token."""
        if not self.refresh_token:
            logger.warning("HH OAuth: нет refresh_token для обновления")
            return False

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.TOKEN_URL, data=data, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    result = await resp.json()
                    if resp.status == 200:
                        self.access_token = result.get("access_token")
                        self.refresh_token = result.get("refresh_token")
                        expires_in = result.get("expires_in", 1209600)
                        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                        await self.save_tokens()
                        logger.info("HH OAuth: токен успешно обновлён")
                        return True
                    else:
                        logger.error(f"HH OAuth: ошибка обновления токена: {result}")
                        return False
        except Exception as e:
            logger.error(f"HH OAuth: ошибка при refresh: {e}")
            return False

    async def ensure_valid_token(self) -> bool:
        """Проверить и при необходимости обновить токен ДО запроса."""
        if not self.has_token():
            return False
        if self.token_expires_at and datetime.now() > self.token_expires_at - timedelta(hours=1):
            logger.info("HH OAuth: токен скоро истекает, обновляем превентивно...")
            return await self.refresh_access_token()
        return True

    def save_tokens_sync(self):
        """Синхронное сохранение токенов в БД."""
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS oauth_tokens (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                access_token TEXT,
                refresh_token TEXT,
                expires_at TIMESTAMP
            )""")
            c.execute("DELETE FROM oauth_tokens")
            c.execute("INSERT INTO oauth_tokens (id, access_token, refresh_token, expires_at) VALUES (?, ?, ?, ?)",
                      (1, self.access_token, self.refresh_token,
                       self.token_expires_at.isoformat() if self.token_expires_at else None))
            conn.commit()
            conn.close()
            logger.info("HH OAuth: токены сохранены в БД (sync)")
            return True
        except Exception as e:
            logger.error(f"HH OAuth: ошибка сохранения токенов: {e}")
            return False

    async def save_tokens(self):
        """Асинхронная обёртка для сохранения."""
        return await asyncio.to_thread(self.save_tokens_sync)

    def load_tokens(self):
        """Загрузить токены из SQLite БД при старте."""
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS oauth_tokens (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                access_token TEXT,
                refresh_token TEXT,
                expires_at TIMESTAMP
            )""")
            c.execute("SELECT access_token, refresh_token, expires_at FROM oauth_tokens WHERE id = 1")
            row = c.fetchone()
            conn.close()
            if row:
                self.access_token = row[0] or self.access_token
                self.refresh_token = row[1] or self.refresh_token
                if row[2]:
                    self.token_expires_at = datetime.fromisoformat(row[2])
                logger.info(f"HH OAuth: токены загружены из БД, access={self.access_token[:10]}..." if self.access_token else "HH OAuth: токены загружены но пустые")
            else:
                logger.info("HH OAuth: токены в БД не найдены")
        except Exception as e:
            logger.error(f"HH OAuth: ошибка загрузки токенов из БД: {e}")

    def get_headers(self) -> dict:
        """Заголовки для авторизованных запросов к API."""
        headers = {
            "User-Agent": "JobSearchBot/1.0 (telegram)",
            "Accept": "application/json",
            "Accept-Language": "ru-RU,ru;q=0.9",
        }
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

hh_oauth = HHOAuthClient()

# ─── ОБНОВЛЁННЫЙ ПОИСК ЧЕРЕЗ API (С АВТОРИЗАЦИЕЙ) ───

async def fetch_hh_api(session, city_id, keyword, per_page=20, page=0):
    """Поиск через официальное API HH.ru (с OAuth или без)."""
    encoded_kw = keyword.replace(" ", "+")
    url = f"https://api.hh.ru/vacancies?text={encoded_kw}&area={city_id}&per_page={per_page}&page={page}"

    if hh_oauth.has_token():
        await hh_oauth.ensure_valid_token()

    headers = hh_oauth.get_headers()

    await hh_rate_limiter.wait()
    logger.info(f"[HH-API] Запрос: {keyword} в area={city_id}")
    token_status = "ЕСТЬ" if hh_oauth.access_token else "НЕТ"
    logger.info(f"[HH-API] Токен: {token_status}")

    try:
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        async with session.get(url, headers=headers, timeout=timeout) as resp:
            logger.info(f"[HH-API] Статус {resp.status} для {city_id}/{keyword}")

            if resp.status == 200:
                data = await resp.json()
                items = data.get("items", [])
                hh_rate_limiter.on_success()
                logger.info(f"[HH-API] УСПЕХ: {len(items)} вакансий получено")
                return items
            elif resp.status == 401:
                logger.warning(f"[HH-API] Токен просрочен (401), пробуем обновить...")
                refreshed = await hh_oauth.refresh_access_token()
                if refreshed:
                    headers = hh_oauth.get_headers()
                    async with session.get(url, headers=headers, timeout=timeout) as retry_resp:
                        if retry_resp.status == 200:
                            data = await retry_resp.json()
                            items = data.get("items", [])
                            logger.info(f"[HH-API] УСПЕХ после обновления токена: {len(items)} вакансий")
                            return items
                logger.error("[HH-API] Не удалось обновить токен")
                return None
            elif resp.status == 403:
                logger.warning(f"[HH-API] БЛОКИРОВКА (403)")
                return None
            elif resp.status == 429:
                logger.warning(f"[HH-API] Rate limit (429)")
                hh_rate_limiter.on_429()
                return None
            else:
                logger.warning(f"[HH-API] Ошибка {resp.status}")
                return None
    except asyncio.TimeoutError:
        logger.warning(f"[HH-API] ТАЙМАУТ для {city_id}/{keyword}")
        return None
    except Exception as e:
        logger.error(f"[HH-API] ОШИБКА: {e}")
        return None

def hh_api_item_to_vacancy(item):
    """Конвертирует элемент из API HH.ru в формат вакансии бота."""
    salary = item.get("salary")
    salary_dict = None
    if salary:
        salary_dict = {
            "from": salary.get("from"),
            "to": salary.get("to"),
            "currency": salary.get("currency", "RUR")
        }

    employer = item.get("employer", {})
    area = item.get("area", {})

    vacancy = {
        "id": str(item.get("id", "")),
        "name": item.get("name", "Без названия"),
        "alternate_url": item.get("alternate_url", ""),
        "employer": {
            "name": employer.get("name", "Не указана")
        },
        "area": {
            "name": area.get("name", "Не указан")
        },
        "salary": salary_dict,
        "snippet": {
            "requirement": item.get("snippet", {}).get("requirement", "") or "",
            "responsibility": item.get("snippet", {}).get("responsibility", "") or ""
        },
        "description": (item.get("snippet", {}).get("requirement", "") or "") + " " + 
                      (item.get("snippet", {}).get("responsibility", "") or ""),
        "published_at": item.get("published_at", "")
    }
    return vacancy


async def fetch_rss(session, city_id, keyword, per_page=20, retries=3):
    """Прямой RSS-запрос к HH.ru (fallback)."""
    encoded_kw = keyword.replace(" ", "+")
    target_url = f"https://hh.ru/search/vacancy/rss?text={encoded_kw}&area={city_id}&items_on_page={per_page}"
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    for attempt in range(retries):
        await hh_rate_limiter.wait()
        logger.info(f"[RSS] Попытка {attempt+1}/{retries} для {city_id}/{keyword}")
        try:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            async with session.get(target_url, headers=headers, timeout=timeout) as resp:
                logger.info(f"[RSS] Статус {resp.status} для {city_id}/{keyword}")
                if resp.status == 200:
                    xml_text = await resp.text()
                    hh_rate_limiter.on_success()
                    logger.info(f"[RSS] УСПЕХ: {len(xml_text)} байт")
                    return parse_rss(xml_text)
                elif resp.status in (403, 429):
                    logger.warning(f"[RSS] БЛОКИРОВКА ({resp.status})")
                    hh_rate_limiter.on_429()
                    await asyncio.sleep(min(2 ** attempt + random.uniform(1, 3), 10))
                else:
                    logger.warning(f"[RSS] Ошибка {resp.status}")
                    await asyncio.sleep(min(2 ** attempt + random.uniform(1, 3), 10))
        except asyncio.TimeoutError:
            logger.warning(f"[RSS] ТАЙМАУТ для {city_id}/{keyword}")
            await asyncio.sleep(min(2 ** attempt + random.uniform(1, 3), 10))
        except Exception as e:
            logger.error(f"[RSS] ОШИБКА: {e} для {city_id}/{keyword}")
            await asyncio.sleep(min(2 ** attempt + random.uniform(1, 3), 10))
    logger.error(f"[RSS] ВСЕ ПОПЫТКИ ИСЧЕРПАНЫ для {city_id}/{keyword}")
    return []

async def fetch_html_fallback(session, city_id, keyword="коммерческий директор"):
    """Прямой HTML-запрос к HH.ru (fallback)."""
    encoded_kw = keyword.replace(" ", "+")
    target_url = f"https://hh.ru/search/vacancy?text={encoded_kw}&area={city_id}&items_on_page=20"
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9",
    }
    await hh_rate_limiter.wait()
    logger.info(f"[HTML] Запрос для {city_id}/{keyword}")
    try:
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        async with session.get(target_url, headers=headers, timeout=timeout) as resp:
            logger.info(f"[HTML] Статус {resp.status} для {city_id}/{keyword}")
            if resp.status == 200:
                html = await resp.text()
                hh_rate_limiter.on_success()
                logger.info(f"[HTML] УСПЕХ: {len(html)} байт")
                return parse_html_vacancies(html, city_id)
            elif resp.status in (403, 429):
                logger.warning(f"[HTML] БЛОКИРОВКА ({resp.status})")
                hh_rate_limiter.on_429()
            else:
                logger.warning(f"[HTML] Ошибка {resp.status}")
    except Exception as e:
        logger.error(f"[HTML] ОШИБКА: {e}")
    return []

def parse_html_vacancies(html, city_id):
    vacancies = []
    vacancy_blocks = re.findall(r'data-qa="vacancy-serp__vacancy-title"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>', html)
    for link, title in vacancy_blocks[:20]:
        vacancy = {}
        match = re.search(r"/vacancy/(\d+)", link)
        vacancy["id"] = match.group(1) if match else link
        vacancy["name"] = title.strip()
        vacancy["alternate_url"] = link if link.startswith("http") else f"https://hh.ru{link}"
        company_match = re.search(r'data-qa="vacancy-serp__vacancy-employer"[^>]*>([^<]*)</a>', html)
        vacancy["employer"] = {"name": company_match.group(1).strip() if company_match else "Не указана"}
        city_match = re.search(r'data-qa="vacancy-serp__vacancy-address"[^>]*>([^<]*)</span>', html)
        vacancy["area"] = {"name": city_match.group(1).strip() if city_match else city_id}
        salary_match = re.search(r'data-qa="vacancy-serp__vacancy-compensation"[^>]*>([^<]*)</span>', html)
        vacancy["salary"] = parse_salary(salary_match.group(1)) if salary_match else None
        desc_match = re.search(r'data-qa="vacancy-serp__vacancy_snippet_requirement"[^>]*>([^<]*)</span>', html)
        desc = desc_match.group(1) if desc_match else ""
        vacancy["description"] = desc
        vacancy["snippet"] = {"requirement": desc[:300] + "..." if len(desc) > 300 else desc, "responsibility": ""}
        vacancies.append(vacancy)
    logger.info(f"HTML fallback: распарсено {len(vacancies)} вакансий для {city_id}")
    return vacancies

def parse_salary(text):
    if not text:
        return None
    clean_text = text.replace("\xa0", " ").replace("\u00a0", " ").replace("\u202f", " ")
    clean_text = clean_text.replace(" ", "").replace("\t", "").replace("\n", "")
    match = re.search(r"(от)?(\d+)([-—](\d+))?(₽|руб|RUB|rub)", clean_text, re.IGNORECASE)
    if not match:
        return None
    try:
        salary_from = int(match.group(2)) if match.group(2) else None
        salary_to = int(match.group(4)) if match.group(4) else None
        return {"from": salary_from, "to": salary_to, "currency": "RUR"}
    except (ValueError, TypeError):
        return None

def parse_rss(xml_text):
    vacancies = []
    try:
        root = ET.fromstring(xml_text)
        for item in root.findall(".//item"):
            vacancy = {}
            link = item.findtext("link", "")
            match = re.search(r"/vacancy/(\d+)", link)
            vacancy["id"] = match.group(1) if match else link
            vacancy["name"] = item.findtext("title", "Без названия")
            vacancy["alternate_url"] = link
            vacancy["published"] = item.findtext("pubDate", "")
            desc = item.findtext("description", "")
            vacancy["description"] = desc
            title = vacancy["name"]
            if " — " in title:
                parts = title.rsplit(" — ", 1)
                vacancy["name"] = parts[0].strip()
                vacancy["employer"] = {"name": parts[1].strip()}
            else:
                vacancy["employer"] = {"name": "Не указана"}
            city_match = re.search(r"([А-Яа-я\s-]+)(?:,|\s*•|\s*—)", desc)
            vacancy["area"] = {"name": city_match.group(1).strip() if city_match else "Не указан"}
            vacancy["salary"] = parse_salary(desc)
            vacancy["snippet"] = {"requirement": desc[:300] + "..." if len(desc) > 300 else desc, "responsibility": ""}
            vacancies.append(vacancy)
    except Exception as e:
        logger.error(f"Ошибка обработки RSS: {e}")
    return vacancies

# ========== FORMATTING ==========
def format_vacancy_message(vacancy: dict, score: VacancyScore, cover_letter: str = None) -> str:
    name = vacancy.get("name", "Без названия")
    company = vacancy.get("employer", {}).get("name", "Не указана")
    city = vacancy.get("area", {}).get("name", "Не указан")
    salary = vacancy.get("salary")
    if salary:
        salary_text = str(salary.get("from", "")) + " - " + str(salary.get("to", "")) + " " + str(salary.get("currency", ""))
        salary_text = salary_text.strip().replace("None", "").strip() or "Не указана"
    else:
        salary_text = "Не указана"
    score_bar = "█" * int(score.total / 10) + "░" * (10 - int(score.total / 10))
    desc = vacancy.get("description", "")
    desc_clean = re.sub("<[^<]+?>", "", desc)
    desc_short = desc_clean[:350] + "..." if len(desc_clean) > 350 else desc_clean
    url = vacancy.get("alternate_url", "")
    msg = "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🔹 " + name + "\n"
    msg += "🏢 " + company + "\n"
    msg += "📍 " + city + "\n"
    msg += "💰 " + salary_text + "\n"
    msg += "📊 Скор: " + str(score.total) + "/100 " + score_bar + "\n"
    msg += "   Роль: " + str(score.role_fit) + " | Индустрия: " + str(score.industry_match) + " | ЗП: " + str(score.salary_match) + "\n"
    msg += "   Локация: " + str(score.location_match) + " | Опыт: " + str(score.experience_match) + " | Навыки: " + str(score.skills_match) + "\n"
    msg += "🎯 Вердикт: " + score.verdict + "\n"
    msg += "💡 " + score.reasoning + "\n\n"
    msg += "📋 Описание:\n"
    msg += desc_short + "\n\n"
    msg += "🔗 " + url
    if cover_letter:
        msg += "\n\n📝 Сопроводительное письмо:\n" + cover_letter[:500] + "..."
    msg += "\n━━━━━━━━━━━━━━━━━━━━"
    return msg

def format_digest(vacancies: List[tuple]) -> str:
    msg = "📋 Дайджест вакансий\n\n"
    for i, (vid, title, company, score, _) in enumerate(vacancies[:10], 1):
        bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
        msg += str(i) + ". [" + str(int(score)) + "] " + bar + " " + title + " — " + company + "\n"
    msg += "\nВсего: " + str(len(vacancies)) + " вакансий. Подробности: /top"
    return msg

# ========== BACKGROUND SEARCH ==========
async def background_search(context: ContextTypes.DEFAULT_TYPE):
    global search_lock, last_search_time
    now = asyncio.get_event_loop().time()
    if now - last_search_time < SEARCH_COOLDOWN_SECONDS:
        logger.warning(f"Поиск в cooldown, прошло {now - last_search_time:.0f}с")
        return
    if search_lock.locked():
        logger.warning("Поиск уже выполняется, пропускаю")
        return
    async with search_lock:
        last_search_time = asyncio.get_event_loop().time()
        start_time = asyncio.get_event_loop().time()
        logger.info("🔄 Запуск фонового поиска...")
        chat_id = None
        if context.job and hasattr(context.job, "chat_id") and context.job.chat_id:
            chat_id = context.job.chat_id
        elif CHAT_ID:
            chat_id = int(CHAT_ID)
        if not chat_id:
            logger.error("Нет chat_id для отправки")
            return
        try:
            all_vacancies = []
            async with aiohttp.ClientSession() as session:
                for city_ru, city_id in PROFILE["filters"]["cities"].items():
                    if asyncio.get_event_loop().time() - start_time > MAX_SEARCH_TIME:
                        logger.warning("Достигнут лимит времени поиска, прерываю")
                        break
                    city_vacancies = []
                    for keyword in PROFILE["filters"]["keywords"]:
                        logger.info(f"Поиск: {keyword} в {city_ru}")
                        api_items = await fetch_hh_api(session, city_id, keyword, per_page=20)
                        if api_items:
                            for item in api_items:
                                city_vacancies.append(hh_api_item_to_vacancy(item))
                            logger.info(f"[HH-API] {city_ru}/{keyword}: {len(api_items)} вакансий")
                        else:
                            logger.info(f"[HH-API] Не сработало, пробуем RSS...")
                            rss_result = await fetch_rss(session, city_id, keyword, per_page=20)
                            if rss_result:
                                city_vacancies.extend(rss_result)
                                logger.info(f"[RSS] {city_ru}/{keyword}: {len(rss_result)} вакансий")
                            else:
                                logger.info(f"[RSS] Не сработало, пробуем HTML...")
                                html_result = await fetch_html_fallback(session, city_id, keyword)
                                city_vacancies.extend(html_result)
                                logger.info(f"[HTML] {city_ru}/{keyword}: {len(html_result)} вакансий")
                        await asyncio.sleep(2 + random.uniform(0, 2))
                    all_vacancies.extend(city_vacancies)
                    logger.info(f"Город {city_ru}: {len(city_vacancies)} вакансий всего")
                    await asyncio.sleep(3 + random.uniform(1, 3))
            seen = set()
            unique = [v for v in all_vacancies if not (v["id"] in seen or seen.add(v["id"]))]
            new_vacancies = []
            for v in unique:
                if not await is_vacancy_sent(v["id"]) and not await is_vacancy_seen(v["id"]):
                    new_vacancies.append(v)
            logger.info(f"Найдено {len(unique)} уникальных, новых: {len(new_vacancies)}")
            # Логируем первые 5 вакансий для отладки
            for v in unique[:5]:
                logger.info(f"  Вакансия: {v.get('name', 'N/A')} | {v.get('employer', {}).get('name', 'N/A')} | {v.get('area', {}).get('name', 'N/A')}")
            if not new_vacancies:
                await context.bot.send_message(chat_id=chat_id, text="🔍 Новых вакансий не найдено.")
                await log_search(0, 0, 0)
                return
            scored_vacancies = []
            for v in new_vacancies:
                if asyncio.get_event_loop().time() - start_time > MAX_SEARCH_TIME:
                    break
                heuristic = score_vacancy_heuristic(v)
                ai_score = None
                if heuristic.total >= 30:
                    ai_score = await ask_deepseek_scoring(v)
                if ai_score:
                    final_score = ai_score
                else:
                    final_score = heuristic
                company_name = v.get("employer", {}).get("name", "").lower()
                if any(bl.lower() in company_name for bl in PROFILE["filters"].get("company_blacklist", [])):
                    final_score.verdict = "SKIP"
                    final_score.reasoning = "Компания в чёрном списке"
                title = v.get("name", "").lower()
                if any(bl.lower() in title for bl in PROFILE["filters"].get("title_blacklist", [])):
                    final_score.verdict = "SKIP"
                    final_score.reasoning = "Заголовок в чёрном списке"
                scored_vacancies.append((v, final_score))
                await mark_vacancy_seen(v["id"], v.get("name", ""), v.get("employer", {}).get("name", ""), final_score.total, final_score.verdict)
                await asyncio.sleep(0.3)
            # DEBUG: показываем все вакансии для теста
            matches = [(v, s) for v, s in scored_vacancies if s.verdict in ("STRONG_MATCH", "MATCH")]
            if not matches and scored_vacancies:
                # Если нет совпадений, показываем топ-3 по скору для отладки
                scored_vacancies.sort(key=lambda x: x[1].total, reverse=True)
                matches = scored_vacancies[:3]
                logger.info(f"DEBUG: Нет MATCH, показываем топ-3 по скору для отладки")
            matches.sort(key=lambda x: x[1].total, reverse=True)
            max_per_cycle = PROFILE["notifications"].get("max_per_cycle", MAX_PUSH_PER_CYCLE)
            to_send = matches[:max_per_cycle]
            for v, s in to_send:
                if s.total >= 75:
                    cover = await generate_cover_letter(v)
                    if cover:
                        v["_cover_letter"] = cover
                await mark_vacancy_sent(v["id"], v.get("name", ""), v.get("employer", {}).get("name", ""), s.total)
                await add_application(v["id"], v.get("name", ""), v.get("employer", {}).get("name", ""), s.total)
            avg_score = sum(s.total for _, s in matches) / len(matches) if matches else 0
            await log_search(len(unique), len(to_send), avg_score)
            ds_status = "✅ с AI-скорингом" if deepseek_available else "⚠️ эвристический скоринг"
            if to_send:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"🔍 Найдено {len(matches)} подходящих, отправляю топ-{len(to_send)} {ds_status}:"
                )
                for v, s in to_send:
                    cover = v.get("_cover_letter")
                    msg = format_vacancy_message(v, s, cover)
                    await context.bot.send_message(chat_id=chat_id, text=msg)
                    await asyncio.sleep(2)
            else:
                weak = [(v, s) for v, s in scored_vacancies if s.verdict == "WEAK_MATCH"]
                if weak and PROFILE["notifications"].get("digest_mode", False):
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"🔍 Сильных совпадений нет. {len(weak)} слабых — смотрите /digest"
                    )
                else:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"🔍 Подходящих вакансий не найдено {ds_status}"
                    )
        except Exception as e:
            logger.error(f"Ошибка в background_search: {e}", exc_info=True)
            try:
                await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Ошибка поиска: {str(e)[:300]}")
            except:
                pass

# ========== TELEGRAM COMMANDS ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ds_status = "✅" if deepseek_available else "❌"
    text = "👋 Привет! Я ищу вакансии коммерческого директора в нефтянке.\n\n"
    text += f"🤖 DeepSeek: {ds_status}\n📡 HH.ru (RSS + HTML)\n"
    text += "📊 Скоринг: 0-100 с разбивкой по 6 критериям\n"
    text += "📝 Генерация сопроводительных писем\n"
    text += "📋 Трекинг откликов (Kanban-style)\n\n"
    text += "Команды:\n"
    text += "/search — поиск сейчас\n"
    text += "/schedule on/off — авто-поиск\n"
    text += "/stats — статистика\n"
    text += "/top — топ вакансий по скору\n"
    text += "/digest — дайджест слабых совпадений\n"
    text += "/applications — трекинг откликов\n"
    text += "/status [id] [status] — обновить статус\n"
    text += "/profile — показать профиль\n"
    text += "/editprofile — редактировать профиль\n"
    text += "/filters — фильтры\n"
    text += "/salary [сумма] — зарплата\n"
    text += "/blacklist [компания] — чёрный список\n"
    text += "/relocate — города\n"
    text += "/cleanup — очистить\n"
    text += "/help — справка"
    await update.message.reply_text(text)

async def search_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text("🔍 Ищу вакансии, подождите...")
    try:
        class JobProxy:
            def __init__(self, cid):
                self.chat_id = cid
        original_job = getattr(context, "job", None)
        context.job = JobProxy(chat_id)
        await background_search(context)
        context.job = original_job
        await update.message.reply_text("✅ Поиск завершён!")
    except Exception as e:
        logger.error(f"Ошибка в search_now: {e}", exc_info=True)
        await update.message.reply_text(f"⚠️ Ошибка поиска: {str(e)[:300]}")

async def schedule_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    existing_jobs = context.job_queue.get_jobs_by_name("auto_search")
    for job in existing_jobs:
        job.schedule_removal()
        logger.info(f"Удалён старый job: {job}")
    if not context.args:
        if existing_jobs:
            await update.message.reply_text("⏰ Авто-поиск активен (9:00 и 18:00 UTC)\nОтключить: /schedule off")
        else:
            await update.message.reply_text("❌ Авто-поиск отключен\nВключить: /schedule on")
        return
    command = context.args[0].lower()
    if command == "off":
        await update.message.reply_text("❌ Авто-поиск отключён")
        return
    if command in ["on", "twice"]:
        from datetime import time as dt_time
        context.job_queue.run_daily(
            background_search,
            time=dt_time(hour=9, minute=0),
            chat_id=chat_id,
            name="auto_search"
        )
        context.job_queue.run_daily(
            background_search,
            time=dt_time(hour=18, minute=0),
            chat_id=chat_id,
            name="auto_search"
        )
        await update.message.reply_text("✅ Авто-поиск включён!\n• 9:00 UTC\n• 18:00 UTC\nОтключить: /schedule off")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        total_sent, total_seen, total_searches, recent, app_stats = await get_stats()
        text = f"📊 Статистика:\n\n• Отправлено: {total_sent}\n• Просмотрено: {total_seen}\n• Поисков: {total_searches}\n"
        if app_stats:
            text += "\n📋 Отклики:\n"
            for status, count in app_stats.items():
                text += f"  {status}: {count}\n"
        if recent:
            text += "\nПоследние поиски:\n"
            for found, sent, avg_score, when in recent:
                text += f"  {str(when)[:16]} — найдено {found}, подошло {sent}, средний скор {avg_score:.1f}\n"
        await update.message.reply_text(text)
    except Exception as e:
        logger.error(f"Ошибка stats: {e}")
        await update.message.reply_text(f"⚠️ Ошибка: {str(e)[:200]}")

async def top_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        top = await get_top_vacancies(10)
        if not top:
            await update.message.reply_text("📭 Пока нет сохранённых вакансий")
            return
        msg = "🏆 Топ вакансий по скору:\n\n"
        for i, (vid, title, company, score, sent_at) in enumerate(top, 1):
            bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
            msg += f"{i}. [{score:.0f}] {bar} {title} — {company}\n"
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка: {str(e)[:200]}")

async def digest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, title, company, score, seen_at FROM seen_vacancies WHERE score >= 40 AND score < 60 ORDER BY score DESC LIMIT 15")
        weak = c.fetchall()
        conn.close()
        if not weak:
            await update.message.reply_text("📭 Нет слабых совпадений для дайджеста")
            return
        msg = format_digest(weak)
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка: {str(e)[:200]}")

async def applications_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        apps = await get_applications()
        if not apps:
            await update.message.reply_text("📭 Пока нет отслеживаемых вакансий")
            return
        msg = "📋 Трекинг откликов:\n\n"
        status_emoji = {"new": "🆕", "applied": "📨", "interview": "🗣", "offer": "🎉", "rejected": "❌", "ghosted": "👻"}
        for row in apps[:15]:
            _, vid, title, company, score, status, _, _, _ = row
            emoji = status_emoji.get(status, "🆕")
            msg += f"{emoji} [{score:.0f}] {title} — {company} ({status})\n"
        msg += "\nОбновить статус: /status [id] [new|applied|interview|offer|rejected|ghosted]"
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка: {str(e)[:200]}")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("Использование: /status [vacancy_id] [status] [notes]")
        return
    vid = context.args[0]
    status = context.args[1]
    notes = " ".join(context.args[2:]) if len(context.args) > 2 else ""
    valid = ["new", "applied", "interview", "offer", "rejected", "ghosted"]
    if status not in valid:
        await update.message.reply_text(f"Статус должен быть: {', '.join(valid)}")
        return
    await update_application_status(vid, status, notes)
    await update.message.reply_text(f"✅ Статус обновлён: {status}")

async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Использование: /ask [vacancy_id] [вопрос]")
        return
    question = " ".join(context.args)
    await update.message.reply_text("💡 Используйте /ask после получения вакансии. Функция в разработке.")

async def cover_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Использование: /cover [vacancy_id]")
        return
    vid = context.args[0]
    await update.message.reply_text("📝 Генерирую сопроводительное письмо...")
    await update.message.reply_text("💡 Функция требует сохранения данных вакансии. Используйте после /search.")

async def profile_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reload_profile()
    p = PROFILE["candidate"]
    msg = "👤 Профиль:\n\n"
    msg += f"Имя: {p['name']}\n"
    msg += f"Возраст: {p['age']}\n"
    msg += f"Город: {p['city']}\n"
    msg += f"Переезд: {', '.join(p['relocation_ready'])}\n"
    msg += f"Должности: {', '.join(p['desired_positions'])}\n"
    msg += f"Мин. зарплата: {p['salary_min']} ₽\n"
    msg += "\nКлючевые навыки:\n"
    for skill in p["key_skills"]:
        msg += f"  • {skill}\n"
    msg += "\nРедактировать: /editprofile"
    await update.message.reply_text(msg)

async def editprofile_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "✏️ Редактирование профиля:\n"
    text += "Отредактируйте файл profile.yaml в репозитории и перезапустите бота.\n\n"
    text += "Или используйте команды:\n"
    text += "/salary [сумма] — изменить мин. зарплату\n"
    text += "/blacklist [компания] — добавить в чёрный список"
    await update.message.reply_text(text)

async def filters_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reload_profile()
    cities_str = ", ".join(PROFILE["filters"]["cities"].keys())
    bl = PROFILE["filters"].get("company_blacklist", [])
    bl_str = ", ".join(bl) if bl else "(пусто)"
    text = "🔧 Фильтры:\n"
    text += f"• Мин. зарплата: {PROFILE['filters']['salary_min']} ₽\n"
    text += f"• Города: {cities_str}\n"
    text += f"• Чёрный список компаний: {bl_str}\n"
    text += f"• Макс. за цикл: {PROFILE['notifications'].get('max_per_cycle', MAX_PUSH_PER_CYCLE)}"
    await update.message.reply_text(text)

async def set_salary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Укажите сумму: /salary 250000")
        return
    try:
        new_salary = int(context.args[0])
        if new_salary < 100000:
            await update.message.reply_text("Минимум 100 000 ₽")
            return
        PROFILE["filters"]["salary_min"] = new_salary
        PROFILE["candidate"]["salary_min"] = new_salary
        save_profile(PROFILE)
        await update.message.reply_text(f"✅ Мин. зарплата: {new_salary} ₽")
    except ValueError:
        await update.message.reply_text("Введите число")

async def blacklist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        bl = PROFILE["filters"].get("company_blacklist", [])
        await update.message.reply_text(f"🚫 Чёрный список: {', '.join(bl) if bl else '(пусто)'}")
        return
    company = " ".join(context.args)
    if "company_blacklist" not in PROFILE["filters"]:
        PROFILE["filters"]["company_blacklist"] = []
    if company.lower() in [c.lower() for c in PROFILE["filters"]["company_blacklist"]]:
        PROFILE["filters"]["company_blacklist"].remove(company)
        action = "удалена из"
    else:
        PROFILE["filters"]["company_blacklist"].append(company)
        action = "добавлена в"
    save_profile(PROFILE)
    await update.message.reply_text(f"✅ Компания '{company}' {action} чёрный список")


async def set_token_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ручная установка HH OAuth токена."""
    if not context.args:
        await update.message.reply_text("Использование: /settoken ВАШ_ACCESS_TOKEN [ВАШ_REFRESH_TOKEN]")
        return
    access_token = context.args[0]
    refresh_token = context.args[1] if len(context.args) > 1 else ""
    hh_oauth.access_token = access_token
    hh_oauth.refresh_token = refresh_token
    hh_oauth.token_expires_at = datetime.now() + timedelta(days=14)
    try:
        await hh_oauth.save_tokens()
        await update.message.reply_text(f"✅ Токен сохранён! Проверка: {hh_oauth.has_token()}")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка сохранения: {str(e)[:200]}")

async def relocate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cities = list(PROFILE["filters"]["cities"].keys())
    await update.message.reply_text(f"🌍 Города: {', '.join(cities)}")

async def cleanup_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    deleted = await cleanup_old_vacancies(days=30)
    await update.message.reply_text(f"🗑 Удалено {deleted} старых записей")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "📖 Команды:\n"
    text += "/search — поиск сейчас\n"
    text += "/schedule on/off — авто-поиск\n"
    text += "/stats — статистика\n"
    text += "/top — топ вакансий\n"
    text += "/digest — дайджест слабых совпадений\n"
    text += "/applications — трекинг откликов\n"
    text += "/status [id] [status] — обновить статус\n"
    text += "/profile — профиль\n"
    text += "/editprofile — редактировать профиль\n"
    text += "/filters — фильтры\n"
    text += "/salary [сумма] — зарплата\n"
    text += "/blacklist [компания] — чёрный список\n"
    text += "/relocate — города\n"
    text += "/cleanup — очистить\n"
    text += "/help — справка"
    await update.message.reply_text(text)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Ошибка: {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text("⚠️ Ошибка. Попробуйте позже.")

# ========== WEB SERVER + WEBHOOK ==========
app = FastAPI()

@app.get("/")
async def root():
    return {
        "status": "alive",
        "bot": "hh-bot-v2",
        "deepseek": deepseek_available,
        "profile": PROFILE["candidate"]["name"],
        "time": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/stats")
async def web_stats():
    total_sent, total_seen, total_searches, recent, app_stats = await get_stats()
    return {
        "sent": total_sent,
        "seen": total_seen,
        "searches": total_searches,
        "recent_searches": recent,
        "applications": app_stats
    }


# ─── HH.RU OAUTH ENDPOINTS ───
@app.get("/oauth/login")
async def oauth_login():
    """Ссылка для авторизации в HH.ru. Откройте в браузере."""
    if not hh_oauth.is_configured():
        return {"error": "HH_CLIENT_ID и HH_CLIENT_SECRET не настроены"}
    return {
        "auth_url": hh_oauth.get_auth_url(),
        "instruction": "Откройте этот URL в браузере, авторизуйтесь, затем скопируйте code из redirect URL"
    }

@app.get("/oauth/callback")
async def oauth_callback(code: str = ""):
    """Callback после авторизации HH.ru."""
    if not code:
        return {"error": "Не получен authorization code"}
    result = await hh_oauth.exchange_code(code)
    if result.get("success"):
        return {
            "success": True,
            "message": "Токены получены и сохранены!",
            "access_token": result["access_token"][:20] + "...",
            "refresh_token": result["refresh_token"][:20] + "...",
            "expires_in": result["expires_in"],
        }
    return {"error": "Не удалось получить токены", "details": result.get("error")}

@app.get("/oauth/status")
async def oauth_status():
    """Статус OAuth подключения."""
    return {
        "configured": hh_oauth.is_configured(),
        "has_token": hh_oauth.has_token(),
        "token_expires": hh_oauth.token_expires_at.isoformat() if hh_oauth.token_expires_at else None,
        "auth_url": hh_oauth.get_auth_url() if hh_oauth.is_configured() else None,
    }

@app.post("/oauth/refresh")
async def oauth_refresh():
    """Ручное обновление токена."""
    success = await hh_oauth.refresh_access_token()
    return {"success": success, "has_token": hh_oauth.has_token()}

@app.get("/oauth/debug")
async def oauth_debug():
    """Отладка OAuth статуса."""
    return {
        "configured": hh_oauth.is_configured(),
        "has_token": hh_oauth.has_token(),
        "token_preview": hh_oauth.access_token[:20] + "..." if hh_oauth.access_token else None,
        "refresh_preview": hh_oauth.refresh_token[:20] + "..." if hh_oauth.refresh_token else None,
        "expires": hh_oauth.token_expires_at.isoformat() if hh_oauth.token_expires_at else None,
    }

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return {"ok": True}

application = None

async def run_webhook():
    global application
    await init_db()
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("search", search_now))
    application.add_handler(CommandHandler("schedule", schedule_search))
    application.add_handler(CommandHandler("stats", stats_cmd))
    application.add_handler(CommandHandler("top", top_cmd))
    application.add_handler(CommandHandler("digest", digest_cmd))
    application.add_handler(CommandHandler("applications", applications_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(CommandHandler("ask", ask_cmd))
    application.add_handler(CommandHandler("cover", cover_cmd))
    application.add_handler(CommandHandler("profile", profile_cmd))
    application.add_handler(CommandHandler("editprofile", editprofile_cmd))
    application.add_handler(CommandHandler("filters", filters_cmd))
    application.add_handler(CommandHandler("salary", set_salary))
    application.add_handler(CommandHandler("blacklist", blacklist_cmd))
    application.add_handler(CommandHandler("settoken", set_token_cmd))
    application.add_handler(CommandHandler("relocate", relocate))
    application.add_handler(CommandHandler("cleanup", cleanup_cmd))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_error_handler(error_handler)
    await application.initialize()
    await application.start()
    
    # ← ПРОВЕРКА DEEPEEK ПЕРЕД УСТАНОВКОЙ WEBHOOK
    await check_deepseek_connection()
    
    if RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        await application.bot.set_webhook(url=webhook_url)
        logger.info(f"🔗 Webhook установлен: {webhook_url}")
    config = uvicorn.Config(app, host="0.0.0.0", port=10000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(run_webhook())
 