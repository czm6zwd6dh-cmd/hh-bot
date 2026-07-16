# -*- coding: utf-8 -*-
import os
import sqlite3
import asyncio
import logging
import re
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, ContextTypes, filters
from openai import AsyncOpenAI
import httpx
import aiohttp
from fastapi import FastAPI, Request
import uvicorn
import yaml

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
        "salary_min": 150000,
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
        "salary_min": 150000,
        "cities": {"Волгоград": "24", "Москва": "1", "Санкт-Петербург": "2", "Казань": "88", "Нижний Новгород": "66",
                   "Уфа": "99", "Самара": "78", "Ростов-на-Дону": "76", "Краснодар": "53", "Воронеж": "26",
                   "Астана": "160", "Баку": "100", "Минск": "1002", "Ереван": "104", "Ташкент": "103",
                   "Тюмень": "95", "Омск": "68", "Челябинск": "104", "Пермь": "72", "Саратов": "79"},
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
        "min_score": 35,
        "strict_requirements": {
            "min_salary": 150000,
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
    city_lower = city.lower()
    for c in allowed_cities:
        if c.lower() in city_lower:
            return 10.0
    variants = {
        "волгоград": ["волжский", "камышин"],
        "москва": ["московская", "химки", "красногорск"],
        "санкт-петербург": ["петербург", "ленинград", "пушкин"],
        "казань": ["набережные челны", "нижнекамск"],
        "нижний новгород": ["дзержинск", "арзамас"],
        "уфа": ["стерлитамак", "салават"],
        "астана": ["нур-султан"],
        "минск": ["брест", "гомель"]
    }
    for main_city, syns in variants.items():
        if main_city in city_lower:
            return 8.0
        for syn in syns:
            if syn in city_lower:
                return 8.0
    return 0.0

def calculate_experience_score(text: str) -> float:
    text_lower = text.lower()
    exp_keywords = ["опыт", "лет", "years", "senior", "руководитель", "директор", "manager"]
    score = sum(2 for kw in exp_keywords if kw in text_lower)
    if re.search(r'\b[5-9]\b|\b10\b|\b20\b', text):
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

# ========== SMART RECRUITER WITH LEARNING ==========
class SmartRecruiter:
    def __init__(self):
        self.feedback_history = []
        self.user_preferences = {
            "liked_companies": [],
            "disliked_companies": [],
            "liked_titles": [],
            "disliked_keywords": [],
            "liked_industries": [],
            "salary_importance": 0.15,
            "location_importance": 0.10,
            "industry_importance": 0.25,
        }

    def on_user_action(self, vacancy: dict, score: VacancyScore, action: str):
        company = vacancy.get("employer", {}).get("name", "")
        title = vacancy.get("name", "")
        text = (title + " " + vacancy.get("description", "")).lower()

        if action == "like":
            self.user_preferences["liked_companies"].append(company)
            self.user_preferences["liked_titles"].append(title)
            self.user_preferences["industry_importance"] = min(0.4, self.user_preferences["industry_importance"] + 0.02)
        elif action == "dislike":
            self.user_preferences["disliked_companies"].append(company)
            self.user_preferences["disliked_keywords"].extend(
                [w for w in title.lower().split() if len(w) > 4]
            )
        elif action == "apply":
            self.user_preferences["liked_companies"].append(company)
            self.user_preferences["industry_importance"] = min(0.5, self.user_preferences["industry_importance"] + 0.05)

        self.feedback_history.append({
            "action": action, "company": company, "title": title, "score": score.total,
            "timestamp": datetime.now().isoformat()
        })

    def adjust_score(self, vacancy: dict, base_score: VacancyScore) -> VacancyScore:
        company = vacancy.get("employer", {}).get("name", "").lower()
        title = vacancy.get("name", "").lower()
        text = (title + " " + vacancy.get("description", "")).lower()

        adjustment = 0
        reasons = []

        liked = [c.lower() for c in self.user_preferences["liked_companies"][-15:]]
        if any(lc in company for lc in liked):
            adjustment += 8
            reasons.append("компания из понравившихся")

        disliked = [c.lower() for c in self.user_preferences["disliked_companies"][-15:]]
        if any(dc in company for dc in disliked):
            adjustment -= 15
            reasons.append("компания из непонравившихся")

        for kw in self.user_preferences["disliked_keywords"][-20:]:
            if len(kw) > 4 and kw in text:
                adjustment -= 5
                reasons.append("содержит '" + kw + "' из нежелательных")
                break

        new_total = min(100, max(0, base_score.total + adjustment))

        if new_total >= 80:
            verdict = "STRONG_MATCH"
        elif new_total >= PROFILE["scoring"]["min_score"]:
            verdict = "MATCH"
        elif new_total >= 40:
            verdict = "WEAK_MATCH"
        else:
            verdict = "SKIP"

        reasoning = base_score.reasoning
        if reasons:
            reasoning += " | Учёт предпочтений: " + "; ".join(reasons)

        return VacancyScore(
            total=round(new_total, 1), role_fit=base_score.role_fit,
            industry_match=base_score.industry_match, salary_match=base_score.salary_match,
            location_match=base_score.location_match, experience_match=base_score.experience_match,
            skills_match=base_score.skills_match, verdict=verdict, reasoning=reasoning
        )

    def get_summary(self) -> str:
        total = len(self.feedback_history)
        likes = sum(1 for f in self.feedback_history if f["action"] == "like")
        dislikes = sum(1 for f in self.feedback_history if f["action"] == "dislike")
        applies = sum(1 for f in self.feedback_history if f["action"] == "apply")
        msg = "🧠 Обучение рекрутера:\n\nВсего оценок: " + str(total) + "\n"
        msg += "  👍 Лайков: " + str(likes) + "\n  👎 Дизлайков: " + str(dislikes) + "\n  📝 Откликов: " + str(applies) + "\n\n"
        msg += "Веса: индустрия=" + str(round(self.user_preferences['industry_importance'], 2))
        msg += ", зарплата=" + str(round(self.user_preferences['salary_importance'], 2))
        return msg

smart_recruiter = SmartRecruiter()

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ЗАГЛУШКИ) ==========
async def fetch_hh_api(session, area_id, keyword, per_page=10):
    logger.warning("fetch_hh_api не реализован, возвращаем пустой список")
    return []

async def fetch_rss(session, area_id, keyword, per_page=10):
    logger.warning("fetch_rss не реализован, возвращаем пустой список")
    return []

def hh_api_item_to_vacancy(item):
    return {
        "id": item.get("id", "0"),
        "name": item.get("name", "Вакансия"),
        "employer": {"name": item.get("employer", {}).get("name", "Компания")},
        "area": {"name": item.get("area", {}).get("name", "Город")},
        "salary": item.get("salary"),
        "snippet": {"requirement": "", "responsibility": ""},
        "description": ""
    }

def format_vacancy_message(vacancy, score):
    msg = f"<b>{vacancy.get('name', 'Без названия')}</b>\n"
    msg += f"🏢 {vacancy.get('employer', {}).get('name', 'Неизвестно')}\n"
    msg += f"📍 {vacancy.get('area', {}).get('name', 'Не указан')}\n"
    salary = vacancy.get('salary')
    if salary:
        s = f"{salary.get('from', '')} - {salary.get('to', '')} {salary.get('currency', '')}".strip()
        if s:
            msg += f"💰 {s}\n"
    msg += f"\n📊 Скор: {score.total} ({score.verdict})\n"
    msg += f"   • Роль: {score.role_fit}/10\n"
    msg += f"   • Индустрия: {score.industry_match}/10\n"
    msg += f"   • Зарплата: {score.salary_match}/10\n"
    msg += f"   • Локация: {score.location_match}/10\n"
    msg += f"   • Опыт: {score.experience_match}/10\n"
    msg += f"   • Навыки: {score.skills_match}/10\n"
    msg += f"\n💬 {score.reasoning}"
    return msg

def format_digest(weak_list):
    msg = "📋 Дайджест слабых совпадений:\n\n"
    for vid, title, company, score, seen_at in weak_list:
        msg += f"• {title} — {company} (скор {score:.0f})\n"
    return msg

async def generate_cover_letter(vacancy):
    logger.warning("generate_cover_letter не реализован, возвращаем заглушку")
    return "Здравствуйте! Меня заинтересовала ваша вакансия. Я обладаю необходимым опытом и навыками. Готов обсудить детали."

async def ask_rag_about_vacancy(vacancy, question):
    return f"Извините, функция ответа на вопросы о вакансии в разработке. Ваш вопрос: {question}"

# ========== OAuth ДЛЯ HH.RU (ЗАГЛУШКА) ==========
class HHOAuth:
    def __init__(self):
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None

    def is_configured(self):
        return False

    def has_token(self):
        return self.access_token is not None

    def get_auth_url(self):
        return "https://hh.ru/oauth/authorize?client_id=FAKE&redirect_uri=FAKE"

    async def exchange_code(self, code):
        return {"success": False, "error": "OAuth не настроен"}

    async def refresh_access_token(self):
        return False

    async def save_tokens(self):
        pass

hh_oauth = HHOAuth()

# ========== ОСНОВНЫЕ ФУНКЦИИ ПОИСКА ==========
async def background_search(context: ContextTypes.DEFAULT_TYPE):
    await smart_background_search(context)

async def smart_background_search(context: ContextTypes.DEFAULT_TYPE):
    global search_lock, last_search_time
    now = asyncio.get_running_loop().time()
    if now - last_search_time < SEARCH_COOLDOWN_SECONDS:
        return
    if search_lock.locked():
        return

    async with search_lock:
        last_search_time = asyncio.get_running_loop().time()
        chat_id = context.job.chat_id if (context.job and hasattr(context.job, "chat_id")) else (int(CHAT_ID) if CHAT_ID else None)
        if not chat_id:
            return

        # Тестовая вакансия
        test_vacancy = {
            "id": "test_123",
            "name": "Коммерческий директор (нефтепродукты)",
            "employer": {"name": "ООО НефтеТрейд"},
            "area": {"name": "Волгоград"},
            "salary": {"from": 200000, "to": 250000, "currency": "RUR"},
            "snippet": {"requirement": "Опыт управления отделом продаж ГСМ от 5 лет", "responsibility": "Руководство отделом"},
            "description": "Управление продажами нефтепродуктов, работа с НПЗ, логистика."
        }
        score = score_vacancy_heuristic(test_vacancy)
        final = smart_recruiter.adjust_score(test_vacancy, score)

        context.chat_data["vac_queue"] = [(test_vacancy, final)]
        context.chat_data["vac_idx"] = 0

        await context.bot.send_message(
            chat_id=chat_id,
            text="🎯 Найдена тестовая вакансия. Для реального поиска реализуйте fetch_hh_api и fetch_rss."
        )
        msg, kb = format_vacancy_with_buttons(test_vacancy, final, 1, 1)
        await context.bot.send_message(chat_id=chat_id, text=msg, reply_markup=kb)

        await log_search(1, 1, final.total)

# ========== SMART RECRUITER CALLBACKS ==========
async def handle_vacancy_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = update.effective_chat.id

    if data == "next":
        idx = context.chat_data.get("vac_idx", 0) + 1
        queue = context.chat_data.get("vac_queue", [])
        context.chat_data["vac_idx"] = idx
        if idx < len(queue):
            v, s = queue[idx]
            msg, kb = format_vacancy_with_buttons(v, s, idx+1, len(queue))
            await query.edit_message_text(msg, reply_markup=kb)
        else:
            await query.edit_message_text("✅ Все вакансии просмотрены!")
        return

    if ":" not in data:
        return
    action, vid = data.split(":", 1)
    queue = context.chat_data.get("vac_queue", [])
    vacancy = score = None
    for v, s in queue:
        if v.get("id") == vid:
            vacancy, score = v, s
            break
    if not vacancy:
        await query.edit_message_text("⚠️ Вакансия не найдена")
        return

    if action == "like":
        smart_recruiter.on_user_action(vacancy, score, "like")
        await query.edit_message_text(query.message.text + "\n\n✅ Отмечено: интересно", reply_markup=None)
    elif action == "dislike":
        smart_recruiter.on_user_action(vacancy, score, "dislike")
        await query.edit_message_text(query.message.text + "\n\n❌ Отмечено: не подходит", reply_markup=None)
    elif action == "apply":
        smart_recruiter.on_user_action(vacancy, score, "apply")
        cover = await generate_cover_letter(vacancy)
        if cover:
            await context.bot.send_message(chat_id=chat_id, text="📝 Письмо:\n" + cover)
        await query.edit_message_text(query.message.text + "\n\n📝 Отмечено для отклика", reply_markup=None)
        await add_application(vid, vacancy.get("name",""), vacancy.get("employer",{}).get("name",""), score.total)
    elif action == "cover":
        cover = await generate_cover_letter(vacancy)
        if cover:
            await context.bot.send_message(chat_id=chat_id, text="📝 Письмо:\n" + cover)
        else:
            await query.answer("Не удалось сгенерировать")

def format_vacancy_with_buttons(vacancy, score, idx=0, total=0):
    msg = format_vacancy_message(vacancy, score)
    if total:
        msg = "📌 " + str(idx) + "/" + str(total) + "\n\n" + msg
    vid = vacancy.get("id", "")
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("👍", callback_data="like:" + vid),
         InlineKeyboardButton("👎", callback_data="dislike:" + vid),
         InlineKeyboardButton("📝", callback_data="apply:" + vid)],
        [InlineKeyboardButton("💬 Письмо", callback_data="cover:" + vid),
         InlineKeyboardButton("➡️ Далее", callback_data="next")]
    ])
    return msg, kb

# ========== КОМАНДЫ ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ds_status = "✅" if deepseek_available else "❌"
    text = "👋 Привет! Я ищу вакансии коммерческого директора в нефтянке.\n\n"
    text += f"🤖 DeepSeek: {ds_status}\n📡 HH.ru (RSS + API) – заглушка\n"
    text += "📊 Скоринг: 0-100 с разбивкой по 6 критериям\n"
    text += "📝 Генерация сопроводительных писем (заглушка)\n"
    text += "📋 Трекинг откликов (Kanban-style)\n\n"
    text += "Команды:\n"
    text += "/search — поиск сейчас (тестовый)\n"
    text += "/smart — умный поиск с обучением\n"
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
        await update.message.reply_text("✅ Поиск завершён (тестовый режим)")
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
    text += "/search — поиск сейчас (тестовый)\n"
    text += "/smart — умный поиск с обучением\n"
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

async def smart_search_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text("🧠 Умный рекрутер ищет...")
    class JobProxy:
        def __init__(self, cid):
            self.chat_id = cid
    context.job = JobProxy(chat_id)
    await smart_background_search(context)
    context.job = None

async def learning_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(smart_recruiter.get_summary())

async def reset_learning_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global smart_recruiter
    smart_recruiter = SmartRecruiter()
    await update.message.reply_text("🔄 Обучение сброшено!")

async def handle_text_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    vacancy_id = context.chat_data.pop("asking_about", None)
    question = update.message.text
    if not vacancy_id:
        await handle_natural_message(update, context)
        return
    queue = context.chat_data.get("vac_queue", [])
    vacancy = None
    for v, s in queue:
        if v.get("id") == vacancy_id:
            vacancy = v
            break
    if not vacancy:
        await update.message.reply_text("⚠️ Вакансия не найдена")
        return
    answer = await ask_rag_about_vacancy(vacancy, question)
    await update.message.reply_text("💡 Ответ:\n\n" + answer)

# ========== NATURAL LANGUAGE ASSISTANT (с эвристическим fallback) ==========
# Промпт для DeepSeek
NL_ASSISTANT_PROMPT = """Ты — интеллектуальный ассистент бота для поиска работы.
Пользователь пишет тебе на естественном языке. Ты должен:
1. Понять намерение пользователя
2. Определить, какое действие нужно выполнить
3. Вернуть JSON с инструкцией

Доступные действия:
- "search" — поиск вакансий
- "smart_search" — умный поиск с обучением
- "set_salary" — изменить минимальную зарплату
- "add_city" — добавить город
- "remove_city" — удалить город
- "add_blacklist" — добавить компанию в чёрный список
- "remove_blacklist" — удалить из чёрного списка
- "show_profile" — показать профиль
- "show_stats" — показать статистику
- "show_learning" — показать обучение
- "reset_learning" — сбросить обучение
- "add_keyword" — добавить ключевое слово поиска
- "remove_keyword" — удалить ключевое слово
- "set_min_score" — изменить минимальный скор
- "explain" — объяснить почему вакансия подошла/не подошла
- "help" — справка
- "unknown" — непонятно, попросить уточнить

Текущий профиль пользователя:
{profile}

Примеры:
"Найди вакансии" → {"action": "search"}
"Поищи работу" → {"action": "smart_search"}
"Хочу зарплату от 200 тысяч" → {"action": "set_salary", "value": 200000}
"Добавь Москву" → {"action": "add_city", "value": "Москва", "area_id": "1"}
"Убери стажёров" → {"action": "add_blacklist", "value": "стажёр"}
"Покажи профиль" → {"action": "show_profile"}
"Сколько вакансий нашёл" → {"action": "show_stats"}
"Почему эта вакансия" → {"action": "explain"}
"Сбрось обучение" → {"action": "reset_learning"}
"Добавь ключевое слово директор по развитию" → {"action": "add_keyword", "value": "директор по развитию"}
"Установи минимальный скор 40" → {"action": "set_min_score", "value": 40}
"Вакансии в Воронеже" → {"action": "search", "filters": {"city": "Воронеж"}}
"Только удалёнка" → {"action": "unknown", "message": "Удалённая работа не настроена в профиле. Используйте /editprofile для настройки."}

Верни СТРОГО JSON без markdown:
{"action": "search", "value": null, "filters": {}, "message": "краткий ответ пользователю"}"""

# Эвристический парсер (без AI) для базовых команд
def heuristic_parse(text: str) -> dict:
    text_lower = text.lower().strip()
    # Поиск
    if any(word in text_lower for word in ["поиск", "найди", "вакансии", "ищи", "найти работу"]):
        return {"action": "search", "value": None, "message": "🔍 Ищу вакансии по вашему запросу"}
    # Статистика
    if any(word in text_lower for word in ["статистика", "стат", "сколько"]):
        return {"action": "show_stats", "value": None, "message": "📊 Показываю статистику"}
    # Профиль
    if any(word in text_lower for word in ["профиль", "мои данные", "кто я"]):
        return {"action": "show_profile", "value": None, "message": "👤 Ваш профиль"}
    # Помощь
    if any(word in text_lower for word in ["помощь", "help", "что умеешь", "команды"]):
        return {"action": "help", "value": None, "message": "📖 Список команд"}
    # Приветствие
    if any(word in text_lower for word in ["привет", "здравствуй", "хай", "hello", "ку"]):
        return {"action": "help", "value": None, "message": "👋 Привет! Я бот для поиска вакансий. Напиши /help для списка команд."}
    # Обучение
    if any(word in text_lower for word in ["обучение", "настройка", "умный"]):
        return {"action": "show_learning", "value": None, "message": "🧠 Показываю настройки рекрутера"}
    # Сброс обучения
    if "сброс" in text_lower and "обуч" in text_lower:
        return {"action": "reset_learning", "value": None, "message": "🔄 Обучение сброшено"}
    # Зарплата
    if "зарплат" in text_lower and ("от" in text_lower or "больше" in text_lower):
        import re
        nums = re.findall(r'\b\d{5,}\b', text)
        if nums:
            return {"action": "set_salary", "value": int(nums[0]), "message": f"Устанавливаю зарплату от {nums[0]}"}
    # Добавить город
    if "добавь" in text_lower and "город" in text_lower:
        # можно вытащить название города, но для простоты оставим unknown
        pass
    return {"action": "unknown", "value": None, "message": "Не понял запрос. Попробуйте /help"}

async def process_natural_language(text: str) -> dict:
    global deepseek_available, client

    # Если AI недоступен – используем эвристику
    if not deepseek_available or not client:
        logger.warning("DeepSeek недоступен, используем эвристический парсер")
        return heuristic_parse(text)

    profile_text = yaml.dump(PROFILE, allow_unicode=True, sort_keys=False)[:2000]
    user_msg = "Сообщение пользователя: \"" + text + "\""
    prompt = NL_ASSISTANT_PROMPT.format(profile=profile_text) + "\n\n" + user_msg

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            ),
            timeout=15.0
        )
        answer = response.choices[0].message.content.strip()
        logger.info(f"Raw AI response: {answer}")  # ← для отладки

        # Очищаем от markdown
        answer = re.sub(r"```json\s*", "", answer)
        answer = re.sub(r"```\s*", "", answer)
        result = json.loads(answer)

        if not isinstance(result, dict):
            raise ValueError("Ответ не является словарём")

        if "action" not in result:
            result["action"] = "unknown"
            result["message"] = result.get("message", "Не удалось определить действие")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}, ответ: {answer}")
        # Fallback на эвристику
        return heuristic_parse(text)
    except Exception as e:
        logger.error(f"NL processing error: {e}")
        return heuristic_parse(text)

async def handle_natural_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    chat_id = update.effective_chat.id

    logger.info(f"[NL] Получено сообщение: {text}")

    if "asking_about" in context.chat_data:
        await handle_text_question(update, context)
        return

    await update.message.reply_text("🤔 Думаю...")
    result = await process_natural_language(text)

    action = result.get("action", "unknown")
    value = result.get("value")
    message = result.get("message", "")

    logger.info(f"[NL] Распознано действие: {action}")

    # Выполняем действие
    if action == "search":
        await update.message.reply_text(message or "🔍 Ищу вакансии...")
        await search_now(update, context)

    elif action == "smart_search":
        await update.message.reply_text(message or "🧠 Умный рекрутер ищет...")
        await smart_search_cmd(update, context)

    elif action == "set_salary":
        if value and isinstance(value, (int, float)):
            PROFILE["filters"]["salary_min"] = int(value)
            PROFILE["candidate"]["salary_min"] = int(value)
            save_profile(PROFILE)
            await update.message.reply_text(f"✅ Минимальная зарплата: {int(value)} ₽")
        else:
            await update.message.reply_text("Укажите сумму, например: Зарплата от 200000")

    elif action == "add_city":
        city_name = value
        area_id = result.get("area_id", "")
        if city_name and area_id:
            PROFILE["filters"]["cities"][city_name] = str(area_id)
            save_profile(PROFILE)
            await update.message.reply_text(f"✅ Город добавлен: {city_name}")
        else:
            await update.message.reply_text("⚠️ Нужен ID города. Пример: Добавь Самару (area=78)")

    elif action == "remove_city":
        city_name = value
        if city_name and city_name in PROFILE["filters"]["cities"]:
            del PROFILE["filters"]["cities"][city_name]
            save_profile(PROFILE)
            await update.message.reply_text(f"✅ Город удалён: {city_name}")
        else:
            await update.message.reply_text(f"Город не найден: {city_name}")

    elif action == "add_blacklist":
        company = value
        if company:
            if "company_blacklist" not in PROFILE["filters"]:
                PROFILE["filters"]["company_blacklist"] = []
            if company not in PROFILE["filters"]["company_blacklist"]:
                PROFILE["filters"]["company_blacklist"].append(company)
                save_profile(PROFILE)
            await update.message.reply_text(f"🚫 Добавлено в чёрный список: {company}")

    elif action == "remove_blacklist":
        company = value
        if company and "company_blacklist" in PROFILE["filters"]:
            if company in PROFILE["filters"]["company_blacklist"]:
                PROFILE["filters"]["company_blacklist"].remove(company)
                save_profile(PROFILE)
            await update.message.reply_text(f"✅ Удалено из чёрного списка: {company}")

    elif action == "add_keyword":
        kw = value
        if kw and kw not in PROFILE["filters"]["keywords"]:
            PROFILE["filters"]["keywords"].append(kw)
            save_profile(PROFILE)
            await update.message.reply_text(f"✅ Ключевое слово добавлено: {kw}")
        else:
            await update.message.reply_text("Ключевое слово уже есть или не указано")

    elif action == "remove_keyword":
        kw = value
        if kw and kw in PROFILE["filters"]["keywords"]:
            PROFILE["filters"]["keywords"].remove(kw)
            save_profile(PROFILE)
            await update.message.reply_text(f"✅ Ключевое слово удалено: {kw}")

    elif action == "set_min_score":
        if value and isinstance(value, (int, float)):
            PROFILE["scoring"]["min_score"] = int(value)
            save_profile(PROFILE)
            await update.message.reply_text(f"✅ Минимальный скор: {int(value)}")
        else:
            await update.message.reply_text("Укажите число от 0 до 100")

    elif action == "show_profile":
        await profile_cmd(update, context)

    elif action == "show_stats":
        await stats_cmd(update, context)

    elif action == "show_learning":
        await learning_cmd(update, context)

    elif action == "reset_learning":
        await reset_learning_cmd(update, context)

    elif action == "explain":
        await update.message.reply_text("💡 Используйте /why [id_вакансии] для объяснения оценки")

    elif action == "help":
        await help_cmd(update, context)

    else:
        help_text = "🤔 Не понял запрос. Попробуйте:\n"
        help_text += "• Найди вакансии\n"
        help_text += "• Зарплата от 200000\n"
        help_text += "• Добавь Москву\n"
        help_text += "• Покажи профиль\n"
        help_text += "• Или используйте /help"
        await update.message.reply_text(message or help_text)

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

@app.get("/oauth/login")
async def oauth_login():
    if not hh_oauth.is_configured():
        return {"error": "HH_CLIENT_ID и HH_CLIENT_SECRET не настроены"}
    return {
        "auth_url": hh_oauth.get_auth_url(),
        "instruction": "Откройте этот URL в браузере, авторизуйтесь, затем скопируйте code из redirect URL"
    }

@app.get("/oauth/callback")
async def oauth_callback(code: str = ""):
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
    return {
        "configured": hh_oauth.is_configured(),
        "has_token": hh_oauth.has_token(),
        "token_expires": hh_oauth.token_expires_at.isoformat() if hh_oauth.token_expires_at else None,
        "auth_url": hh_oauth.get_auth_url() if hh_oauth.is_configured() else None,
    }

@app.post("/oauth/refresh")
async def oauth_refresh():
    success = await hh_oauth.refresh_access_token()
    return {"success": success, "has_token": hh_oauth.has_token()}

@app.get("/oauth/debug")
async def oauth_debug():
    return {
        "configured": hh_oauth.is_configured(),
        "has_token": hh_oauth.has_token(),
        "token_preview": hh_oauth.access_token[:20] + "..." if hh_oauth.access_token else None,
        "refresh_preview": hh_oauth.refresh_token[:20] + "..." if hh_oauth.refresh_token else None,
        "expires": hh_oauth.token_expires_at.isoformat() if hh_oauth.token_expires_at else None,
    }

@app.post("/webhook")
async def webhook(request: Request):
    global application
    data = await request.json()
    if application is None:
        return {"error": "Application not initialized"}
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
    application.add_handler(CommandHandler("smart", smart_search_cmd))
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
    application.add_handler(CommandHandler("learning", learning_cmd))
    application.add_handler(CommandHandler("resetlearning", reset_learning_cmd))
    application.add_handler(CallbackQueryHandler(handle_vacancy_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_natural_message))
    application.add_error_handler(error_handler)

    await application.initialize()
    await application.start()

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