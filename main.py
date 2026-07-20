# -*- coding: utf-8 -*-
"""
Бот-поисковик вакансий с DeepSeek скорингом v2.0

Исправления:
- Увеличен лимит вакансий (10-20 за раз)
- Все города и ключевые слова из профиля
- Гибкий парсер команд и естественного языка
- Умный фильтр: зарплата, город, удалёнка из текста
- Дедупликация по названию+компании
- Fallback: показывает лучшие, если порог не пройден
- Пагинация: кнопка "Ещё вакансии"

Переменные окружения:
  TELEGRAM_TOKEN      — токен бота (обязательно)
  HH_CLIENT_ID        — client_id HH
  HH_CLIENT_SECRET    — client_secret HH
  HH_USER_AGENT       — "App/1.0 (email@example.com)"
  SUPERJOB_KEY        — ключ SuperJob (опционально)
  DEEPSEEK_API_KEY    — для скоринга (обязательно)
  RENDER_EXTERNAL_URL — для webhook (опционально)
"""

import os
import sqlite3
import asyncio
import logging
import re
import yaml
import aiohttp
import random
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from fastapi import FastAPI, Request
import uvicorn
from openai import AsyncOpenAI
import httpx

from proxy_pool import ProxyPool

# --- Логирование ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Загрузка .env ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Конфигурация ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")

HH_CLIENT_ID = os.getenv("HH_CLIENT_ID")
HH_CLIENT_SECRET = os.getenv("HH_CLIENT_SECRET")
HH_USER_AGENT = os.getenv("HH_USER_AGENT", "JobSearchBot/1.0 (your_email@gmail.com)")
SUPERJOB_KEY = os.getenv("SUPERJOB_KEY")

PROXY_RETRIES = 6
REQUEST_TIMEOUT = 15

# === НОВОЕ: настройки выдачи ===
VACANCIES_PER_PAGE = 10        # сколько показывать за раз
MAX_VACANCIES_TOTAL = 50       # максимум собирать с API
MIN_VACANCIES_TO_SHOW = 5      # минимум для показа (fallback на лучшие)
HH_PER_PAGE = 20               # было 5 → теперь 20
SJ_COUNT = 20                  # было 5 → теперь 20
TV_PER_PAGE = 20               # было 10 → теперь 20

DB_PATH = "vacancies.db"
PROFILE_PATH = "profile.yaml"

search_in_progress = False

# --- DeepSeek клиент ---
deepseek_client = None
deepseek_available = False
if DEEPSEEK_API_KEY:
    try:
        http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        deepseek_client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
            http_client=http_client,
        )
        deepseek_available = True
        logger.info("DeepSeek клиент создан")
    except Exception as e:
        logger.warning(f"DeepSeek init error: {e}")

# --- Профиль ---
DEFAULT_PROFILE = {
    "candidate": {
        "name": "Виктор Зинченко",
        "desired_positions": ["коммерческий директор", "руководитель отдела продаж", "директор по продажам"],
        "key_skills": ["управление продажами", "нефтепродукты", "B2B", "логистика"],
        "salary_min": 150000,
    },
    "filters": {
        "cities": {
            "Волгоград": "24",
            "Москва": "1",
            "Казань": "88",
            "Нижний Новгород": "66",
            "Уфа": "99",
            "Санкт-Петербург": "2",
        },
        "keywords": [
            "коммерческий директор",
            "директор по продажам",
            "руководитель отдела продаж",
        ],
        "industry_keywords": [
            "нефтепродукты", "ГСМ", "топливо", "нефть", "нефтегаз",
            "нефтепереработка", "азс", "бензин", "дизель", "мазут",
            "нефтебаза", "трубопровод", "нефтяной", "газовый", "пгт",
            "сжиженный газ", "спг",
        ],
        "exclude_words": ["стажёр", "junior", "стажер", "ассистент", "помощник"],
    },
    "scoring": {
        "min_score": 55,
        "weights": {
            "role_fit": 0.25,
            "industry_match": 0.30,
            "salary_match": 0.15,
            "location_match": 0.10,
            "experience_match": 0.10,
            "skills_match": 0.10,
        },
    },
}

def load_profile():
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        yaml.dump(DEFAULT_PROFILE, f, allow_unicode=True, sort_keys=False)
    return DEFAULT_PROFILE.copy()

PROFILE = load_profile()

# --- База данных ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS seen_vacancies (
            id TEXT PRIMARY KEY,
            title TEXT,
            company TEXT,
            score REAL,
            seen_at TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vacancy_id TEXT,
            title TEXT,
            company TEXT,
            score REAL,
            status TEXT,
            created_at TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS seen_signatures (
            signature TEXT PRIMARY KEY,
            seen_at TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def is_seen(vid: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("SELECT 1 FROM seen_vacancies WHERE id=?", (vid,))
        return c.fetchone() is not None
    finally:
        conn.close()

def is_seen_signature(signature: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("SELECT 1 FROM seen_signatures WHERE signature=?", (signature,))
        return c.fetchone() is not None
    finally:
        conn.close()

def mark_seen(vid: str, title: str, company: str, score: float):
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO seen_vacancies VALUES (?,?,?,?,?)",
            (vid, title, company, score, datetime.now())
        )
        signature = f"{title.lower().strip()}|{company.lower().strip()}"
        c.execute(
            "INSERT OR IGNORE INTO seen_signatures VALUES (?,?)",
            (signature, datetime.now())
        )
        conn.commit()
    finally:
        conn.close()

def add_application(vid: str, title: str, company: str, score: float):
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute(
            "INSERT OR IGNORE INTO applications (vacancy_id, title, company, score, status, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (vid, title, company, score, "new", datetime.now())
        )
        conn.commit()
    finally:
        conn.close()

def get_stats() -> Tuple[int, int]:
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM seen_vacancies")
        seen = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM applications")
        apps = c.fetchone()[0]
        return seen, apps
    finally:
        conn.close()

def clear_seen():
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("DELETE FROM seen_vacancies WHERE id LIKE 'demo_%'")
        c.execute("DELETE FROM seen_signatures WHERE signature LIKE '%|demo%'")
        conn.commit()
    finally:
        conn.close()

# =====================================================================
#  АНТИБЛОК-СЛОЙ
# =====================================================================

proxy_pool = ProxyPool(
    test_url="https://api.hh.ru/vacancies",
    test_params={"text": "test", "per_page": 1},
    test_headers={"User-Agent": HH_USER_AGENT},
)

_aiohttp_session: Optional[aiohttp.ClientSession] = None

async def get_session() -> aiohttp.ClientSession:
    global _aiohttp_session
    if _aiohttp_session is None or _aiohttp_session.closed:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        _aiohttp_session = aiohttp.ClientSession(timeout=timeout)
    return _aiohttp_session

async def close_session():
    global _aiohttp_session
    if _aiohttp_session and not _aiohttp_session.closed:
        await _aiohttp_session.close()
        _aiohttp_session = None


class HHAuth:
    def __init__(self):
        self.token: Optional[str] = None
        self._lock = asyncio.Lock()

    async def get(self) -> Optional[str]:
        if not (HH_CLIENT_ID and HH_CLIENT_SECRET):
            return None
        if self.token:
            return self.token
        async with self._lock:
            if self.token:
                return self.token
            try:
                session = await get_session()
                async with session.post(
                    "https://hh.ru/oauth/token",
                    data={
                        "grant_type": "client_credentials",
                        "client_id": HH_CLIENT_ID,
                        "client_secret": HH_CLIENT_SECRET,
                    },
                    headers={"User-Agent": HH_USER_AGENT},
                    ssl=False,
                ) as r:
                    if r.status == 200:
                        data = await r.json()
                        self.token = data.get("access_token")
                        logger.info("HH: токен приложения получен")
                        return self.token
                    text = await r.text()
                    logger.warning(f"HH oauth/token status {r.status}: {text[:150]}")
            except Exception as e:
                logger.error(f"HH oauth error: {e}")
        return None

    def reset(self):
        self.token = None

hh_auth = HHAuth()


async def fetch_json(url: str, *, params=None, headers=None, retry_statuses=(403, 404, 429, 502, 503)) -> Tuple[Optional[Dict], Optional[int]]:
    session = await get_session()
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

    direct_status = None
    try:
        async with session.get(url, params=params, headers=headers, ssl=False) as r:
            direct_status = r.status
            if r.status == 200:
                data = await r.json(content_type=None)
                return data, 200
    except Exception as e:
        logger.warning(f"Direct {url}: {e}")
        direct_status = None

    if direct_status is not None and direct_status not in retry_statuses:
        return None, direct_status

    logger.info(f"Прямой запрос к {url} не прошёл (status={direct_status}), иду через прокси...")
    tried = set()
    for attempt in range(PROXY_RETRIES):
        proxy = proxy_pool.get()
        if not proxy or proxy in tried:
            if proxy_pool.size == 0:
                break
            continue
        tried.add(proxy)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as proxy_session:
                async with proxy_session.get(
                    url, params=params, headers=headers,
                    proxy=f"http://{proxy}", ssl=False
                ) as r:
                    if r.status == 200:
                        try:
                            data = await r.json(content_type=None)
                        except Exception:
                            proxy_pool.report_bad(proxy)
                            continue
                        if isinstance(data, dict):
                            proxy_pool.report_good(proxy)
                            logger.info(f"OK через прокси {proxy} (попытка {attempt + 1})")
                            return data, 200
                    proxy_pool.report_bad(proxy)
        except Exception:
            proxy_pool.report_bad(proxy)
        await asyncio.sleep(random.uniform(0.3, 1.0))

    return None, direct_status


# =====================================================================
#  ИСТОЧНИКИ ВАКАНСИЙ
# =====================================================================

async def fetch_hh_api(city_id: str, keyword: str, per_page: int = 20) -> List[Dict]:
    url = "https://api.hh.ru/vacancies"
    params = {
        "area": city_id,
        "text": keyword,
        "per_page": per_page,
        "order_by": "publication_time",
    }
    headers = {"User-Agent": HH_USER_AGENT, "Accept": "application/json"}

    token = await hh_auth.get()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data, status = await fetch_json(url, params=params, headers=headers)

    if status == 401 and token:
        hh_auth.reset()
        headers = {"User-Agent": HH_USER_AGENT, "Accept": "application/json"}
        data, status = await fetch_json(url, params=params, headers=headers)

    if not data or not isinstance(data, dict):
        logger.warning(f"HH API недоступен (status={status}) для '{keyword}'")
        return []

    items = data.get("items", [])
    logger.info(f"HH API: {len(items)} вакансий по '{keyword}' (город {city_id})")
    
    result = []
    for item in items:
        vacancy = {
            "id": f"hh_{item.get('id')}",
            "url": item.get("alternate_url"),
            "name": item.get("name"),
            "employer": {"name": (item.get("employer") or {}).get("name", "")},
            "area": {"name": (item.get("area") or {}).get("name", "")},
            "salary": item.get("salary"),
            "snippet": {
                "requirement": (item.get("snippet") or {}).get("requirement", "") or "",
                "responsibility": (item.get("snippet") or {}).get("responsibility", "") or "",
            },
            "description": "",
            "source": "hh.ru",
            "published_at": item.get("published_at", ""),
        }
        result.append(vacancy)
    return result


async def fetch_superjob(city_name: str, keyword: str, count: int = 20) -> List[Dict]:
    if not SUPERJOB_KEY:
        return []
    url = "https://api.superjob.ru/2.0/vacancies/"
    params = {
        "keyword": keyword,
        "town": city_name,
        "count": count,
        "order_field": "date",
        "order_direction": "desc",
    }
    headers = {"X-Api-App-Id": SUPERJOB_KEY}
    data, status = await fetch_json(url, params=params, headers=headers)
    
    if not data or not isinstance(data, dict):
        logger.warning(f"SuperJob недоступен (status={status}) для '{keyword}'")
        return []
    
    items = data.get("objects", [])
    logger.info(f"SuperJob: {len(items)} вакансий по '{keyword}' ({city_name})")
    
    result = []
    for it in items:
        payment_from = it.get("payment_from") or None
        payment_to = it.get("payment_to") or None
        salary = None
        if payment_from or payment_to:
            salary = {
                "from": payment_from,
                "to": payment_to,
                "currency": it.get("currency", "rub").upper()
            }
        result.append({
            "id": f"sj_{it.get('id')}",
            "url": it.get("link"),
            "name": it.get("profession"),
            "employer": {"name": it.get("firm_name", "")},
            "area": {"name": (it.get("town") or {}).get("title", city_name)},
            "salary": salary,
            "snippet": {
                "requirement": (it.get("candidat") or "")[:300],
                "responsibility": (it.get("work") or "")[:300],
            },
            "description": f"{it.get('candidat') or ''} {it.get('work') or ''}".strip(),
            "source": "superjob.ru",
            "published_at": it.get("date_published", ""),
        })
    return result


async def fetch_trudvsem(keyword: str, per_page: int = 20) -> List[Dict]:
    url = "https://opendata.trudvsem.ru/api/v1/vacancies"
    params = {"text": keyword, "offset": 0, "limit": per_page}
    headers = {"User-Agent": HH_USER_AGENT}
    data, status = await fetch_json(url, params=params, headers=headers)
    
    if not data or not isinstance(data, dict):
        logger.warning(f"Trudvsem недоступен (status={status}) для '{keyword}'")
        return []
    
    items = (data.get("results") or {}).get("vacancies") or []
    logger.info(f"Trudvsem: {len(items)} вакансий по '{keyword}'")
    
    result = []
    for it in items:
        v = it.get("vacancy", it)
        salary = None
        if v.get("salary_min") or v.get("salary_max"):
            salary = {
                "from": v.get("salary_min") or None,
                "to": v.get("salary_max") or None,
                "currency": "RUR"
            }
        result.append({
            "id": f"tv_{v.get('id')}",
            "url": v.get("vac_url"),
            "name": v.get("job-name"),
            "employer": {"name": (v.get("company") or {}).get("name", "")},
            "area": {"name": (v.get("region") or {}).get("name", "")},
            "salary": salary,
            "snippet": {
                "requirement": (v.get("requirements") or v.get("qualification") or "")[:300],
                "responsibility": (v.get("duty") or "")[:300],
            },
            "description": " ".join(filter(None, [
                v.get("duty"), v.get("requirements"), v.get("qualification")
            ])),
            "source": "trudvsem.ru",
            "published_at": v.get("creation-date", ""),
        })
    return result


DEMO_VACANCIES = [
    {
        "id": "demo_1",
        "url": "https://hh.ru/vacancy/123",
        "name": "Коммерческий директор (нефтепродукты)",
        "employer": {"name": "ООО НефтеТрейд"},
        "area": {"name": "Волгоград"},
        "salary": {"from": 200000, "to": 250000, "currency": "RUR"},
        "snippet": {"requirement": "Опыт управления отделом продаж ГСМ от 5 лет", "responsibility": "Руководство отделом"},
        "description": "Управление продажами нефтепродуктов, работа с НПЗ, логистика.",
        "source": "demo",
        "published_at": datetime.now().isoformat(),
    },
    {
        "id": "demo_2",
        "url": "https://hh.ru/vacancy/456",
        "name": "Директор по продажам (ГСМ)",
        "employer": {"name": "ООО Топливный Альянс"},
        "area": {"name": "Москва"},
        "salary": {"from": 180000, "to": 220000, "currency": "RUR"},
        "snippet": {"requirement": "Опыт руководства отделом продаж от 3 лет, знание рынка нефтепродуктов", "responsibility": "Развитие клиентской базы"},
        "description": "Развитие направления оптовых продаж, контроль дебиторской задолженности.",
        "source": "demo",
        "published_at": datetime.now().isoformat(),
    },
    {
        "id": "demo_3",
        "url": "https://hh.ru/vacancy/789",
        "name": "Руководитель отдела продаж (B2B)",
        "employer": {"name": "ООО НефтеСнаб"},
        "area": {"name": "Казань"},
        "salary": {"from": 150000, "to": 180000, "currency": "RUR"},
        "snippet": {"requirement": "Опыт работы в оптовых продажах, управление командой", "responsibility": "Построение отдела продаж"},
        "description": "Организация работы отдела продаж, поиск новых клиентов, работа с контрактами.",
        "source": "demo",
        "published_at": datetime.now().isoformat(),
    },
]


# =====================================================================
#  УМНЫЙ ПАРСЕР ПОИСКОВОГО ЗАПРОСА
# =====================================================================

@dataclass
class SearchQuery:
    raw_text: str
    keywords: List[str]
    cities: List[str]
    salary_min: Optional[int]
    remote: bool
    exclude: List[str]
    
    def build_hh_text(self) -> str:
        parts = []
        for kw in self.keywords:
            parts.append(f'"{kw}"')
        if not parts:
            parts = [self.raw_text]
        text = " AND ".join(parts)
        for ex in self.exclude:
            text += f' NOT "{ex}"'
        return text
    
    def __str__(self):
        parts = [f"🔍 Ключевые слова: {', '.join(self.keywords)}"]
        if self.cities:
            parts.append(f"📍 Города: {', '.join(self.cities)}")
        if self.salary_min:
            parts.append(f"💰 От {self.salary_min:,} ₽")
        if self.remote:
            parts.append("🏠 Удалённо")
        if self.exclude:
            parts.append(f"🚫 Исключить: {', '.join(self.exclude)}")
        return "\n".join(parts)


def parse_search_query(text: str) -> SearchQuery:
    text_lower = text.lower().strip()
    
    if text_lower.startswith("/search"):
        text_lower = text_lower[7:].strip()
        text = text[7:].strip()
    
    query = SearchQuery(
        raw_text=text,
        keywords=[],
        cities=[],
        salary_min=None,
        remote=False,
        exclude=[],
    )
    
    # Парсим зарплату
    salary_patterns = [
        r'(?:от\s+)?(\d{2,3})\s*к(?:\b|$)',
        r'(?:от\s+)?(\d{3})\s*тыс(?:\w*)',
        r'(?:от\s+)?(\d{5,6})(?:\s*руб|\s*₽)?',
    ]
    for pattern in salary_patterns:
        match = re.search(pattern, text_lower)
        if match:
            sal = int(match.group(1))
            if sal < 1000:
                sal *= 1000
            query.salary_min = sal
            text_lower = re.sub(pattern, '', text_lower, count=1)
            break
    
    # Парсим удалёнку
    remote_words = ['удален', 'удалён', 'remote', 'удаленка', 'удалёнка', 'дистанц', 'на дому']
    for rw in remote_words:
        if rw in text_lower:
            query.remote = True
            text_lower = text_lower.replace(rw, '')
            break
    
    # Парсим города
    known_cities = list(PROFILE["filters"]["cities"].keys()) + [
        "москва", "питер", "спб", "санкт-петербург", "волгоград", "казань",
        "нижний новгород", "уфа", "екатеринбург", "новосибирск", "краснодар",
        "ростов", "самара", "омск", "челябинск"
    ]
    for city in known_cities:
        if city.lower() in text_lower:
            query.cities.append(city)
            text_lower = text_lower.replace(city.lower(), '')
    
    # Парсим исключения
    exclude_patterns = [
        r'(?:не|без|кроме|исключая)\s+(\w+)',
        r'(?:без|кроме)\s+(\w+)',
    ]
    for pattern in exclude_patterns:
        for match in re.finditer(pattern, text_lower):
            query.exclude.append(match.group(1))
            text_lower = text_lower.replace(match.group(0), '')
    
    # Парсим ключевые слова
    stop_words = {
        'ищу', 'найди', 'поиск', 'вакансии', 'работу', 'ищи', 'покажи', 'вакансия',
        'нужна', 'хочу', 'работа', 'по', 'в', 'на', 'с', 'для', 'как', 'что', 'где',
        'когда', 'кто', 'зарплат', 'зп', 'оклад', 'рублей', 'руб', '₽', 'тысяч',
        'миллион', 'от', 'до', 'и', 'или',
    }
    
    words = re.findall(r'\b[а-яa-zё]+\b', text_lower)
    phrases = re.findall(r'\b[а-яa-zё]+(?:\s+[а-яa-zё]+){1,3}\b', text_lower)
    
    candidates = phrases + words
    keywords = []
    for cand in candidates:
        cand = cand.strip()
        if len(cand) < 3:
            continue
        if cand in stop_words:
            continue
        if cand in [k.lower() for k in keywords]:
            continue
        keywords.append(cand)
    
    if not keywords and text.strip():
        keywords = [text.strip()]
    
    query.keywords = keywords[:5]
    return query


# =====================================================================
#  СКОРИНГ
# =====================================================================

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


def calc_keyword_score(text: str, keywords: List[str]) -> float:
    text = text.lower()
    if not keywords:
        return 0.0
    matches = sum(1 for kw in keywords if kw.lower() in text)
    return min(matches / max(len(keywords) * 0.3, 1), 1.0) * 10


def calc_salary_score(salary: Optional[Dict], min_sal: Optional[int] = None) -> float:
    if not salary:
        return 5.0
    from_ = salary.get("from")
    to_ = salary.get("to")
    min_sal = min_sal or PROFILE["candidate"]["salary_min"]
    if from_ and from_ >= min_sal:
        return 10.0
    if to_ and to_ >= min_sal:
        return 8.0
    if from_ and from_ >= min_sal * 0.8:
        return 6.0
    if to_ and to_ >= min_sal * 0.8:
        return 4.0
    return 2.0


def calc_location_score(city: str, allowed_cities: Optional[List[str]] = None) -> float:
    city = (city or "").lower()
    allowed = allowed_cities or list(PROFILE["filters"]["cities"].keys())
    for allowed_city in allowed:
        if allowed_city.lower() in city:
            return 10.0
    return 0.0


def calc_experience_score(text: str) -> float:
    text = text.lower()
    score = sum(2 for kw in ["опыт", "лет", "руководитель", "директор"] if kw in text)
    if re.search(r"\b[5-9]\b|\b1[0-9]\b|\b20\b", text):
        score += 3
    return min(score, 10.0)


def calc_skills_score(text: str) -> float:
    return calc_keyword_score(text, PROFILE["candidate"]["key_skills"])


# =====================================================================
#  DEEPSEEK СКОРИНГ
# =====================================================================

async def deepseek_score_vacancy(vacancy: Dict, query: Optional[SearchQuery] = None) -> Optional[float]:
    if not deepseek_available:
        return None
    
    name = vacancy.get("name", "")
    employer = (vacancy.get("employer") or {}).get("name", "")
    description = vacancy.get("description", "")
    snippet_req = (vacancy.get("snippet") or {}).get("requirement", "")
    snippet_resp = (vacancy.get("snippet") or {}).get("responsibility", "")
    city = (vacancy.get("area") or {}).get("name", "")
    salary = vacancy.get("salary")
    
    salary_str = "не указана"
    if salary:
        parts = []
        if salary.get("from"):
            parts.append(f"от {salary['from']:,}")
        if salary.get("to"):
            parts.append(f"до {salary['to']:,}")
        salary_str = " ".join(parts)
        if salary.get("currency"):
            salary_str += f" {salary['currency']}"
    
    description = description[:800] if description else ""
    snippet_req = snippet_req[:400] if snippet_req else ""
    snippet_resp = snippet_resp[:400] if snippet_resp else ""
    
    user_query_str = ""
    if query and query.keywords:
        user_query_str = f"\nЗАПРОС ПОЛЬЗОВАТЕЛЯ: {' '.join(query.keywords)}"
        if query.cities:
            user_query_str += f"\nГорода пользователя: {', '.join(query.cities)}"
        if query.salary_min:
            user_query_str += f"\nМин. зарплата пользователя: {query.salary_min:,} ₽"
        if query.remote:
            user_query_str += "\nПользователь ищет удалённую работу"
    
    prompt = f"""Оцени релевантность вакансии для кандидата. Ответь ТОЛЬКО числом от 0 до 100, без объяснений.

ПРОФИЛЬ КАНДИДАТА:
- Целевые должности: коммерческий директор, руководитель отдела продаж, директор по продажам
- Отрасль: нефтепродукты, ГСМ, топливо, нефть, нефтегаз, АЗС, логистика нефтепродуктов
- Ключевые навыки: управление продажами, B2B, работа с НПЗ, нефтетрейдинг, логистика ГСМ
- Минимальная зарплата: 150000 рублей
- Города: Волгоград, Москва, Казань, Нижний Новгород, Уфа, Санкт-Петербург
{user_query_str}

ВАКАНСИЯ:
- Название: {name}
- Компания: {employer}
- Город: {city}
- Зарплата: {salary_str}
- Требования: {snippet_req}
- Обязанности: {snippet_resp}
- Описание: {description}

ШКАЛА ОЦЕНКИ:
100 = Идеально: руководящая должность в нефтегазе/ГСМ, подходящий город, зарплата >= 150к
80-99 = Отлично: руководящая должность в смежной отрасли (B2B, логистика), подходящий город
60-79 = Хорошо: руководящая должность, но другая отрасль ИЛИ неподходящий город
40-59 = Средне: не руководящая должность, но связано с продажами/нефтегазом
20-39 = Слабо: далёкая отрасль или низкая зарплата
0-19 = Не подходит: стажёр, junior, розница, IT, совсем другая сфера

ВАЖНО: Если в названии есть "коммерческий директор", "директор по продажам" или "руководитель отдела продаж" — это плюс. Если компания явно из нефтегаза (НПЗ, ГСМ, АЗС, нефтетрейдинг) — это большой плюс. Если город не из списка — минус. Если запрос пользователя указан — учитывай его приоритетно.

Оценка (только число 0-100):"""

    try:
        response = await asyncio.wait_for(
            deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,
                max_tokens=10
            ),
            timeout=20.0
        )
        content = response.choices[0].message.content.strip()
        
        match = re.search(r'\b(\d{1,3})\b', content)
        if match:
            score = float(match.group(1))
            score = min(max(score, 0), 100)
            logger.info(f"DeepSeek скор для '{name[:50]}': {score}")
            return score
            
    except asyncio.TimeoutError:
        logger.warning(f"DeepSeek timeout для вакансии '{name[:50]}'")
    except Exception as e:
        logger.warning(f"DeepSeek scoring error: {e}")
    
    return None


async def score_vacancy(vacancy: Dict, query: Optional[SearchQuery] = None) -> VacancyScore:
    text = " ".join(filter(None, [
        vacancy.get("name", ""),
        vacancy.get("description", ""),
        (vacancy.get("snippet") or {}).get("requirement", ""),
        (vacancy.get("snippet") or {}).get("responsibility", ""),
    ])).lower()
    
    w = PROFILE["scoring"]["weights"]
    
    search_keywords = query.keywords if query else PROFILE["filters"]["keywords"]
    search_cities = query.cities if query else list(PROFILE["filters"]["cities"].keys())
    min_salary = query.salary_min if query else None
    
    role = calc_keyword_score(text, search_keywords)
    industry = calc_keyword_score(text, PROFILE["filters"]["industry_keywords"])
    salary = calc_salary_score(vacancy.get("salary"), min_salary)
    location = calc_location_score((vacancy.get("area") or {}).get("name", ""), search_cities)
    experience = calc_experience_score(text)
    skills = calc_skills_score(text)
    
    keyword_total = (
        role * w["role_fit"] +
        industry * w["industry_match"] +
        salary * w["salary_match"] +
        location * w["location_match"] +
        experience * w["experience_match"] +
        skills * w["skills_match"]
    ) * 10
    
    for word in PROFILE["filters"]["exclude_words"]:
        if word.lower() in text:
            return VacancyScore(
                total=0.0,
                role_fit=round(role, 1),
                industry_match=round(industry, 1),
                salary_match=round(salary, 1),
                location_match=round(location, 1),
                experience_match=round(experience, 1),
                skills_match=round(skills, 1),
                verdict="SKIP",
                reasoning=f"Исключено: найдено слово '{word}'"
            )
    
    if query and query.exclude:
        for word in query.exclude:
            if word.lower() in text:
                return VacancyScore(
                    total=0.0,
                    role_fit=round(role, 1),
                    industry_match=round(industry, 1),
                    salary_match=round(salary, 1),
                    location_match=round(location, 1),
                    experience_match=round(experience, 1),
                    skills_match=round(skills, 1),
                    verdict="SKIP",
                    reasoning=f"Исключено по запросу: '{word}'"
                )
    
    deep_score = await deepseek_score_vacancy(vacancy, query)
    
    if deep_score is not None:
        final_score = deep_score * 0.8 + keyword_total * 0.2
        
        if final_score >= 75:
            verdict = "STRONG_MATCH"
            reason = f"Отличное совпадение (AI: {deep_score})"
        elif final_score >= PROFILE["scoring"]["min_score"]:
            verdict = "MATCH"
            reason = f"Хорошее совпадение (AI: {deep_score})"
        elif final_score >= 35:
            verdict = "WEAK_MATCH"
            reason = f"Среднее совпадение (AI: {deep_score})"
        else:
            verdict = "SKIP"
            reason = f"Низкий скор (AI: {deep_score})"
        
        return VacancyScore(
            total=round(final_score, 1),
            role_fit=round(role, 1),
            industry_match=round(industry, 1),
            salary_match=round(salary, 1),
            location_match=round(location, 1),
            experience_match=round(experience, 1),
            skills_match=round(skills, 1),
            verdict=verdict,
            reasoning=reason
        )
    
    if keyword_total >= 75:
        verdict = "STRONG_MATCH"
        reason = "Отличное совпадение (keyword)"
    elif keyword_total >= PROFILE["scoring"]["min_score"]:
        verdict = "MATCH"
        reason = "Хорошее совпадение (keyword)"
    elif keyword_total >= 35:
        verdict = "WEAK_MATCH"
        reason = "Среднее совпадение (keyword)"
    else:
        verdict = "SKIP"
        reason = "Низкий скор (keyword)"
    
    return VacancyScore(
        total=round(keyword_total, 1),
        role_fit=round(role, 1),
        industry_match=round(industry, 1),
        salary_match=round(salary, 1),
        location_match=round(location, 1),
        experience_match=round(experience, 1),
        skills_match=round(skills, 1),
        verdict=verdict,
        reasoning=reason
    )


# =====================================================================
#  ФОРМАТИРОВАНИЕ И UI
# =====================================================================

def format_vacancy(v: Dict, score: VacancyScore, idx: int, total: int) -> Tuple[str, InlineKeyboardMarkup]:
    sal = v.get("salary")
    if sal:
        sal_from = sal.get("from")
        sal_to = sal.get("to")
        sal_cur = sal.get("currency", "")
        parts = []
        if sal_from:
            parts.append(f"от {sal_from:,}")
        if sal_to:
            parts.append(f"до {sal_to:,}")
        sal_str = " ".join(parts)
        if sal_cur:
            sal_str += f" {sal_cur}"
        sal_str = sal_str.strip() or "не указана"
    else:
        sal_str = "не указана"
    
    url = v.get("url", "")
    src = v.get("source", "")
    src_tag = f" ({src})" if src and src != "demo" else ""
    
    verdict_emoji = {
        "STRONG_MATCH": "🟢",
        "MATCH": "🟡",
        "WEAK_MATCH": "🟠",
        "SKIP": "🔴",
    }.get(score.verdict, "⚪")
    
    msg = (
        f"{verdict_emoji} <b>{idx}/{total}</b>{src_tag}\n\n"
        f"<b>{v.get('name', '')}</b>\n"
        f"🏢 {v.get('employer', {}).get('name', 'Не указано')}\n"
        f"📍 {v.get('area', {}).get('name', 'Не указано')}\n"
        f"💰 {sal_str}\n\n"
        f"📊 <b>Скор: {score.total}</b> ({score.verdict})\n"
        f"• Роль: {score.role_fit}/10\n"
        f"• Индустрия: {score.industry_match}/10\n"
        f"• Зарплата: {score.salary_match}/10\n"
        f"• Локация: {score.location_match}/10\n"
        f"• Опыт: {score.experience_match}/10\n"
        f"• Навыки: {score.skills_match}/10\n\n"
        f"💬 {score.reasoning}"
    )
    
    kb_buttons = [
        [
            InlineKeyboardButton("👍 Интересно", callback_data=f"like:{v['id']}"),
            InlineKeyboardButton("👎 Пропустить", callback_data=f"dislike:{v['id']}"),
            InlineKeyboardButton("📝 Отклик", callback_data=f"apply:{v['id']}")
        ],
        []
    ]
    if url:
        kb_buttons[1].append(InlineKeyboardButton("🔗 Открыть", url=url))
    
    if idx < total:
        kb_buttons[1].append(InlineKeyboardButton("➡️ Далее", callback_data="next"))
    else:
        kb_buttons[1].append(InlineKeyboardButton("🔄 Ещё вакансии", callback_data="more"))
    
    return msg, InlineKeyboardMarkup(kb_buttons)


# =====================================================================
#  ОБРАБОТЧИКИ КОМАНД
# =====================================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я бот для поиска вакансий.\n\n"
        "Ищу позиции коммерческого директора и руководителей отдела продаж "
        "в нефтегазовой отрасли.\n\n"
        "<b>Команды:</b>\n"
        "/search — найти вакансии по профилю\n"
        "/search [запрос] — гибкий поиск (примеры ниже)\n"
        "/stats — статистика\n"
        "/clear — очистить историю\n"
        "/help — помощь\n\n"
        "<b>Примеры запросов:</b>\n"
        "• <code>/search коммерческий директор москва</code>\n"
        "• <code>/search руководитель продаж удаленно от 100к</code>\n"
        "• <code>/search директор по продажам не стажер</code>\n"
        "• Просто напиши: <i>\"ищу работу в нефтегазе\"</i>",
        parse_mode="HTML"
    )


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global search_in_progress
    if search_in_progress:
        await update.message.reply_text("⏳ Поиск уже выполняется, подождите...")
        return
    
    args = context.args
    if args:
        query_text = " ".join(args)
        query = parse_search_query(query_text)
        await update.message.reply_text(
            f"🔍 Ищу по твоему запросу:\n{query}\n\nЭто займёт около минуты..."
        )
    else:
        query = None
        await update.message.reply_text("🔍 Начинаю поиск вакансий по профилю, это займёт около минуты...")
    
    await do_search(update, context, query=query)


async def do_search(update: Update, context: ContextTypes.DEFAULT_TYPE, query: Optional[SearchQuery] = None, force: bool = False):
    global search_in_progress
    if search_in_progress:
        return
    search_in_progress = True
    chat_id = update.effective_chat.id
    
    try:
        if query and query.cities:
            city_map = PROFILE["filters"]["cities"]
            cities = []
            for city_name in query.cities:
                city_lower = city_name.lower()
                for prof_city, prof_id in city_map.items():
                    if prof_city.lower() in city_lower or city_lower in prof_city.lower():
                        cities.append((prof_city, prof_id))
                        break
                else:
                    cities.append((city_name, "1"))
            if not cities:
                cities = list(city_map.items())
        else:
            cities = list(PROFILE["filters"]["cities"].items())
        
        if query and query.keywords:
            keywords = query.keywords
        else:
            keywords = PROFILE["filters"]["keywords"]
        
        all_vacancies: List[Dict] = []
        total_combinations = 0
        
        for city_name, city_id in cities:
            for kw in keywords[:3]:
                total_combinations += 1
                if total_combinations > 15:
                    break
                
                if query:
                    hh_text = query.build_hh_text()
                else:
                    hh_text = kw
                
                try:
                    vacs = await fetch_hh_api(city_id, hh_text, per_page=HH_PER_PAGE)
                    all_vacancies.extend(vacs)
                except Exception as e:
                    logger.error(f"HH error for {city_name}/{kw}: {e}")
                await asyncio.sleep(random.uniform(1.0, 2.0))
                
                if SUPERJOB_KEY:
                    try:
                        sj = await fetch_superjob(city_name, kw, count=SJ_COUNT)
                        all_vacancies.extend(sj)
                    except Exception as e:
                        logger.error(f"SJ error for {city_name}/{kw}: {e}")
                    await asyncio.sleep(random.uniform(0.5, 1.5))
                
                try:
                    tv = await fetch_trudvsem(kw, per_page=TV_PER_PAGE)
                    all_vacancies.extend(tv)
                except Exception as e:
                    logger.error(f"TV error for {kw}: {e}")
                await asyncio.sleep(random.uniform(0.5, 1.5))
            
            if total_combinations > 15:
                break
            await asyncio.sleep(1)

        if not all_vacancies:
            logger.info("Реальных вакансий не найдено, показываем демо")
            clear_seen()
            demo_list = list(DEMO_VACANCIES)
            
            await context.bot.send_message(
                chat_id,
                "⚠️ <b>Реальные вакансии недоступны</b>\n\n"
                "Проверьте:\n"
                "• Заданы ли HH_CLIENT_ID / HH_CLIENT_SECRET\n"
                f"• Пул прокси: {proxy_pool.size} рабочих\n"
                "• Подключение к интернету\n\n"
                "Пока показываю демонстрационные вакансии.",
                parse_mode="HTML"
            )
            all_vacancies = demo_list

        seen_ids = set()
        seen_sigs = set()
        unique = []
        for v in all_vacancies:
            vid = v["id"]
            if vid in seen_ids:
                continue
            seen_ids.add(vid)
            
            sig = f"{v.get('name','').lower().strip()}|{v.get('employer',{}).get('name','').lower().strip()}"
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            
            if not force and is_seen(vid):
                continue
            if not force and is_seen_signature(sig):
                continue
            
            unique.append(v)
            
            if len(unique) >= MAX_VACANCIES_TOTAL:
                break

        logger.info(f"После дедупликации: {len(unique)} уникальных вакансий")

        if not unique:
            await context.bot.send_message(
                chat_id,
                "📭 <b>Новых вакансий не найдено</b>\n\n"
                "Все доступные вакансии уже просмотрены.\n"
                "Попробуйте:\n"
                "• Подождать появления новых вакансий\n"
                "• Изменить критерии в profile.yaml\n"
                "• Использовать /clear для сброса истории",
                parse_mode="HTML"
            )
            return

        scored = []
        for v in unique:
            score = await score_vacancy(v, query)
            scored.append((v, score))
            await asyncio.sleep(0.2)
        
        scored.sort(key=lambda x: x[1].total, reverse=True)
        
        good = [(v, s) for v, s in scored if s.verdict in ("MATCH", "STRONG_MATCH")]
        
        if len(good) < MIN_VACANCIES_TO_SHOW:
            weak = [(v, s) for v, s in scored if s.verdict == "WEAK_MATCH"]
            good.extend(weak)
        
        if len(good) < MIN_VACANCIES_TO_SHOW:
            remaining = [(v, s) for v, s in scored if s.verdict not in ("MATCH", "STRONG_MATCH", "WEAK_MATCH") and s.verdict != "SKIP"]
            remaining.sort(key=lambda x: x[1].total, reverse=True)
            good.extend(remaining[:MIN_VACANCIES_TO_SHOW - len(good)])
        
        good.sort(key=lambda x: x[1].total, reverse=True)
        good = good[:VACANCIES_PER_PAGE]
        
        if not good:
            await context.bot.send_message(
                chat_id,
                "📭 <b>Нет вакансий, соответствующих твоему профилю.</b>\n\n"
                f"Проверено: <b>{len(scored)}</b> вакансий\n"
                f"Порог скоринга: <b>{PROFILE['scoring']['min_score']}</b>\n"
                f"DeepSeek: {'✅' if deepseek_available else '❌'}\n\n"
                "Попробуй:\n"
                "• Снизить min_score в profile.yaml\n"
                "• Добавить города\n"
                "• Подождать новых вакансий",
                parse_mode="HTML"
            )
            return

        context.chat_data["vacancies"] = good
        context.chat_data["index"] = 0
        context.chat_data["search_query"] = query
        
        if good:
            await show_vacancy(update, context, 0)
        else:
            await context.bot.send_message(chat_id, "❌ Не удалось подобрать вакансии.")
            
    except Exception as e:
        logger.exception("Ошибка при поиске")
        await context.bot.send_message(chat_id, f"❌ Ошибка при поиске: {str(e)[:200]}")
    finally:
        search_in_progress = False


async def show_vacancy(update: Update, context: ContextTypes.DEFAULT_TYPE, idx: int):
    vacancies = context.chat_data.get("vacancies", [])
    if not vacancies or idx >= len(vacancies):
        text = "✅ Все вакансии просмотрены.\n\nНажми /search для нового поиска."
        try:
            if update.callback_query:
                await update.callback_query.edit_message_text(text)
            else:
                await update.message.reply_text(text)
        except Exception:
            pass
        return
    
    v, s = vacancies[idx]
    msg, kb = format_vacancy(v, s, idx + 1, len(vacancies))
    
    try:
        if update.callback_query:
            await update.callback_query.edit_message_text(msg, reply_markup=kb, parse_mode="HTML")
        else:
            await update.message.reply_text(msg, reply_markup=kb, parse_mode="HTML")
    except Exception as e:
        logger.warning(f"Ошибка отображения вакансии: {e}")
        try:
            await context.bot.send_message(
                update.effective_chat.id,
                msg,
                reply_markup=kb,
                parse_mode="HTML"
            )
        except Exception as e2:
            logger.error(f"Не удалось отправить вакансию: {e2}")
    
    context.chat_data["index"] = idx


async def more_vacancies(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = context.chat_data.get("search_query")
    await update.callback_query.answer("Загружаю ещё...")
    await do_search(update, context, query=query, force=True)


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data == "next":
        idx = context.chat_data.get("index", 0) + 1
        await show_vacancy(update, context, idx)
        return
    
    if data == "more":
        await more_vacancies(update, context)
        return
    
    if ":" not in data:
        return
    
    action, vid = data.split(":", 1)
    vacancies = context.chat_data.get("vacancies", [])
    target = None
    for v, s in vacancies:
        if v["id"] == vid:
            target = (v, s)
            break
    
    if not target:
        await query.edit_message_text("⚠️ Вакансия не найдена в текущей сессии.")
        return
    
    v, s = target
    
    if action == "like":
        try:
            await query.edit_message_text(
                query.message.text + "\n\n✅ <b>Отмечено:</b> интересно",
                reply_markup=None,
                parse_mode="HTML"
            )
        except Exception:
            pass
    elif action == "dislike":
        try:
            await query.edit_message_text(
                query.message.text + "\n\n❌ <b>Отмечено:</b> не подходит",
                reply_markup=None,
                parse_mode="HTML"
            )
        except Exception:
            pass
    elif action == "apply":
        try:
            await query.edit_message_text(
                query.message.text + "\n\n📝 <b>Отмечено для отклика</b>",
                reply_markup=None,
                parse_mode="HTML"
            )
        except Exception:
            pass
        add_application(vid, v.get("name"), v.get("employer", {}).get("name"), s.total)
    
    mark_seen(vid, v.get("name"), v.get("employer", {}).get("name"), s.total)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("DELETE FROM seen_vacancies")
        c.execute("DELETE FROM seen_signatures")
        conn.commit()
        await update.message.reply_text("🗑️ История просмотров очищена. Можно искать заново!")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")
    finally:
        conn.close()


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    seen, apps = get_stats()
    hh_status = "✅ есть" if hh_auth.token else "❌ нет"
    await update.message.reply_text(
        f"📊 <b>Статистика</b>\n\n"
        f"• Просмотрено вакансий: <b>{seen}</b>\n"
        f"• Откликов: <b>{apps}</b>\n"
        f"• Прокси в пуле: <b>{proxy_pool.size}</b>\n"
        f"• HH токен: {hh_status}\n"
        f"• DeepSeek: {'✅' if deepseek_available else '❌'}\n"
        f"• Выдача за раз: <b>{VACANCIES_PER_PAGE}</b>",
        parse_mode="HTML"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 <b>Справка</b>\n\n"
        "<b>/search</b> — найти вакансии по профилю (10 штук)\n"
        "<b>/search [запрос]</b> — гибкий поиск. Примеры:\n"
        "  • <code>/search коммерческий директор москва</code>\n"
        "  • <code>/search руководитель продаж удаленно от 100к</code>\n"
        "  • <code>/search директор по продажам не стажер</code>\n\n"
        "<b>/stats</b> — статистика\n"
        "<b>/clear</b> — очистить историю просмотров\n"
        "<b>/help</b> — эта справка\n\n"
        "Также можно писать свободно:\n"
        "• <i>\"ищу работу в нефтегазе\"</i>\n"
        "• <i>\"покажи вакансии коммерческого директора\"</i>\n"
        "• <i>\"статистика\"</i>",
        parse_mode="HTML"
    )


# =====================================================================
#  DeepSeek и обработка текста
# =====================================================================

async def deepseek_chat(message: str) -> str:
    if not deepseek_available:
        return (
            "Извините, ИИ-диалог недоступен (не задан DEEPSEEK_API_KEY).\n"
            "Используйте команды: /search, /stats, /help"
        )
    
    system_prompt = (
        "Ты — профессиональный карьерный консультант. Ты помогаешь найти работу "
        "коммерческого директора или руководителя отдела продаж в нефтегазовой отрасли. "
        "Отвечай кратко, по делу, на русском языке. Если спрашивают о вакансиях — "
        "предложи команду /search. Если вопрос не по теме — вежливо укажи на специализацию."
    )
    
    try:
        response = await asyncio.wait_for(
            deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=500
            ),
            timeout=20.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"DeepSeek error: {e}")
        return "⚠️ Ошибка при обращении к ИИ. Попробуйте позже или используйте команды."


def parse_natural(text: str) -> Optional[str]:
    text = text.lower().strip()
    
    search_triggers = [
        "найди", "поиск", "вакансии", "работу", "ищи", "покажи", "вакансия",
        "ищу", "нужна работа", "хочу работать", "подбери", "посоветуй",
        "вакансии по", "работа в", "вакансия на",
    ]
    if any(t in text for t in search_triggers):
        return "search"
    
    stat_triggers = ["статистика", "стат", "сколько", "цифры", "просмотрено", "откликов"]
    if any(t in text for t in stat_triggers):
        return "stats"
    
    help_triggers = ["помощь", "help", "команды", "что умеешь", "привет", "здравствуй", "ку", "хай"]
    if any(t in text for t in help_triggers):
        return "help"
    
    clear_triggers = ["очисти", "сбрось", "clear", "удали историю", "забудь"]
    if any(t in text for t in clear_triggers):
        return "clear"
    
    return None


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    action = parse_natural(text)
    
    if action == "search":
        query = parse_search_query(text)
        await update.message.reply_text(
            f"🔍 Ищу по твоему запросу:\n{query}\n\nЭто займёт около минуты..."
        )
        await do_search(update, context, query=query)
        return
    elif action == "stats":
        await stats_command(update, context)
        return
    elif action == "help":
        await help_command(update, context)
        return
    elif action == "clear":
        await clear_command(update, context)
        return

    await update.message.reply_text("🤔 Думаю...")
    reply = await deepseek_chat(text)
    await update.message.reply_text(reply)


# =====================================================================
#  FastAPI и запуск
# =====================================================================

app = FastAPI()
telegram_app: Optional[Application] = None


@app.post("/webhook")
async def webhook(request: Request):
    global telegram_app
    if telegram_app is None:
        return {"error": "not ready"}
    try:
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        await telegram_app.process_update(update)
        return {"ok": True}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"error": str(e)}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "proxies": proxy_pool.size,
        "hh_token": bool(hh_auth.token),
        "deepseek": deepseek_available,
        "search_in_progress": search_in_progress
    }


@app.get("/")
async def root():
    return {
        "status": "alive",
        "deepseek": deepseek_available,
        "proxies": proxy_pool.size,
        "hh_token": bool(hh_auth.token),
        "timestamp": datetime.now().isoformat()
    }


async def shutdown():
    proxy_pool.stop()
    await proxy_pool.close()
    await close_session()
    if telegram_app:
        await telegram_app.stop()
    logger.info("Shutdown complete")


async def run():
    global telegram_app
    
    proxy_task = asyncio.create_task(proxy_pool.run())
    await asyncio.sleep(5)
    
    telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()
    telegram_app.add_handler(CommandHandler("start", start))
    telegram_app.add_handler(CommandHandler("search", search_command))
    telegram_app.add_handler(CommandHandler("stats", stats_command))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(CommandHandler("clear", clear_command))
    telegram_app.add_handler(CallbackQueryHandler(callback_handler))
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    await telegram_app.initialize()
    await telegram_app.start()

    if RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        await telegram_app.bot.set_webhook(url=webhook_url)
        logger.info(f"Webhook установлен: {webhook_url}")

    config = uvicorn.Config(app, host="0.0.0.0", port=10000, log_level="info")
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    except asyncio.CancelledError:
        logger.info("Сервер остановлен")
    finally:
        await shutdown()
        try:
            proxy_task.cancel()
            await proxy_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
