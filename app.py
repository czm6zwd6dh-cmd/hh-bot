# -*- coding: utf-8 -*-
import os
import sqlite3
import asyncio
import logging
import re
import json
import yaml
import aiohttp
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
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

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Переменные окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан")

RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")

# Константы
MAX_VACANCIES_PER_CYCLE = 5
DB_PATH = "vacancies.db"
PROFILE_PATH = "profile.yaml"

# ========== ПРОФИЛЬ ПО УМОЛЧАНИЮ ==========
DEFAULT_PROFILE = {
    "candidate": {
        "name": "Виктор Зинченко",
        "desired_positions": [
            "коммерческий директор",
            "руководитель отдела продаж",
            "директор по продажам",
        ],
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
            "Астана": "160",
        },
        "keywords": [
            "коммерческий директор",
            "директор по продажам",
            "руководитель отдела продаж",
        ],
        "industry_keywords": ["нефтепродукты", "ГСМ", "топливо", "нефть"],
        "exclude_words": ["стажёр", "junior", "стажер"],
    },
    "scoring": {
        "min_score": 35,
        "weights": {
            "role_fit": 0.30,
            "industry_match": 0.25,
            "salary_match": 0.15,
            "location_match": 0.10,
            "experience_match": 0.10,
            "skills_match": 0.10,
        },
    },
}


def load_profile() -> dict:
    """Загружает профиль из YAML, если нет – создаёт с DEFAULT_PROFILE."""
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки profile.yaml: {e}, создаём заново")
    # Если файла нет или он битый – создаём новый
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        yaml.dump(DEFAULT_PROFILE, f, allow_unicode=True, sort_keys=False)
    return DEFAULT_PROFILE.copy()


PROFILE = load_profile()


# ========== РАБОТА С БАЗОЙ ДАННЫХ ==========
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS seen_vacancies (
            id TEXT PRIMARY KEY,
            title TEXT,
            company TEXT,
            score REAL,
            seen_at TIMESTAMP
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS sent_vacancies (
            id TEXT PRIMARY KEY,
            title TEXT,
            company TEXT,
            score REAL,
            sent_at TIMESTAMP
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vacancy_id TEXT,
            title TEXT,
            company TEXT,
            score REAL,
            status TEXT,
            created_at TIMESTAMP
        )"""
    )
    conn.commit()
    conn.close()


init_db()


def is_seen(vacancy_id: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT 1 FROM seen_vacancies WHERE id=?", (vacancy_id,))
    res = c.fetchone() is not None
    conn.close()
    return res


def mark_seen(vacancy_id: str, title: str, company: str, score: float):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO seen_vacancies VALUES (?,?,?,?,?)",
        (vacancy_id, title, company, score, datetime.now()),
    )
    conn.commit()
    conn.close()


def mark_sent(vacancy_id: str, title: str, company: str, score: float):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO sent_vacancies VALUES (?,?,?,?,?)",
        (vacancy_id, title, company, score, datetime.now()),
    )
    conn.commit()
    conn.close()


def add_application(vacancy_id: str, title: str, company: str, score: float):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO applications (vacancy_id, title, company, score, status, created_at) VALUES (?,?,?,?,?,?)",
        (vacancy_id, title, company, score, "new", datetime.now()),
    )
    conn.commit()
    conn.close()


def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM seen_vacancies")
    seen = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM applications")
    apps = c.fetchone()[0]
    conn.close()
    return seen, apps


# ========== СКОРИНГ ==========
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
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return min(matches / max(len(keywords) * 0.3, 1), 1.0) * 10


def calc_salary_score(salary: Optional[dict]) -> float:
    if not salary:
        return 5.0
    from_ = salary.get("from")
    to_ = salary.get("to")
    min_sal = PROFILE["candidate"]["salary_min"]
    if from_ and from_ >= min_sal:
        return 10.0
    if to_ and to_ >= min_sal:
        return 8.0
    if from_ and from_ >= min_sal * 0.8:
        return 6.0
    if to_ and to_ >= min_sal * 0.8:
        return 4.0
    return 2.0


def calc_location_score(city: str) -> float:
    city_lower = city.lower()
    for allowed in PROFILE["filters"]["cities"]:
        if allowed.lower() in city_lower:
            return 10.0
    return 0.0


def calc_experience_score(text: str) -> float:
    text_lower = text.lower()
    score = sum(2 for kw in ["опыт", "лет", "руководитель", "директор"] if kw in text_lower)
    if re.search(r"\b[5-9]\b|\b10\b|\b20\b", text):
        score += 3
    return min(score, 10.0)


def calc_skills_score(text: str) -> float:
    return calc_keyword_score(text, PROFILE["candidate"]["key_skills"])


def score_vacancy(vacancy: dict) -> VacancyScore:
    text = (
        vacancy.get("name", "")
        + " "
        + vacancy.get("description", "")
        + " "
        + vacancy.get("snippet", {}).get("requirement", "")
    ).lower()
    weights = PROFILE["scoring"]["weights"]
    role = calc_keyword_score(text, PROFILE["filters"]["keywords"])
    industry = calc_keyword_score(text, PROFILE["filters"]["industry_keywords"])
    salary = calc_salary_score(vacancy.get("salary"))
    location = calc_location_score(vacancy.get("area", {}).get("name", ""))
    experience = calc_experience_score(text)
    skills = calc_skills_score(text)

    total = (
        role * weights["role_fit"]
        + industry * weights["industry_match"]
        + salary * weights["salary_match"]
        + location * weights["location_match"]
        + experience * weights["experience_match"]
        + skills * weights["skills_match"]
    )

    # Стоп-слова
    for word in PROFILE["filters"]["exclude_words"]:
        if word.lower() in text:
            total = 0
            break

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
        role_fit=round(role, 1),
        industry_match=round(industry, 1),
        salary_match=round(salary, 1),
        location_match=round(location, 1),
        experience_match=round(experience, 1),
        skills_match=round(skills, 1),
        verdict=verdict,
        reasoning=reasoning,
    )


# ========== ЗАПРОС К HH.RU (ОТКРЫТОЕ API) ==========
async def fetch_hh_vacancies(city_id: str, keyword: str, per_page: int = 10) -> List[dict]:
    url = "https://api.hh.ru/vacancies"
    params = {
        "area": city_id,
        "text": keyword,
        "per_page": per_page,
        "search_field": "name",
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    logger.error(f"HH API error: {resp.status}")
                    return []
                data = await resp.json()
                items = data.get("items", [])
                result = []
                for item in items:
                    result.append(
                        {
                            "id": item.get("id"),
                            "name": item.get("name"),
                            "employer": {"name": item.get("employer", {}).get("name")},
                            "area": {"name": item.get("area", {}).get("name")},
                            "salary": item.get("salary"),
                            "snippet": {
                                "requirement": item.get("snippet", {}).get("requirement", ""),
                                "responsibility": item.get("snippet", {}).get("responsibility", ""),
                            },
                            "description": item.get("description", ""),
                        }
                    )
                return result
        except Exception as e:
            logger.error(f"fetch_hh error: {e}")
            return []


# ========== ФОРМАТИРОВАНИЕ ВАКАНСИИ ==========
def format_vacancy(vacancy: dict, score: VacancyScore, idx: int, total: int):
    sal = vacancy.get("salary")
    if sal:
        sal_str = f"{sal.get('from', '')} - {sal.get('to', '')} {sal.get('currency', '')}".strip()
        if not sal_str:
            sal_str = "не указана"
    else:
        sal_str = "не указана"

    msg = (
        f"📌 {idx}/{total}\n\n"
        f"<b>{vacancy.get('name', 'Без названия')}</b>\n"
        f"🏢 {vacancy.get('employer', {}).get('name', 'Неизвестно')}\n"
        f"📍 {vacancy.get('area', {}).get('name', 'Не указан')}\n"
        f"💰 {sal_str}\n\n"
        f"📊 Скор: {score.total} ({score.verdict})\n"
        f"• Роль: {score.role_fit}/10\n"
        f"• Индустрия: {score.industry_match}/10\n"
        f"• Зарплата: {score.salary_match}/10\n"
        f"• Локация: {score.location_match}/10\n"
        f"• Опыт: {score.experience_match}/10\n"
        f"• Навыки: {score.skills_match}/10\n\n"
        f"💬 {score.reasoning}"
    )

    vid = vacancy.get("id")
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("👍", callback_data=f"like:{vid}"),
                InlineKeyboardButton("👎", callback_data=f"dislike:{vid}"),
                InlineKeyboardButton("📝", callback_data=f"apply:{vid}"),
            ],
            [InlineKeyboardButton("➡️ Далее", callback_data="next")],
        ]
    )
    return msg, keyboard


# ========== ОСНОВНЫЕ ОБРАБОТЧИКИ ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я ищу вакансии коммерческого директора.\n"
        "Используй /search или просто напиши «найди работу»."
    )


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Ищу вакансии, подождите...")
    # Запускаем поиск в фоне
    await do_search(update, context)


async def do_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    # Берём первые 3 города и 2 ключевых слова (чтобы не перегружать API)
    cities = list(PROFILE["filters"]["cities"].items())[:3]
    keywords = PROFILE["filters"]["keywords"][:2]

    all_vacancies = []
    for city_name, city_id in cities:
        for kw in keywords:
            vacs = await fetch_hh_vacancies(city_id, kw, per_page=5)
            all_vacancies.extend(vacs)
            await asyncio.sleep(0.5)  # пауза, чтобы не забанили

    # Удаляем дубликаты и уже просмотренные
    seen_ids = set()
    unique = []
    for v in all_vacancies:
        if v["id"] in seen_ids:
            continue
        seen_ids.add(v["id"])
        if is_seen(v["id"]):
            continue
        unique.append(v)

    if not unique:
        await context.bot.send_message(chat_id, "📭 Новых вакансий не найдено.")
        return

    # Скоринг
    scored = []
    for v in unique:
        s = score_vacancy(v)
        scored.append((v, s))

    # Сортируем по убыванию скор
    scored.sort(key=lambda x: x[1].total, reverse=True)

    # Оставляем только MATCH/STRONG, если есть, иначе топ-5
    good = [(v, s) for v, s in scored if s.verdict in ("MATCH", "STRONG_MATCH")]
    if not good:
        good = scored[:MAX_VACANCIES_PER_CYCLE]
    else:
        good = good[:MAX_VACANCIES_PER_CYCLE]

    # Сохраняем в контекст
    context.chat_data["vacancies"] = good
    context.chat_data["index"] = 0

    # Показываем первую
    await show_vacancy(update, context, 0)


async def show_vacancy(update: Update, context: ContextTypes.DEFAULT_TYPE, idx: int):
    vacancies = context.chat_data.get("vacancies", [])
    if not vacancies or idx >= len(vacancies):
        await update.effective_message.reply_text("✅ Все вакансии просмотрены.")
        return

    v, s = vacancies[idx]
    msg, kb = format_vacancy(v, s, idx + 1, len(vacancies))

    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=kb, parse_mode="HTML")
    else:
        await update.message.reply_text(msg, reply_markup=kb, parse_mode="HTML")

    context.chat_data["index"] = idx


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "next":
        idx = context.chat_data.get("index", 0) + 1
        await show_vacancy(update, context, idx)
        return

    if ":" in data:
        action, vid = data.split(":", 1)
        vacancies = context.chat_data.get("vacancies", [])
        target = None
        for v, s in vacancies:
            if v["id"] == vid:
                target = (v, s)
                break
        if not target:
            await query.edit_message_text("⚠️ Вакансия не найдена.")
            return

        v, s = target
        if action == "like":
            await query.edit_message_text(query.message.text + "\n\n✅ Отмечено: интересно", reply_markup=None)
        elif action == "dislike":
            await query.edit_message_text(query.message.text + "\n\n❌ Отмечено: не подходит", reply_markup=None)
        elif action == "apply":
            await query.edit_message_text(query.message.text + "\n\n📝 Отмечено для отклика", reply_markup=None)
            add_application(vid, v.get("name"), v.get("employer", {}).get("name"), s.total)

        # Помечаем как просмотренное
        mark_seen(vid, v.get("name"), v.get("employer", {}).get("name"), s.total)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    seen, apps = get_stats()
    await update.message.reply_text(f"📊 Статистика:\n• Просмотрено: {seen}\n• Откликов: {apps}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 Команды:\n"
        "/search – поиск вакансий\n"
        "/stats – статистика\n"
        "/help – помощь\n\n"
        "Также можно писать:\n"
        "• «найди работу» – поиск\n"
        "• «привет» – справка\n"
        "• «статистика» – статистика"
    )


# ========== ЭВРИСТИЧЕСКИЙ ПАРСЕР (БЕЗ AI) ==========
def parse_natural(text: str) -> Optional[str]:
    text = text.lower().strip()
    if any(w in text for w in ["привет", "здравствуй", "ку", "hi", "hello"]):
        return "help"
    if any(w in text for w in ["найди", "поиск", "вакансии", "работу", "ищи", "найти"]):
        return "search"
    if any(w in text for w in ["статистика", "стат", "сколько"]):
        return "stats"
    return None


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    action = parse_natural(text)
    if action == "search":
        await search_command(update, context)
    elif action == "stats":
        await stats_command(update, context)
    elif action == "help":
        await help_command(update, context)
    else:
        await update.message.reply_text(
            "🤔 Не понял. Используйте /help для списка команд.\n"
            "Или просто напишите «найди работу»."
        )


# ========== WEBHOOK И ЗАПУСК ==========
app = FastAPI()
telegram_app: Optional[Application] = None


@app.post("/webhook")
async def webhook(request: Request):
    global telegram_app
    if telegram_app is None:
        return {"error": "Application not initialized"}
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"status": "alive", "bot": "hh-bot"}


async def run():
    global telegram_app
    telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Регистрация обработчиков
    telegram_app.add_handler(CommandHandler("start", start))
    telegram_app.add_handler(CommandHandler("search", search_command))
    telegram_app.add_handler(CommandHandler("stats", stats_command))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(CallbackQueryHandler(callback_handler))
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Инициализация и запуск
    await telegram_app.initialize()
    await telegram_app.start()

    # Установка webhook, если задан внешний URL
    if RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        await telegram_app.bot.set_webhook(url=webhook_url)
        logger.info(f"Webhook установлен: {webhook_url}")

    # Запускаем FastAPI сервер
    config = uvicorn.Config(app, host="0.0.0.0", port=10000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(run())