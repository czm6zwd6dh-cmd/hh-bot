# -*- coding: utf-8 -*-
import os
import sqlite3
import asyncio
import logging
import re
import json
import yaml
import aiohttp
import random
from datetime import datetime
from typing import Optional, List
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
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")

MAX_VACANCIES_PER_CYCLE = 5
DB_PATH = "vacancies.db"
PROFILE_PATH = "profile.yaml"

# Глобальный флаг для блокировки одновременных поисков
search_in_progress = False

# --- DeepSeek (опционально) ---
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
    c.execute("CREATE TABLE IF NOT EXISTS seen_vacancies (id TEXT PRIMARY KEY, title TEXT, company TEXT, score REAL, seen_at TIMESTAMP)")
    c.execute("CREATE TABLE IF NOT EXISTS applications (id INTEGER PRIMARY KEY AUTOINCREMENT, vacancy_id TEXT, title TEXT, company TEXT, score REAL, status TEXT, created_at TIMESTAMP)")
    conn.commit(); conn.close()
init_db()

def is_seen(vid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT 1 FROM seen_vacancies WHERE id=?", (vid,))
    res = c.fetchone() is not None
    conn.close()
    return res

def mark_seen(vid, title, company, score):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO seen_vacancies VALUES (?,?,?,?,?)", (vid, title, company, score, datetime.now()))
    conn.commit(); conn.close()

def add_application(vid, title, company, score):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO applications (vacancy_id, title, company, score, status, created_at) VALUES (?,?,?,?,?,?)",
              (vid, title, company, score, "new", datetime.now()))
    conn.commit(); conn.close()

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM seen_vacancies")
    seen = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM applications")
    apps = c.fetchone()[0]
    conn.close()
    return seen, apps

# --- Скоринг ---
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

def calc_keyword_score(text, keywords):
    text = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text)
    return min(matches / max(len(keywords)*0.3, 1), 1.0) * 10

def calc_salary_score(salary):
    if not salary:
        return 5.0
    from_ = salary.get("from")
    to_ = salary.get("to")
    min_sal = PROFILE["candidate"]["salary_min"]
    if from_ and from_ >= min_sal: return 10.0
    if to_ and to_ >= min_sal: return 8.0
    if from_ and from_ >= min_sal*0.8: return 6.0
    if to_ and to_ >= min_sal*0.8: return 4.0
    return 2.0

def calc_location_score(city):
    city = city.lower()
    for allowed in PROFILE["filters"]["cities"]:
        if allowed.lower() in city: return 10.0
    return 0.0

def calc_experience_score(text):
    text = text.lower()
    score = sum(2 for kw in ["опыт","лет","руководитель","директор"] if kw in text)
    if re.search(r"\b[5-9]\b|\b10\b|\b20\b", text): score += 3
    return min(score, 10.0)

def calc_skills_score(text):
    return calc_keyword_score(text, PROFILE["candidate"]["key_skills"])

def score_vacancy(vacancy):
    text = (vacancy.get("name","") + " " + vacancy.get("description","") + " " + vacancy.get("snippet",{}).get("requirement","")).lower()
    w = PROFILE["scoring"]["weights"]
    role = calc_keyword_score(text, PROFILE["filters"]["keywords"])
    industry = calc_keyword_score(text, PROFILE["filters"]["industry_keywords"])
    salary = calc_salary_score(vacancy.get("salary"))
    location = calc_location_score(vacancy.get("area",{}).get("name",""))
    experience = calc_experience_score(text)
    skills = calc_skills_score(text)
    total = role*w["role_fit"] + industry*w["industry_match"] + salary*w["salary_match"] + location*w["location_match"] + experience*w["experience_match"] + skills*w["skills_match"]
    for word in PROFILE["filters"]["exclude_words"]:
        if word.lower() in text:
            total = 0
            break
    if total >= 80: verdict="STRONG_MATCH"; reason="Отлично"
    elif total >= PROFILE["scoring"]["min_score"]: verdict="MATCH"; reason="Хорошо"
    elif total >= 40: verdict="WEAK_MATCH"; reason="Слабо"
    else: verdict="SKIP"; reason="Низкий скор"
    return VacancyScore(total=round(total,1), role_fit=round(role,1), industry_match=round(industry,1),
                        salary_match=round(salary,1), location_match=round(location,1),
                        experience_match=round(experience,1), skills_match=round(skills,1),
                        verdict=verdict, reasoning=reason)

# --- Функция для RSS (запасной вариант) ---
async def fetch_hh_rss(city_id, keyword, per_page=10):
    """Получает вакансии через RSS-ленту."""
    # Убираем per_page, так как RSS всегда возвращает 10 последних
    url = f"https://hh.ru/rss/vacancy?text={keyword}&area={city_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status != 200:
                    logger.warning(f"RSS status {resp.status} for {keyword} in {city_id}")
                    return []
                xml_text = await resp.text()
                root = ET.fromstring(xml_text)
                items = []
                for item in root.findall(".//item"):
                    title = item.find("title").text if item.find("title") is not None else ""
                    link = item.find("link").text if item.find("link") is not None else ""
                    description = item.find("description").text if item.find("description") is not None else ""
                    # Извлекаем компанию из description
                    company = "Неизвестно"
                    if "Компания:" in description:
                        company = description.split("Компания:")[1].split("<")[0].strip()
                    # Извлекаем город
                    city = ""
                    if "Город:" in description:
                        city = description.split("Город:")[1].split("<")[0].strip()
                    # ID из ссылки
                    vid = link.split("/")[-1] if link else "0"
                    items.append({
                        "id": vid,
                        "url": link,
                        "name": title,
                        "employer": {"name": company},
                        "area": {"name": city},
                        "salary": None,
                        "snippet": {"requirement": description[:200], "responsibility": ""},
                        "description": description,
                    })
                return items[:per_page]
        except Exception as e:
            logger.error(f"RSS error: {e}")
            return []

# --- Запрос к HH.ru API с полными заголовками и случайным User-Agent ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

async def fetch_hh_api(city_id, keyword, per_page=15):
    url = "https://api.hh.ru/vacancies"
    params = {
        "area": city_id,
        "text": keyword,
        "per_page": per_page,
        "order_by": "publication_time",
    }
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://hh.ru/",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params, headers=headers, timeout=15) as resp:
                if resp.status == 403:
                    logger.warning("HH API вернул 403, пробуем RSS")
                    return None  # сигнал, что API заблокирован
                if resp.status != 200:
                    logger.warning(f"HH API status {resp.status} for {keyword}")
                    return []
                data = await resp.json()
                items = data.get("items", [])
                logger.info(f"API: найдено {len(items)} вакансий по '{keyword}' в городе {city_id}")
                result = []
                for item in items:
                    result.append({
                        "id": item.get("id"),
                        "url": item.get("alternate_url"),
                        "name": item.get("name"),
                        "employer": {"name": item.get("employer", {}).get("name")},
                        "area": {"name": item.get("area", {}).get("name")},
                        "salary": item.get("salary"),
                        "snippet": {
                            "requirement": item.get("snippet", {}).get("requirement", ""),
                            "responsibility": item.get("snippet", {}).get("responsibility", ""),
                        },
                        "description": item.get("description", ""),
                    })
                return result
        except Exception as e:
            logger.error(f"fetch_hh_api error: {e}")
            return []

# --- Основная функция поиска ---
async def fetch_hh_vacancies(city_id, keyword, per_page=15):
    api_result = await fetch_hh_api(city_id, keyword, per_page)
    if api_result is None:
        logger.info(f"Используем RSS для {keyword} в {city_id}")
        return await fetch_hh_rss(city_id, keyword, per_page)
    return api_result if api_result is not None else []

# --- Форматирование с ссылкой ---
def format_vacancy(v, score, idx, total):
    sal = v.get("salary")
    sal_str = f"{sal.get('from','')} - {sal.get('to','')} {sal.get('currency','')}".strip() if sal else "не указана"
    url = v.get("url", "")
    msg = (f"📌 {idx}/{total}\n\n<b>{v.get('name','')}</b>\n🏢 {v.get('employer',{}).get('name','')}\n📍 {v.get('area',{}).get('name','')}\n💰 {sal_str}\n\n"
           f"📊 Скор: {score.total} ({score.verdict})\n• Роль: {score.role_fit}/10\n• Индустрия: {score.industry_match}/10\n"
           f"• Зарплата: {score.salary_match}/10\n• Локация: {score.location_match}/10\n• Опыт: {score.experience_match}/10\n"
           f"• Навыки: {score.skills_match}/10\n\n💬 {score.reasoning}")
    vid = v.get("id")
    # Клавиатура с кнопкой "Перейти"
    kb_buttons = [
        [InlineKeyboardButton("👍", callback_data=f"like:{vid}"),
         InlineKeyboardButton("👎", callback_data=f"dislike:{vid}"),
         InlineKeyboardButton("📝", callback_data=f"apply:{vid}")],
        []
    ]
    if url:
        kb_buttons[1].append(InlineKeyboardButton("🔗 Перейти", url=url))
    kb_buttons[1].append(InlineKeyboardButton("➡️ Далее", callback_data="next"))
    kb = InlineKeyboardMarkup(kb_buttons)
    return msg, kb

# --- Команды ---
async def start(update, context):
    await update.message.reply_text("👋 Привет! Я умею искать вакансии коммерческого директора.\n"
                                    "Напиши что-нибудь, я постараюсь помочь.\n"
                                    "Команды: /search, /stats, /help")

async def search_command(update, context):
    global search_in_progress
    if search_in_progress:
        await update.message.reply_text("⏳ Поиск уже выполняется, подождите немного.")
        return
    await update.message.reply_text("🔍 Ищу вакансии, подождите...")
    await do_search(update, context)

async def do_search(update, context, force=False):
    global search_in_progress
    if search_in_progress:
        return
    search_in_progress = True
    try:
        chat_id = update.effective_chat.id
        # Берём первые 2 города и 2 ключевых слова для скорости
        cities = list(PROFILE["filters"]["cities"].items())[:2]
        keywords = PROFILE["filters"]["keywords"][:2]

        all_vacancies = []
        api_blocked = False
        for city_name, city_id in cities:
            for kw in keywords:
                vacs = await fetch_hh_vacancies(city_id, kw, per_page=10)
                if vacs is None:
                    api_blocked = True
                    continue
                all_vacancies.extend(vacs)
                # Случайная задержка 3-5 секунд между запросами
                await asyncio.sleep(random.uniform(3, 5))

        if api_blocked and not all_vacancies:
            await context.bot.send_message(chat_id, "⚠️ API HH.ru временно недоступен (403). Попробуйте позже или измените ключевые слова.")
            return

        # Удаляем дубликаты и уже просмотренные
        seen_ids = set()
        unique = []
        for v in all_vacancies:
            if v["id"] in seen_ids:
                continue
            seen_ids.add(v["id"])
            if not force and is_seen(v["id"]):
                continue
            unique.append(v)

        if not unique:
            await context.bot.send_message(chat_id, "📭 Новых вакансий не найдено.\n"
                                            "Попробуйте расширить критерии поиска (измените profile.yaml) "
                                            "или подождите появления новых вакансий.")
            return

        # Скоринг
        scored = [(v, score_vacancy(v)) for v in unique]
        scored.sort(key=lambda x: x[1].total, reverse=True)
        good = [(v,s) for v,s in scored if s.verdict in ("MATCH","STRONG_MATCH")]
        if not good:
            good = scored[:MAX_VACANCIES_PER_CYCLE]
        else:
            good = good[:MAX_VACANCIES_PER_CYCLE]

        context.chat_data["vacancies"] = good
        context.chat_data["index"] = 0
        await show_vacancy(update, context, 0)
    finally:
        search_in_progress = False

async def show_vacancy(update, context, idx):
    vacancies = context.chat_data.get("vacancies", [])
    if not vacancies or idx >= len(vacancies):
        await update.effective_message.reply_text("✅ Все вакансии просмотрены.")
        return
    v, s = vacancies[idx]
    msg, kb = format_vacancy(v, s, idx+1, len(vacancies))
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=kb, parse_mode="HTML")
    else:
        await update.message.reply_text(msg, reply_markup=kb, parse_mode="HTML")
    context.chat_data["index"] = idx

async def callback_handler(update, context):
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
        for v,s in vacancies:
            if v["id"] == vid:
                target = (v,s); break
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
            add_application(vid, v.get("name"), v.get("employer",{}).get("name"), s.total)
        mark_seen(vid, v.get("name"), v.get("employer",{}).get("name"), s.total)

async def stats_command(update, context):
    seen, apps = get_stats()
    await update.message.reply_text(f"📊 Статистика:\n• Просмотрено: {seen}\n• Откликов: {apps}")

async def help_command(update, context):
    await update.message.reply_text("📖 Команды:\n/search – поиск\n/stats – статистика\n/help – помощь\n\nПросто пишите, я отвечу.")

# --- DeepSeek диалог ---
async def deepseek_chat(message: str) -> str:
    if not deepseek_available:
        return "Извините, я сейчас не могу ответить (нет подключения к ИИ). Попробуйте команду /help."
    system_prompt = (
        "Ты — помощник по поиску работы. Ты помогаешь пользователю найти вакансии коммерческого директора "
        "в нефтяной отрасли. Отвечай кратко, по делу, на русском языке. Если спрашивают о вакансиях – уточни, "
        "что можно использовать /search для поиска. Если вопрос не по теме – вежливо скажи, что ты специализированный бот."
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
        return "⚠️ Ошибка при обращении к ИИ. Попробуйте позже."

# --- Эвристический парсер ---
def parse_natural(text: str) -> Optional[str]:
    text = text.lower().strip()
    if any(w in text for w in ["привет", "здравствуй", "ку"]):
        return "help"
    if any(w in text for w in ["найди", "поиск", "вакансии", "работу", "ищи"]):
        return "search"
    if any(w in text for w in ["статистика", "стат", "сколько"]):
        return "stats"
    return None

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    action = parse_natural(text)
    if action == "search":
        await search_command(update, context)
        return
    elif action == "stats":
        await stats_command(update, context)
        return
    elif action == "help":
        await help_command(update, context)
        return

    await update.message.reply_text("🤔 Думаю...")
    reply = await deepseek_chat(text)
    await update.message.reply_text(reply)

# --- Webhook ---
app = FastAPI()
telegram_app = None

@app.post("/webhook")
async def webhook(request: Request):
    global telegram_app
    if telegram_app is None:
        return {"error": "not ready"}
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"status": "alive", "deepseek": deepseek_available}

async def run():
    global telegram_app
    telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()
    telegram_app.add_handler(CommandHandler("start", start))
    telegram_app.add_handler(CommandHandler("search", search_command))
    telegram_app.add_handler(CommandHandler("stats", stats_command))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(CallbackQueryHandler(callback_handler))
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    await telegram_app.initialize()
    await telegram_app.start()

    if RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        await telegram_app.bot.set_webhook(url=webhook_url)
        logger.info(f"Webhook set to {webhook_url}")

    config = uvicorn.Config(app, host="0.0.0.0", port=10000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(run())