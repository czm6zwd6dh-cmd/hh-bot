# -*- coding: utf-8 -*-
"""
proxy_pool.py — бесплатный пул прокси с автоматической загрузкой,
проверкой и ротацией. Не требует платных сервисов.
"""

import asyncio
import logging
import random
import time

import aiohttp

logger = logging.getLogger(__name__)

PROXY_SOURCES = [
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/http/data.txt",
]

DOWNLOAD_LIMIT_PER_SOURCE = 400
VALIDATE_WORKERS = 80
VALIDATE_TIMEOUT = 8
POOL_SIZE = 25
REFRESH_SEC = 1800
MAX_FAILS = 2


class ProxyPool:
    def __init__(self, test_url, test_params=None, test_headers=None,
                 pool_size=POOL_SIZE, refresh_sec=REFRESH_SEC):
        self.test_url = test_url
        self.test_params = test_params or {}
        self.test_headers = test_headers or {}
        self.pool_size = pool_size
        self.refresh_sec = refresh_sec
        self._alive = []
        self._fails = {}
        self._lock = asyncio.Lock()
        self._updated_at = 0.0
        self._stop = False
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=20)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def get(self):
        alive_copy = list(self._alive)
        candidates = [p for p in alive_copy if self._fails.get(p, 0) < MAX_FAILS]
        return random.choice(candidates) if candidates else None

    def report_bad(self, proxy):
        if not proxy:
            return
        self._fails[proxy] = self._fails.get(proxy, 0) + 1
        if self._fails[proxy] >= MAX_FAILS and proxy in self._alive:
            try:
                self._alive.remove(proxy)
                logger.info(f"Прокси {proxy} удалён из пула (осталось {len(self._alive)})")
            except ValueError:
                pass

    def report_good(self, proxy):
        if proxy:
            self._fails[proxy] = 0

    @property
    def size(self):
        return len(self._alive)

    async def run(self):
        try:
            while not self._stop:
                try:
                    await self.refresh()
                except Exception as e:
                    logger.error(f"ProxyPool refresh error: {e}")
                await asyncio.sleep(self.refresh_sec)
        except asyncio.CancelledError:
            logger.info("ProxyPool: фоновая задача отменена")
            raise

    def stop(self):
        self._stop = True

    async def close(self):
        self.stop()
        if self._session and not self._session.closed:
            await self._session.close()

    async def refresh(self):
        raw = await self._download()
        async with self._lock:
            old_alive = list(self._alive)
        candidates = list(dict.fromkeys(old_alive + raw))[:1500]
        logger.info(f"ProxyPool: проверяю {len(candidates)} прокси...")
        good = await self._validate_all(candidates)
        async with self._lock:
            self._alive = good[: self.pool_size]
            self._fails = {p: 0 for p in self._alive}
            self._updated_at = time.time()
        logger.info(f"ProxyPool: рабочих прокси в пуле: {len(self._alive)}")

    async def _download(self):
        out = []
        session = await self._get_session()
        for url in PROXY_SOURCES:
            try:
                async with session.get(url) as r:
                    if r.status != 200:
                        continue
                    text = await r.text()
                    lines = [l.strip() for l in text.splitlines()
                             if l.strip() and ":" in l and l[0].isdigit()]
                    out.extend(lines[:DOWNLOAD_LIMIT_PER_SOURCE])
                    logger.info(f"ProxyPool: {url.split('/')[2]} -> {len(lines)} шт.")
            except Exception as e:
                logger.warning(f"ProxyPool download {url}: {e}")
        return list(dict.fromkeys(out))

    async def _validate_all(self, candidates):
        sem = asyncio.Semaphore(VALIDATE_WORKERS)
        good = []

        async def check(proxy):
            async with sem:
                if await self._is_alive(proxy):
                    good.append(proxy)

        await asyncio.gather(*(check(p) for p in candidates), return_exceptions=True)
        return good

    async def _is_alive(self, proxy):
        try:
            timeout = aiohttp.ClientTimeout(total=VALIDATE_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as check_session:
                async with check_session.get(
                    self.test_url,
                    params=self.test_params,
                    headers=self.test_headers,
                    proxy=f"http://{proxy}",
                    ssl=False
                ) as r:
                    if r.status != 200:
                        return False
                    try:
                        data = await r.json(content_type=None)
                    except Exception:
                        return False
                    return isinstance(data, dict)
        except Exception:
            return False
