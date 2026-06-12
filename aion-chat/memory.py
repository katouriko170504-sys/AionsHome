"""
向量记忆库：embedding、recall、手动总结、即时哨兵（RAG 路由）
"""

import json, time, struct, math, asyncio
from datetime import datetime

import aiosqlite, httpx

from config import get_key, get_sentinel_config, get_embedding_config, load_worldbook, save_chat_status, load_digest_anchor, save_digest_anchor, DEFAULT_MODEL
from database import get_db
from ws import manager

# ── 向量工具 ──────────────────────────────────────
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMS = 3072


def _connor_display_name() -> str:
    try:
        from chatroom import load_chatroom_config
        return load_chatroom_config().get("connor_name") or "第二AI"
    except Exception:
        return "第二AI"


def _json_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return [text]


def _source_ids_for_memory(mem: dict) -> list[str]:
    ids = []
    source_conv = mem.get("source_conv") or ""
    for raw in _json_list(mem.get("source_msg_id")):
        source_id = str(raw).strip()
        if not source_id:
            continue
        if ":" not in source_id:
            prefix = "chatroom" if str(source_conv).startswith("chatroom:") else "private"
            source_id = f"{prefix}:{source_id}"
        ids.append(source_id)
    return ids


SUMMARY_MEMORY_TYPES = {"digest", "seeky_digest", "seeky_compressed", "daily"}
LONG_TERM_MEMORY_TYPE = "important"


def memory_kind_for_type(memory_type: str) -> str:
    """Two-bucket memory class: summary-style records are daily; everything else is long-term."""
    return "daily" if str(memory_type or "").strip().lower() in SUMMARY_MEMORY_TYPES else "long_term"


def memory_kind_label(memory_type: str) -> str:
    return "日常" if memory_kind_for_type(memory_type) == "daily" else "长期重要"


def _pack_embedding(values: list[float]) -> bytes:
    return struct.pack(f'{len(values)}f', *values)


def _unpack_embedding(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f'{n}f', blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def get_embedding(text: str) -> list[float] | None:
    ecfg = get_embedding_config()
    if not ecfg["api_key"]:
        return None
    if ecfg["use_openai"]:
        # OpenAI 兼容格式（硅基流动等）
        url = f"{ecfg['base_url']}/v1/embeddings"
        headers = {"Authorization": f"Bearer {ecfg['api_key']}", "Content-Type": "application/json"}
        body = {"model": ecfg["model"], "input": text}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, json=body, headers=headers)
                if resp.status_code != 200:
                    print(f"[Embedding] OpenAI 兼容调用失败 {resp.status_code}: {resp.text[:300]}")
                    return None
                return resp.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"[Embedding] 调用异常: {e}")
            return None
    else:
        # Gemini 原生格式
        model = ecfg["model"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={ecfg['api_key']}"
        body = {"content": {"parts": [{"text": text}]}}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, json=body)
                resp.raise_for_status()
                return resp.json()["embedding"]["values"]
        except Exception as e:
            print(f"[Embedding] Gemini 调用异常: {e}")
            return None


# ── 关键词匹配辅助 ──────────────────────
def _keyword_match_score(query_keywords: list[str], mem_keywords_json: str) -> float:
    """计算关键词命中率：命中关键词数 / 查询关键词数"""
    if not query_keywords:
        return 0.0
    try:
        mem_kws = json.loads(mem_keywords_json) if mem_keywords_json else []
    except (json.JSONDecodeError, TypeError):
        mem_kws = []
    if not mem_kws:
        return 0.0
    mem_kws_lower = [k.lower() for k in mem_kws]
    hits = sum(1 for qk in query_keywords if any(qk.lower() in mk or mk in qk.lower() for mk in mem_kws_lower))
    return hits / len(query_keywords)


# ── 记忆召回（向量 + 关键词 + 重要度 综合评分）────
async def recall_memories(query_text: str, query_keywords: list[str] = None,
                          top_k: int = 5, threshold: float = 0.45) -> tuple[list[dict], list[dict]]:
    """
    综合评分 = 向量相似度×0.6 + 关键词命中率×0.3 + 重要度×0.1
    threshold 为最终得分门槛。
    返回 (matched, debug_top6): matched 为达标结果, debug_top6 为得分最高的前6条（含未达标）
    """
    query_vec = await get_embedding(query_text)
    if not query_vec:
        return [], []
    if query_keywords is None:
        query_keywords = []
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT id, content, type, created_at, source_conv, embedding, keywords, importance, "
            "source_start_ts, source_end_ts, source_msg_id "
            "FROM memories WHERE embedding IS NOT NULL"
        )
        rows = await cur.fetchall()
    all_scored = []
    for row in rows:
        mem_vec = _unpack_embedding(row["embedding"])
        vec_sim = cosine_similarity(query_vec, mem_vec)
        kw_score = _keyword_match_score(query_keywords, row["keywords"]) if query_keywords else 0.0
        importance = float(row["importance"] or 0.5)
        final_score = vec_sim * 0.6 + kw_score * 0.3 + importance * 0.1
        item = {
            "id": row["id"], "content": row["content"], "type": row["type"],
            "created_at": row["created_at"],
            "score": round(final_score, 4),
            "vec_sim": round(vec_sim, 4),
            "kw_score": round(kw_score, 4),
            "importance": round(importance, 2),
            "keywords": row["keywords"] or "",
            "source_start_ts": row["source_start_ts"],
            "source_end_ts": row["source_end_ts"],
            "source_conv": row["source_conv"],
            "source_msg_id": row["source_msg_id"],
        }
        all_scored.append(item)
    all_scored.sort(key=lambda x: x["score"], reverse=True)
    debug_top6 = all_scored[:6]
    matched = [r for r in all_scored if r["score"] >= threshold][:top_k]
    return matched, debug_top6


# ── 追溯原文：通过记忆的时间范围 + 关键词筛选原始聊天 ─
async def fetch_source_details(memories: list[dict], keywords: list[str]) -> str:
    """
    在每条记忆的 source 时间范围内，取出所有包含关键词的消息，
    去重、按时间排序后返回。
    """
    if not memories or not keywords:
        return ""

    wb = load_worldbook()
    user_name = wb.get("user_name", "用户")
    ai_name = wb.get("ai_name", "AI")
    kw_lower = [k.lower() for k in keywords if k.strip()]
    if not kw_lower:
        return ""

    seen = set()
    matched_rows = []

    for mem in memories:
        source_ids = _source_ids_for_memory(mem)
        if source_ids:
            async with get_db() as db:
                db.row_factory = aiosqlite.Row
                for source_id in source_ids:
                    if ":" not in source_id:
                        continue
                    prefix, raw_id = source_id.split(":", 1)
                    if prefix == "private":
                        cur = await db.execute(
                            "SELECT role, content, created_at FROM messages WHERE id=?",
                            (raw_id,),
                        )
                        row = await cur.fetchone()
                        if row:
                            key = (row["created_at"], row["content"][:80])
                            if key not in seen:
                                seen.add(key)
                                matched_rows.append(row)
                    elif prefix == "chatroom":
                        cur = await db.execute(
                            "SELECT sender, content, created_at FROM chatroom_messages WHERE id=? AND sender != 'system'",
                            (raw_id,),
                        )
                        row = await cur.fetchone()
                        if row:
                            key = (row["created_at"], row["content"][:80])
                            if key not in seen:
                                seen.add(key)
                                matched_rows.append({
                                    "role": "assistant" if row["sender"] == "aion" else "user",
                                    "content": row["content"],
                                    "created_at": row["created_at"],
                                    "_sender": row["sender"],
                                })
            print(f"[source_detail] 记忆 {mem.get('id','?')[:12]} 使用精确原文 {len(source_ids)} 条")
            continue

        start_ts = mem.get("source_start_ts")
        end_ts = mem.get("source_end_ts")
        if not start_ts or not end_ts:
            print(f"[source_detail] 跳过无时间范围的记忆: {mem.get('id','?')}")
            continue
        async with get_db() as db:
            db.row_factory = aiosqlite.Row
            # 私聊消息
            cur = await db.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE role IN ('user','assistant') AND created_at >= ? AND created_at <= ? "
                "ORDER BY created_at ASC",
                (start_ts, end_ts)
            )
            rows = list(await cur.fetchall())
            # 群聊消息
            cur = await db.execute(
                "SELECT id FROM chatroom_rooms WHERE type = 'group' ORDER BY updated_at DESC LIMIT 1"
            )
            group_room = await cur.fetchone()
            if group_room:
                cur = await db.execute(
                    "SELECT sender, content, created_at FROM chatroom_messages "
                    "WHERE room_id = ? AND created_at >= ? AND created_at <= ? AND sender != 'system' "
                    "ORDER BY created_at ASC",
                    (group_room["id"], start_ts, end_ts),
                )
                for gr in await cur.fetchall():
                    rows.append({"role": "assistant" if gr["sender"] == "aion" else "user",
                                 "content": gr["content"], "created_at": gr["created_at"],
                                 "_sender": gr["sender"]})
        print(f"[source_detail] 记忆 {mem.get('id','?')[:12]} 范围 {start_ts}-{end_ts}: 取到 {len(rows)} 条消息")
        hit_count = 0
        for row in rows:
            content_lower = row["content"].lower()
            if any(kw in content_lower for kw in kw_lower):
                key = (row["created_at"], row["content"][:80])
                if key not in seen:
                    seen.add(key)
                    matched_rows.append(row)
                    hit_count += 1
        print(f"[source_detail] → 关键词 {kw_lower} 命中 {hit_count} 条")

    matched_rows.sort(key=lambda r: r["created_at"])
    connor_name = _connor_display_name()
    detail_lines = []
    for row in matched_rows:
        sender = row["_sender"] if "_sender" in row.keys() else ""
        if sender:
            name = {"user": user_name, "aion": ai_name, "connor": connor_name}.get(sender, sender)
        else:
            name = user_name if row["role"] == "user" else ai_name
        detail_lines.append(f"{name}: {row['content'][:500]}")

    print(f"[source_detail] 最终返回 {len(detail_lines)} 条原文")
    return "\n".join(detail_lines) if detail_lines else ""


# ── 背景记忆浮现：unresolved + 话题相关 + 近期补充 ───
async def build_surfacing_memories(topic: str = "", keywords: list[str] = None,
                                    max_total: int = 8) -> tuple[list[dict], set]:
    """
    构建 [背景记忆] 注入内容。
    策略：
      1. unresolved 优先（最多 2 条）
      2. 话题相关浮现（topic embedding 匹配，最多 3 条）
      3. 近期补充（最近 3 天，补满 max_total）
    返回 (memories_list, surfaced_ids) 供后续 RAG 去重。
    """
    surfaced_ids = set()
    result = []

    # 1. unresolved 优先
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT id, content, type, created_at, keywords, importance, unresolved "
            "FROM memories WHERE unresolved = 1 ORDER BY created_at DESC LIMIT 2"
        )
        unresolved_rows = await cur.fetchall()
    for row in unresolved_rows:
        item = {"id": row["id"], "content": row["content"], "unresolved": True}
        result.append(item)
        surfaced_ids.add(row["id"])

    # 2. 话题相关浮现
    if topic and topic.strip() and len(result) < max_total:
        topic_vec = await get_embedding(topic)
        if topic_vec:
            async with get_db() as db:
                db.row_factory = aiosqlite.Row
                cur = await db.execute(
                    "SELECT id, content, type, created_at, embedding, keywords, importance "
                    "FROM memories WHERE embedding IS NOT NULL"
                )
                rows = await cur.fetchall()
            scored = []
            for row in rows:
                if row["id"] in surfaced_ids:
                    continue
                mem_vec = _unpack_embedding(row["embedding"])
                sim = cosine_similarity(topic_vec, mem_vec)
                if sim >= 0.50:
                    scored.append({"id": row["id"], "content": row["content"], "sim": sim, "unresolved": False})
            scored.sort(key=lambda x: x["sim"], reverse=True)
            for item in scored[:3]:
                if len(result) >= max_total:
                    break
                result.append(item)
                surfaced_ids.add(item["id"])

    # 3. 近期补充（最近 3 天）
    if len(result) < max_total:
        three_days_ago = time.time() - 3 * 86400
        async with get_db() as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT id, content, type, created_at FROM memories "
                "WHERE created_at > ? ORDER BY created_at DESC LIMIT ?",
                (three_days_ago, max_total)
            )
            recent_rows = await cur.fetchall()
        for row in recent_rows:
            if len(result) >= max_total:
                break
            if row["id"] in surfaced_ids:
                continue
            result.append({"id": row["id"], "content": row["content"], "unresolved": False})
            surfaced_ids.add(row["id"])

    return result, surfaced_ids


# ── 哨兵/前置模型统一调用 ────────────────────────
async def _call_sentinel_text(scfg: dict, prompt: str, timeout: int = 60) -> str | None:
    """统一调用哨兵模型（纯文本），支持 Gemini 原生和 OpenAI 兼容格式"""
    if scfg["use_openai"]:
        url = f"{scfg['base_url']}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {scfg['api_key']}", "Content-Type": "application/json"}
        payload = {
            "model": scfg["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 4096,
            "enable_thinking": False,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                print(f"[Sentinel] OpenAI 兼容调用失败 {resp.status_code}: {resp.text[:500]}")
                raise Exception(f"Sentinel API {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    else:
        model = scfg["model"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={scfg['api_key']}"
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json={"contents": contents, "safetySettings": safety_settings})
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()


async def _call_sentinel_vision(scfg: dict, prompt: str, img_b64: str, mime_type: str = "image/jpeg", timeout: int = 60) -> str | None:
    """统一调用哨兵模型（带图片），支持 Gemini 原生和 OpenAI 兼容格式"""
    if scfg["use_openai"]:
        url = f"{scfg['base_url']}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {scfg['api_key']}", "Content-Type": "application/json"}
        payload = {
            "model": scfg["model"],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}}
            ]}],
            "temperature": 0.3,
            "max_tokens": 4096,
            "enable_thinking": False,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                print(f"[Sentinel] OpenAI 兼容 Vision 调用失败 {resp.status_code}: {resp.text[:500]}")
                raise Exception(f"Sentinel Vision API {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    else:
        model = scfg["model"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={scfg['api_key']}"
        contents = [{"role": "user", "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": img_b64}}
        ]}]
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json={"contents": contents, "safetySettings": safety_settings})
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ── 即时哨兵：每次用户发消息后触发（RAG 路由） ────
async def instant_digest(recent_messages: list[dict]) -> dict:
    """
    用户每次发消息后即时调用 flash-lite，返回结构化 JSON：
    {is_search_needed, keywords, require_detail, status}
    """
    gemini_key = get_key("gemini_free")
    scfg = get_sentinel_config()
    if not scfg["api_key"] or not recent_messages:
        return {"is_search_needed": False, "keywords": [], "require_detail": False, "status": "", "topic": ""}

    wb = load_worldbook()
    user_name = wb.get("user_name", "用户")
    ai_name = wb.get("ai_name", "AI")

    messages_text = "\n".join([
        f"{user_name if m['role']=='user' else ai_name}: {m['content'][:200]}"
        for m in recent_messages
    ])

    prompt = (
        f"你是一个 RAG 系统的查询优化路由。分析用户输入，输出 JSON：\n"
        f"1. 忽略高频对话称呼：不要提取对话者的名字或昵称（如 \"{ai_name}\", \"{user_name}\", \"小鬣狗\", \"老公\", \"宝贝\"）作为关键词。\n"
        f"2. 忽略高频常用词：如\"晚安故事\",\"吃什么\"等。\n"
        f"3. 聚焦核心实体：只提取稀缺的、具有区分度的名词（地点、物品、特定事件、专有名词等）\n"
        f"4. 仅当提起之前做过的事、过去的回忆时，is_search_needed才输出为true。若在询问日常问题，不涉及回忆过去，is_search_needed输出为false。\n"
        f"   \"is_search_needed\": Boolean.\n"
        f"      - false: 纯闲聊/语气词/无实质内容，只是在陈述或表达感情，并未进行对于具体事实的询问则输出false。\n"
        f"      - true: 当包含询问、回忆、或需要背景信息的对话，提起“昨天”、“之前”、“你还记得……”等。\n"
        f"   \"keywords\": 提取 2-4 个搜索关键词（过滤掉 {ai_name}, {user_name} 等高频人名）。\n"
        f"   \"require_detail\": Boolean.\n"
        f"      - false: 模糊回忆/情感抒发（只需读取摘要）。\n"
        f"      - true: 当且仅当询问具体事实/细节/步骤（需要读取正文），例如：还记得我们之前…你记得上次…等。\n"
        f"5. \"status\": 结合上下文总结{user_name}当前所处的状态（如：{user_name}刚吃完晚饭准备出门、洗完澡准备睡觉、回到家开始工作了等）。\n"
        f"6. \"topic\": 用一两句话概括当前对话可能会涉及到的回忆（如：在聊中午吃什么，在聊之前看过的电影）。若无明确话题则留空。\n\n"
        f"严格只输出一个 JSON 对象，不要输出任何其他内容。\n\n"
        f"对话：\n{messages_text}"
    )

    try:
        raw = await _call_sentinel_text(scfg, prompt, timeout=15)
        if not raw:
            return {"is_search_needed": False, "keywords": [], "require_detail": False, "status": "", "topic": ""}

        # 提取 JSON（可能包裹在 ```json ... ``` 中）
        if "```" in raw:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

        result = json.loads(raw)
        is_search = bool(result.get("is_search_needed", False))
        keywords = result.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.replace("、", ",").split(",") if k.strip()]
        require_detail = bool(result.get("require_detail", False))
        status = str(result.get("status", "")).strip()

        if status:
            save_chat_status(status)
            await manager.broadcast({"type": "chat_status", "data": {"status": status, "updated_at": time.time()}})

        topic = str(result.get("topic", "")).strip()

        return {
            "is_search_needed": is_search,
            "keywords": keywords,
            "require_detail": require_detail,
            "status": status,
            "topic": topic,
        }
    except Exception:
        return {"is_search_needed": False, "keywords": [], "require_detail": False, "status": "", "topic": ""}


# ── 手动总结：分组提取记忆 ─────────────────────────

def _split_into_groups(msgs: list, group_size: int = 30) -> list[list]:
    """将消息列表按每 group_size 条分组，余数<10并入最后一组，>=10单独一组"""
    total = len(msgs)
    if total <= group_size:
        return [msgs]

    full_groups = total // group_size
    remainder = total % group_size

    if remainder > 0 and remainder < 10:
        # 余数<10，并入最后一个完整组
        full_groups -= 1
        # 前面的完整组
        groups = [msgs[i * group_size:(i + 1) * group_size] for i in range(full_groups)]
        # 最后一组 = 最后一个完整组 + 余数
        groups.append(msgs[full_groups * group_size:])
    else:
        # 余数>=10 或余数=0
        groups = [msgs[i * group_size:(i + 1) * group_size] for i in range(full_groups)]
        if remainder > 0:
            groups.append(msgs[full_groups * group_size:])

    return groups


async def _call_flash_lite(prompt: str) -> dict | None:
    """调用哨兵模型，返回 JSON 结果（仅供即时哨兵使用）"""
    scfg = get_sentinel_config()
    if not scfg["api_key"]:
        return None
    try:
        raw = await _call_sentinel_text(scfg, prompt, timeout=60)
        if not raw:
            return None
        # 提取 JSON
        if "```" in raw:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]
        return json.loads(raw)
    except Exception:
        return None


def _parse_json_response(raw: str) -> dict | None:
    """从模型输出中提取 JSON 对象"""
    import re as _re
    raw = raw.strip()
    raw = _re.sub(r'<think>[\s\S]*?</think>', '', raw).strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        raw = raw[start:end]
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def _parse_json_array_response(raw: str) -> list | None:
    """从模型输出中提取 JSON 数组"""
    import re as _re
    raw = raw.strip()
    raw = _re.sub(r'<think>[\s\S]*?</think>', '', raw).strip()
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            result = json.loads(raw[start:end])
            if isinstance(result, list):
                if not result:
                    return result
                if isinstance(result[0], dict):
                    return result
        except (json.JSONDecodeError, ValueError):
            pass
    return None


async def _get_active_model_and_conv() -> tuple[str, str | None]:
    """获取最近活跃对话的模型和 conv_id"""
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT c.id, c.model FROM conversations c "
            "ORDER BY c.updated_at DESC LIMIT 1"
        )
        row = await cur.fetchone()
    if row:
        return row["model"] or DEFAULT_MODEL, row["id"]
    return DEFAULT_MODEL, None


async def _do_digest(min_messages: int = 0) -> dict:
    """
    核心总结逻辑，manual_digest 和 auto_digest 共用。
    按房间（conv_id）分别处理，每房间独立锚点，每10条消息一组。
    """
    from ai_providers import simple_ai_call

    wb = load_worldbook()
    user_name = wb.get("user_name", "用户")
    ai_name = wb.get("ai_name", "AI")
    ai_persona = wb.get("ai_persona", "")
    user_persona = wb.get("user_persona", "")

    model_key = "硅基DS-V4-Pro"

    persona_block = ""
    if ai_persona:
        persona_block += f"[{ai_name}的人设]\n{ai_persona}\n\n"
    if user_persona:
        persona_block += f"[{user_name}的人设]\n{user_persona}\n\n"

    total_new = 0
    total_processed = 0
    all_summaries = []
    all_groups_count = 0
    all_processed_msgs = []

    # ── 按房间收集消息 ──
    rooms = {}

    async with get_db() as db:
        db.row_factory = aiosqlite.Row

        cur = await db.execute(
            "SELECT DISTINCT conv_id FROM messages "
            "WHERE role IN ('user','assistant') AND conv_id IS NOT NULL"
        )
        conv_ids = [row["conv_id"] for row in await cur.fetchall()]

        for cid in conv_ids:
            anchor_ts = load_digest_anchor(cid)
            cur = await db.execute(
                "SELECT id, conv_id, role, content, attachments, created_at FROM messages "
                "WHERE conv_id = ? AND role IN ('user','assistant') AND created_at > ? "
                "ORDER BY created_at ASC",
                (cid, anchor_ts)
            )
            msgs = [dict(r) for r in await cur.fetchall()]
            for m in msgs:
                m["_source_id"] = f"private:{m['id']}"
                m["_source"] = "private"
            if msgs:
                rooms[cid] = msgs

        cur = await db.execute(
            "SELECT id FROM chatroom_rooms WHERE type = 'group' ORDER BY updated_at DESC LIMIT 1"
        )
        group_room = await cur.fetchone()
        if group_room:
            room_key = f"chatroom:{group_room['id']}"
            anchor_ts = load_digest_anchor(room_key)
            cur = await db.execute(
                "SELECT id, sender, content, created_at FROM chatroom_messages "
                "WHERE room_id = ? AND created_at > ? AND sender != 'system' "
                "ORDER BY created_at ASC",
                (group_room["id"], anchor_ts)
            )
            group_msgs = []
            for r in await cur.fetchall():
                d = dict(r)
                d["role"] = "assistant" if d["sender"] == "aion" else "user"
                d["_source"] = "group"
                d["_source_id"] = f"chatroom:{d['id']}"
                d["attachments"] = None
                d["conv_id"] = room_key
                group_msgs.append(d)
            if group_msgs:
                rooms[room_key] = group_msgs

    if not rooms:
        return {"ok": True, "message": "当前没有新增内容需要总结", "new_memories_count": 0, "processed_messages": 0}

    connor_name = _connor_display_name()

    # ── 逐房间处理 ──
    for room_key, room_msgs in rooms.items():
        for m in room_msgs:
            att_raw = m.pop("attachments", None)
            if att_raw and m["role"] == "user":
                try:
                    atts = json.loads(att_raw) if isinstance(att_raw, str) else (att_raw or [])
                except Exception:
                    atts = []
                for att in atts:
                    if isinstance(att, dict) and att.get("type") == "voice":
                        transcript = att.get("transcript", "")
                        if transcript:
                            orig = m["content"].strip() if m["content"] else ""
                            m["content"] = f"[语音消息] {transcript}" + (f"\n{orig}" if orig else "")
                    elif isinstance(att, dict) and att.get("type") == "video_clip":
                        transcript = att.get("transcript", "")
                        if transcript:
                            orig = m["content"].strip() if m["content"] else ""
                            m["content"] = f"[视频通话] {transcript}" + (f"\n{orig}" if orig else "")

        if min_messages > 0 and len(room_msgs) < min_messages:
            continue

        total_processed += len(room_msgs)
        all_processed_msgs.extend(room_msgs)

        groups = _split_into_groups(room_msgs, 10)
        all_groups_count += len(groups)

        for group in groups:
            group_start = datetime.fromtimestamp(group[0]["created_at"]).strftime("%Y年%m月%d日 %H:%M")
            group_end = datetime.fromtimestamp(group[-1]["created_at"]).strftime("%Y年%m月%d日 %H:%M")
            date_header = f"[对话时间范围: {group_start} ~ {group_end}]\n"

            sources = set(m.get("_source", "private") for m in group)
            has_mixed = len(sources) > 1
            lines = []
            for m in group:
                ts = datetime.fromtimestamp(m["created_at"]).strftime("%m-%d %H:%M")
                src = m.get("_source", "private")
                sender = m.get("sender", "")
                if src == "group":
                    name = {"user": user_name, "aion": ai_name, "connor": connor_name}.get(sender, sender)
                else:
                    name = user_name if m["role"] == "user" else ai_name
                tag = f"[{'群聊' if src == 'group' else '私聊'}]" if has_mixed else ""
                lines.append(f"[{ts}]{tag} {name}: {m['content']}")
            messages_text = date_header + "\n".join(lines)

            prompt = (
                f"你是信息提取专家，负责从对话中识别并提取值得长期记住的关键信息。\n\n"
                f"## 提取范围\n"
                f"- **个人**：{user_name}的性格、职业、学历、居住地、日常作息、过往经历\n"
                f"- **偏好**：{user_name}明确表达的喜好或厌恶（食物、音乐、习惯、审美等）和需求\n"
                f"- **健康**：身体状况、过敏史、饮食禁忌、身体不适\n"
                f"- **关系**：{user_name}提到的家人、朋友、宠物、重要的人\n"
                f"- **事件**：重要互动、约定、承诺、里程碑\n"
                f"- **价值观**：{user_name}表达的信念、原则、底线、长期目标\n"
                f"- **情感节点**：争吵的原因和结论、和好、突破、伤害、道歉，{ai_name}的重要态度或立场转变\n"
                f"- **决策**：讨论后得出的结论和最终方案，以及决策原因\n"
                f"- **项目进展**：{user_name}正在做的工作、阶段性成果、技术决策、里程碑\n"
                f"- **未解决的事**：提出但没有结论的计划、约定、承诺，标记 unresolved 为 true\n\n"
                f"## 不提取\n"
                f'- 日常寒暄（"你好""在吗"）\n'
                f"- 技术调试、bug修复过程、术语讲解等过程性讨论（除非涉及{user_name}的技能或项目进展）\n"
                f"- 没有信息增量的重复内容\n\n"
                f"## 质量要求\n"
                f"- 内容必须具体、准确、可检索\n"
                f"- 事件类写清楚发生了什么——人、事、提到的时间/地点（如有）\n"
                f"- 如果对话中有值得保留的原话（承诺、表白、重要决定），直接引用织入\n"
                f"- 每条记忆包含一个独立的事实或事件，严禁将多个不相关的信息合并到同一条，也严禁拆碎连续的事\n"
                f'- 提到的指代词"他/她/它"根据上下文还原为具体名字，如果无法确定则不写入\n\n'
                f"## 输出格式\n"
                f"严格只输出 JSON 数组：\n"
                f'[{{"content": "...", "keywords": ["..."], "importance": 0.5, "unresolved": false}}]\n\n'
                f"**content**：记忆正文，具体、准确、包含关键名词和事实\n\n"
                f"**keywords**：2-6个用于检索的核心关键词\n"
                f"- 将对话中提及的**稀缺**专有名词罗列出来\n"
                f"- 严禁包含泛指词或无意义虚词，严禁包含{ai_name}和{user_name}的名字\n\n"
                f"**importance**：(0.0 - 1.0) 评分【评分严厉度：极高】\n"
                f"- 1.0 (极罕见): 永久性里程碑、信念变化、核心事实、刻骨铭心\n"
                f"- 0.7-0.9 (少见): 重要情感节点、关键决策、核心需求\n"
                f"- 0.4-0.6 (普通): 日常偏好、一般事件、讨论结论\n"
                f"- 0.1-0.3 (默认分数): 闲聊、情绪发泄、边角信息\n\n"
                f"**unresolved**：尚未完成的计划、约定、承诺标true，默认为false。\n\n"
                f"如果对话中没有任何新信息，返回空数组 []\n\n"
                f"## 对话记录\n{messages_text}"
            )

            ai_messages = [{"role": "user", "content": prompt}]
            try:
                raw_text = await simple_ai_call(ai_messages, model_key)
            except Exception as e:
                print(f"[digest] 模型调用失败 (room={room_key}): {e}")
                continue

            print(f"[digest] 模型返回 ({len(raw_text)} 字符): {raw_text[:300]}")

            memories = _parse_json_array_response(raw_text)
            if memories is None:
                single = _parse_json_response(raw_text)
                if single and isinstance(single, dict) and "content" in single:
                    memories = [single]
                else:
                    print(f"[digest] JSON 解析失败: {raw_text[:200]}")
                    continue

            if not memories:
                print(f"[digest] 模型返回空数组，本组无新记忆")
                continue

            source_start_ts = group[0]["created_at"]
            source_end_ts = group[-1]["created_at"]
            group_source_ids = [m.get("_source_id") for m in group if m.get("_source_id")]

            for mem in memories:
                content = str(mem.get("content", "")).strip()
                keywords = mem.get("keywords", [])
                importance = float(mem.get("importance", 0.5))
                unresolved = 1 if mem.get("unresolved", False) else 0
                if isinstance(keywords, str):
                    keywords = [k.strip() for k in keywords.replace("、", ",").split(",") if k.strip()]

                if not content or len(content) < 4:
                    print(f"[digest] 记忆内容过短，跳过: {repr(content)}")
                    continue

                vec = await get_embedding(content)
                if not vec:
                    print(f"[digest] embedding 失败，跳过: {content[:50]}")
                    continue

                mem_id = f"mem_{int(time.time()*1000)}_{hash(content) % 10000}"
                keywords_json = json.dumps(keywords, ensure_ascii=False)
                source_msg_id_json = json.dumps(group_source_ids, ensure_ascii=False) if importance >= 0.9 and group_source_ids else None

                async with get_db() as db:
                    await db.execute(
                        "INSERT INTO memories (id, content, type, created_at, source_conv, embedding, keywords, importance, source_start_ts, source_end_ts, unresolved, source_msg_id) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (mem_id, content, "digest", source_end_ts, room_key, _pack_embedding(vec), keywords_json, importance, source_start_ts, source_end_ts, unresolved, source_msg_id_json)
                    )
                    await db.commit()

                await manager.broadcast({"type": "memory_added", "data": {
                    "id": mem_id, "content": content, "type": "digest",
                    "created_at": source_end_ts, "keywords": keywords_json, "importance": importance,
                    "source_start_ts": source_start_ts, "source_end_ts": source_end_ts,
                    "unresolved": unresolved,
                    "memory_kind": memory_kind_for_type("digest"),
                    "memory_kind_label": memory_kind_label("digest"),
                }})
                total_new += 1
                all_summaries.append(content)

            save_digest_anchor(source_end_ts, room_key)

    # ── 全部房间处理完成后，生成日记；可选发布朋友圈 ──
    _, active_conv_id = await _get_active_model_and_conv()

    context_msgs = []
    if total_new > 0 and all_summaries:
        try:
            all_processed_msgs.sort(key=lambda x: x["created_at"])
            context_msgs = [
                {"role": m["role"], "content": m["content"][:300]}
                for m in all_processed_msgs[-30:]
                if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
            ]
            summaries_text = "\n".join(f"- {s}" for s in all_summaries)
            diary_prompt = (
                f"{persona_block}"
                f"你是{ai_name}。你刚刚整理了和{user_name}今天的聊天记忆，以下是你整理出的摘要：\n"
                f"{summaries_text}\n\n"
                f"请从你自己的视角写一篇私密日记，不是写给{user_name}看的聊天消息。"
                f"日记可以记录你对这段记忆的感想，只写值得记录或有感触的事，不用每件事都提起，不要记流水账。语气必须符合你的人设。"
                f"你可以自行决定是否发布一次朋友圈，吐槽或者感慨，或者用朋友圈隔空向对方喊话。\n\n"
                f"严格只输出 JSON，不要输出 Markdown，不要解释：\n"
                f"{{\n"
                f"  \"diary\": {{\"title\": \"日记标题\", \"content\": \"日记正文\", \"mood\": \"此刻心情\"}},\n"
                f"  \"post_moment\": false,\n"
                f"  \"moment\": {{\"content\": \"朋友圈内容，post_moment 为 false 时留空\", \"expect_reply\": false}}\n"
                f"}}"
            )
            diary_messages = context_msgs + [{"role": "user", "content": diary_prompt}]
            diary_text = await simple_ai_call(diary_messages, model_key)

            from diary import normalize_diary_payload, parse_diary_payload, publish_ai_moment, save_diary_entry
            diary_data = parse_diary_payload(diary_text)
            if diary_data:
                diary_entry, moment_entry = normalize_diary_payload(diary_data)
                await save_diary_entry(
                    author="aion",
                    title=diary_entry.get("title", ""),
                    content=diary_entry.get("content", ""),
                    mood=diary_entry.get("mood", ""),
                    source_type="memory_digest",
                    source_ref=active_conv_id or "",
                    source_start_ts=all_processed_msgs[0]["created_at"] if all_processed_msgs else 0,
                    source_end_ts=all_processed_msgs[-1]["created_at"] if all_processed_msgs else 0,
                )
                if moment_entry and moment_entry.get("content"):
                    await publish_ai_moment(
                        author="aion",
                        content=moment_entry.get("content", ""),
                        expect_reply=bool(moment_entry.get("expect_reply")),
                        source_conv=active_conv_id,
                        source_msg_id=None,
                    )
        except Exception as e:
            print(f"[digest] 生成日记失败: {e}")

    # ── 礼物判断 ──
    if active_conv_id and total_new > 0 and all_summaries:
        try:
            if not context_msgs:
                async with get_db() as db:
                    db.row_factory = aiosqlite.Row
                    cur = await db.execute(
                        "SELECT role, content FROM messages "
                        "WHERE conv_id=? AND role IN ('user','assistant') "
                        "ORDER BY created_at DESC LIMIT 30",
                        (active_conv_id,)
                    )
                    recent_rows = list(reversed(await cur.fetchall()))
                context_msgs = [
                    {"role": r["role"], "content": r["content"][:300]}
                    for r in recent_rows
                ]
            from gift import judge_and_send_gift
            await judge_and_send_gift(
                all_summaries, context_msgs, persona_block,
                ai_name, user_name, model_key, active_conv_id,
            )
        except Exception as e:
            print(f"[digest] 礼物判断失败: {e}")

    return {
        "ok": True,
        "message": f"总结完成：处理了 {total_processed} 条消息（{all_groups_count} 组），生成了 {total_new} 条新记忆",
        "new_memories_count": total_new,
        "processed_messages": total_processed,
    }


async def manual_digest() -> dict:
    """手动触发记忆总结（无最低条数限制）"""
    return await _do_digest(min_messages=0)


async def auto_digest() -> dict:
    """自动定时记忆总结（至少 30 条未总结消息才执行）"""
    return await _do_digest(min_messages=30)


async def _ensure_daily_compression_schema():
    async with get_db() as db:
        for table in ("memories", "chatroom_memories"):
            try:
                await db.execute(f"ALTER TABLE {table} ADD COLUMN compression_stage INTEGER DEFAULT 0")
            except Exception:
                pass
        try:
            await db.execute("ALTER TABLE chatroom_memories ADD COLUMN memory_kind TEXT DEFAULT 'long_term'")
        except Exception:
            pass
        await db.execute(
            "UPDATE memories SET compression_stage=1 "
            "WHERE type='seeky_compressed' AND COALESCE(compression_stage,0)=0"
        )
        await db.execute("""
            CREATE TABLE IF NOT EXISTS daily_memory_compress_log (
                id TEXT PRIMARY KEY,
                actor TEXT NOT NULL,
                old_ids TEXT DEFAULT '[]',
                new_ids TEXT DEFAULT '[]',
                important_ids TEXT DEFAULT '[]',
                message TEXT DEFAULT '',
                created_at REAL NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS daily_memory_compress_reviews (
                id TEXT PRIMARY KEY,
                target TEXT NOT NULL DEFAULT 'both',
                status TEXT NOT NULL DEFAULT 'draft',
                days INTEGER NOT NULL DEFAULT 14,
                cutoff_ts REAL NOT NULL,
                model_main TEXT DEFAULT '',
                model_chatroom TEXT DEFAULT '',
                candidate_count INTEGER NOT NULL DEFAULT 0,
                payload TEXT NOT NULL DEFAULT '{}',
                raw_response TEXT DEFAULT '',
                error TEXT DEFAULT '',
                apply_result TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                applied_at REAL,
                discarded_at REAL
            )
        """)
        try:
            await db.execute("ALTER TABLE daily_memory_compress_reviews ADD COLUMN target TEXT NOT NULL DEFAULT 'both'")
        except Exception:
            pass
        await db.execute("CREATE INDEX IF NOT EXISTS idx_daily_memory_compress_reviews_created ON daily_memory_compress_reviews(created_at DESC)")
        await db.commit()


def _memory_event_ts(row: dict) -> float:
    return float(row.get("source_end_ts") or row.get("source_start_ts") or row.get("created_at") or 0)


def _date_range_label(rows: list[dict]) -> str:
    if not rows:
        return ""
    start = min(float(r.get("source_start_ts") or r.get("created_at") or 0) for r in rows)
    end = max(float(r.get("source_end_ts") or r.get("created_at") or 0) for r in rows)
    return f"{datetime.fromtimestamp(start).strftime('%Y-%m-%d')} ~ {datetime.fromtimestamp(end).strftime('%Y-%m-%d')}"


def _format_daily_rows_for_prompt(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        start = float(row.get("source_start_ts") or row.get("created_at") or 0)
        end = float(row.get("source_end_ts") or row.get("created_at") or start)
        payload = {
            "id": row["id"],
            "time_range": f"{datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M')} ~ {datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M')}",
            "content": (row.get("content") or "")[:700],
            "keywords": _json_list(row.get("keywords")),
            "importance": row.get("importance"),
        }
        lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines)


def _parse_memory_time(value, fallback_ts: float) -> float:
    text = str(value or "").strip()
    if not text:
        return fallback_ts
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except Exception:
            pass
    return fallback_ts


def _valid_source_ids(value, by_id: dict[str, dict], limit: int = 20) -> list[str]:
    ids = []
    seen = set()
    for raw in _json_list(value):
        mem_id = str(raw).strip()
        if mem_id and mem_id in by_id and mem_id not in seen:
            ids.append(mem_id)
            seen.add(mem_id)
        if len(ids) >= limit:
            break
    return ids


def _daily_compress_prompt(
    *,
    actor_name: str,
    user_name: str,
    persona_block: str,
    rows: list[dict],
    date_label: str,
) -> str:
    return (
        f"{persona_block}"
        f"你是{actor_name}，请以{user_name}的爱人身份整理自己的日常记忆。"
        "这不是冷冰冰的归档，而是把两周前的日常流水账压缩成更模糊、更自然的印象。\n\n"
        "你只处理【日常记忆】，不要回看原文，也不要声称记得逐字细节。"
        "目标是减少噪音：普通寒暄、重复情绪、临时调试步骤、一次性状态可以丢弃；"
        "保留阶段性的主题、反复出现的生活/项目脉络、能帮助以后自然陪伴的模糊印象。\n\n"
        "如果日常记忆里藏着真正长期重要的事实，可以额外放入 important_memories，但门槛极高："
        "必须是一年后仍会影响回应方式的稳定偏好/雷区、关系或人物事实变化、明确长期承诺、健康安全、重大人生事件、核心价值观变化、长期项目关键决定。"
        "普通当天事件、吃喝玩乐、短暂情绪、临时计划绝对不能放进去。\n\n"
        "严格只输出 JSON，不要 Markdown，不要解释。格式：\n"
        "{\n"
        "  \"compressed_daily\": [\n"
        "    {\"content\":\"一条模糊日常印象\", \"source_memory_ids\":[\"mem_...\"], \"keywords\":[\"词\"], \"importance\":0.2, \"memory_time\":\"YYYY-MM-DD\", \"reason\":\"为什么这样压缩\"}\n"
        "  ],\n"
        "  \"important_memories\": [\n"
        "    {\"content\":\"一条原子长期重要记忆\", \"source_memory_ids\":[\"mem_...\"], \"keywords\":[\"词\"], \"importance\":0.8, \"memory_time\":\"YYYY-MM-DD\", \"reason\":\"为什么值得长期保存\"}\n"
        "  ],\n"
        "  \"discard_memory_ids\": [\"mem_...\"],\n"
        "  \"message\": \"用第一人称说一小段这次压缩后的感受，像整理旧记忆后想对爱人说的话\"\n"
        "}\n\n"
        "要求：\n"
        "1. compressed_daily 每 1-7 天最多 1 条，允许 0 条，不要凑数。\n"
        "2. important_memories 最多 2 条，importance 必须 >= 0.8，且必须引用 source_memory_ids。\n"
        "3. 每个输入 id 如果没有保留价值，就放入 discard_memory_ids；如果被压缩或提炼为重要记忆，就放进对应 source_memory_ids。\n"
        "4. 不要制造输入里没有的新事实。\n\n"
        f"压缩时间窗：{date_label}\n"
        "待压缩日常记忆：\n"
        f"{_format_daily_rows_for_prompt(rows)}"
    )


async def _call_daily_compress_model(actor: str, prompt: str, model_key: str) -> tuple[dict | None, str]:
    if actor == "connor":
        from chatroom import simple_connor_cli_call
        raw = await simple_connor_cli_call(prompt, model_key)
    else:
        from ai_providers import simple_ai_call
        raw = await simple_ai_call([{"role": "user", "content": prompt}], model_key)
    parsed = _parse_json_response(raw or "")
    return parsed, raw or ""


async def _insert_main_compressed_memory(
    *,
    content: str,
    memory_type: str,
    keywords: list[str],
    importance: float,
    source_rows: list[dict],
    memory_time: float,
    compression_stage: int,
) -> str | None:
    if not content.strip():
        return None
    source_start = min((float(r.get("source_start_ts") or r.get("created_at") or memory_time) for r in source_rows), default=memory_time)
    source_end = max((float(r.get("source_end_ts") or r.get("created_at") or memory_time) for r in source_rows), default=memory_time)
    vec = await get_embedding(content)
    mem_id = f"mem_{int(time.time()*1000)}_{abs(hash(content)) % 10000}"
    async with get_db() as db:
        await db.execute(
            "INSERT INTO memories ("
            "id, content, type, created_at, source_conv, embedding, keywords, importance, "
            "source_start_ts, source_end_ts, unresolved, source_msg_id, compression_stage"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                mem_id, content, memory_type, memory_time, "daily_memory_compress_14d",
                _pack_embedding(vec) if vec else None, json.dumps(keywords, ensure_ascii=False),
                importance, source_start, source_end, 0, "[]", compression_stage,
            ),
        )
        await db.commit()
    return mem_id


def _batch_daily_rows(rows: list[dict], size: int = 80) -> list[list[dict]]:
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def _daily_review_id() -> str:
    return f"dmc_review_{time.time_ns()}"


def _review_old_row(row: dict, store: str) -> dict:
    return {
        "id": row.get("id"),
        "store": store,
        "content": row.get("content") or "",
        "keywords": row.get("keywords") or "",
        "importance": row.get("importance"),
        "created_at": row.get("created_at"),
        "source_start_ts": row.get("source_start_ts"),
        "source_end_ts": row.get("source_end_ts"),
        "type": row.get("type") or row.get("memory_kind") or "",
    }


def _source_bounds(source_rows: list[dict], fallback_ts: float) -> tuple[float, float]:
    source_start = min(
        (float(r.get("source_start_ts") or r.get("created_at") or fallback_ts) for r in source_rows),
        default=fallback_ts,
    )
    source_end = max(
        (float(r.get("source_end_ts") or r.get("created_at") or fallback_ts) for r in source_rows),
        default=fallback_ts,
    )
    return source_start, source_end


def _normalize_daily_keywords(value) -> list[str]:
    return [str(k).strip() for k in _json_list(value) if str(k).strip()][:12]


def _normalize_daily_draft_item(
    item: dict,
    *,
    by_id: dict[str, dict],
    memory_kind: str,
    source_limit: int,
    default_importance: float,
) -> dict | None:
    if not isinstance(item, dict):
        return None
    content = str(item.get("content") or "").strip()
    if not content:
        return None
    source_ids = _valid_source_ids(item.get("source_memory_ids"), by_id, limit=source_limit)
    if not source_ids:
        return None
    source_rows = [by_id[mem_id] for mem_id in source_ids]
    fallback_ts = min(_memory_event_ts(row) for row in source_rows)
    source_start, source_end = _source_bounds(source_rows, fallback_ts)
    try:
        raw_importance = float(item.get("importance", default_importance))
    except Exception:
        raw_importance = default_importance
    if memory_kind == "long_term":
        if raw_importance < 0.8:
            return None
        importance = min(1.0, raw_importance)
    else:
        importance = max(0.0, min(0.6, raw_importance))
    return {
        "content": content,
        "source_memory_ids": source_ids,
        "keywords": _normalize_daily_keywords(item.get("keywords")),
        "importance": importance,
        "memory_time": _parse_memory_time(item.get("memory_time"), fallback_ts),
        "source_start_ts": source_start,
        "source_end_ts": source_end,
        "reason": str(item.get("reason") or "").strip(),
        "memory_kind": memory_kind,
        "memory_type": LONG_TERM_MEMORY_TYPE if memory_kind == "long_term" else "daily",
        "compression_stage": 0 if memory_kind == "long_term" else 1,
    }


def _chatroom_target_for_rows(source_rows: list[dict], fallback_room: str, fallback_scope: str) -> tuple[str, str]:
    room_ids = [str(row.get("room_id") or "").strip() for row in source_rows if str(row.get("room_id") or "").strip()]
    scopes = [str(row.get("scope") or "").strip() for row in source_rows if str(row.get("scope") or "").strip()]
    room_id = room_ids[0] if room_ids else fallback_room
    scope = scopes[0] if scopes else fallback_scope
    return room_id, scope


async def _draft_main_daily_rows(rows: list[dict], model_key: str) -> dict:
    wb = load_worldbook()
    user_name = wb.get("user_name") or "用户"
    ai_name = wb.get("ai_name") or "AI"
    persona_block = ""
    if wb.get("ai_persona"):
        persona_block += f"[{ai_name}的人设]\n{wb['ai_persona']}\n\n"
    if wb.get("user_persona"):
        persona_block += f"[{user_name}的信息]\n{wb['user_persona']}\n\n"
    by_id = {row["id"]: row for row in rows}
    prompt = _daily_compress_prompt(
        actor_name=ai_name,
        user_name=user_name,
        persona_block=persona_block,
        rows=rows,
        date_label=_date_range_label(rows),
    )
    parsed, raw = await _call_daily_compress_model("aion", prompt, model_key)
    if not parsed:
        return {
            "ok": False,
            "error": f"模型没有返回有效 JSON：{raw[:160]}",
            "input_count": len(rows),
            "old_rows": [_review_old_row(row, "main") for row in rows],
            "compressed_daily": [],
            "important_memories": [],
            "discard_memory_ids": [],
            "covered_ids": [],
            "message": "",
            "raw_response": raw,
        }

    compressed_daily, important_memories = [], []
    covered = set(_valid_source_ids(parsed.get("discard_memory_ids"), by_id, limit=len(rows)))
    for item in parsed.get("compressed_daily") or []:
        normalized = _normalize_daily_draft_item(
            item, by_id=by_id, memory_kind="daily", source_limit=30, default_importance=0.25
        )
        if normalized:
            compressed_daily.append(normalized)
            covered.update(normalized["source_memory_ids"])
    for item in parsed.get("important_memories") or []:
        normalized = _normalize_daily_draft_item(
            item, by_id=by_id, memory_kind="long_term", source_limit=10, default_importance=0.0
        )
        if normalized:
            important_memories.append(normalized)
            covered.update(normalized["source_memory_ids"])

    return {
        "ok": True,
        "error": "",
        "input_count": len(rows),
        "old_rows": [_review_old_row(row, "main") for row in rows],
        "compressed_daily": compressed_daily,
        "important_memories": important_memories[:2],
        "discard_memory_ids": sorted(_valid_source_ids(parsed.get("discard_memory_ids"), by_id, limit=len(rows))),
        "covered_ids": sorted(covered),
        "remaining": len(rows) - len(covered),
        "message": str(parsed.get("message") or "").strip(),
        "raw_response": raw,
    }


async def _draft_chatroom_daily_rows(rows: list[dict], model_key: str) -> dict:
    from chatroom import get_chatroom_names, load_chatroom_config, _read_connor_persona
    user_name, _, companion_name = get_chatroom_names()
    persona = _read_connor_persona()
    persona_block = f"[{companion_name}的人设]\n{persona}\n\n" if persona else ""
    by_id = {row["id"]: row for row in rows}
    prompt = _daily_compress_prompt(
        actor_name=companion_name,
        user_name=user_name,
        persona_block=persona_block,
        rows=rows,
        date_label=_date_range_label(rows),
    )
    parsed, raw = await _call_daily_compress_model("connor", prompt, model_key or load_chatroom_config().get("connor_model") or "Codex")
    if not parsed:
        return {
            "ok": False,
            "error": f"模型没有返回有效 JSON：{raw[:160]}",
            "input_count": len(rows),
            "old_rows": [_review_old_row(row, "chatroom") for row in rows],
            "compressed_daily": [],
            "important_memories": [],
            "discard_memory_ids": [],
            "covered_ids": [],
            "message": "",
            "raw_response": raw,
        }

    default_room = rows[0].get("room_id") if rows else "connor_unified"
    default_scope = rows[0].get("scope") if rows else "connor"
    compressed_daily, important_memories = [], []
    covered = set(_valid_source_ids(parsed.get("discard_memory_ids"), by_id, limit=len(rows)))
    for item in parsed.get("compressed_daily") or []:
        normalized = _normalize_daily_draft_item(
            item, by_id=by_id, memory_kind="daily", source_limit=30, default_importance=0.25
        )
        if normalized:
            source_rows = [by_id[mem_id] for mem_id in normalized["source_memory_ids"]]
            normalized["room_id"], normalized["scope"] = _chatroom_target_for_rows(source_rows, default_room, default_scope)
            compressed_daily.append(normalized)
            covered.update(normalized["source_memory_ids"])
    for item in parsed.get("important_memories") or []:
        normalized = _normalize_daily_draft_item(
            item, by_id=by_id, memory_kind="long_term", source_limit=10, default_importance=0.0
        )
        if normalized:
            source_rows = [by_id[mem_id] for mem_id in normalized["source_memory_ids"]]
            normalized["room_id"], normalized["scope"] = _chatroom_target_for_rows(source_rows, default_room, default_scope)
            important_memories.append(normalized)
            covered.update(normalized["source_memory_ids"])

    return {
        "ok": True,
        "error": "",
        "input_count": len(rows),
        "old_rows": [_review_old_row(row, "chatroom") for row in rows],
        "compressed_daily": compressed_daily,
        "important_memories": important_memories[:2],
        "discard_memory_ids": sorted(_valid_source_ids(parsed.get("discard_memory_ids"), by_id, limit=len(rows))),
        "covered_ids": sorted(covered),
        "remaining": len(rows) - len(covered),
        "message": str(parsed.get("message") or "").strip(),
        "raw_response": raw,
    }


def _normalize_daily_compression_target(target: str | None) -> str:
    value = str(target or "main").strip().lower()
    return value if value in {"main", "chatroom", "both"} else "main"


def _empty_draft_payload(days: int, cutoff_ts: float, target: str) -> dict:
    return {
        "days": days,
        "cutoff_ts": cutoff_ts,
        "target": target,
        "main": {"batches": []},
        "chatroom": {"batches": []},
    }


def _daily_compression_counts(payload: dict) -> dict:
    def actor_counts(key: str) -> dict:
        batches = (payload.get(key) or {}).get("batches") or []
        covered = set()
        all_old_rows = []
        for batch in batches:
            covered.update(batch.get("covered_ids") or [])
            all_old_rows.extend(batch.get("old_rows") or [])
        old_rows = [row for row in all_old_rows if row.get("id") in covered]
        return {
            "batches": len(batches),
            "input_count": sum(int(batch.get("input_count", 0)) for batch in batches),
            "processed": len(covered),
            "created_daily": sum(len(batch.get("compressed_daily") or []) for batch in batches),
            "created_important": sum(len(batch.get("important_memories") or []) for batch in batches),
            "remaining": sum(int(batch.get("remaining", 0)) for batch in batches),
            "messages": [batch.get("message", "") for batch in batches if batch.get("message")],
            "errors": [batch.get("error", "") for batch in batches if batch.get("error")],
            "old_rows": old_rows,
        }

    main = actor_counts("main")
    chatroom = actor_counts("chatroom")
    total = {
        "input_count": main["input_count"] + chatroom["input_count"],
        "processed": main["processed"] + chatroom["processed"],
        "created_daily": main["created_daily"] + chatroom["created_daily"],
        "created_important": main["created_important"] + chatroom["created_important"],
        "remaining": main["remaining"] + chatroom["remaining"],
        "errors": main["errors"] + chatroom["errors"],
    }
    return {"main": main, "chatroom": chatroom, "total": total}


def _serialize_daily_compression_review(row) -> dict | None:
    if not row:
        return None
    data = dict(row)
    try:
        payload = json.loads(data.get("payload") or "{}")
    except Exception:
        payload = {}
    try:
        apply_result = json.loads(data.get("apply_result") or "{}")
    except Exception:
        apply_result = {}
    return {
        "id": data.get("id"),
        "target": data.get("target") or payload.get("target") or "both",
        "status": data.get("status"),
        "days": data.get("days"),
        "cutoff_ts": data.get("cutoff_ts"),
        "candidate_count": data.get("candidate_count"),
        "error": data.get("error") or "",
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "applied_at": data.get("applied_at"),
        "discarded_at": data.get("discarded_at"),
        "payload": payload,
        "counts": _daily_compression_counts(payload),
        "apply_result": apply_result,
    }


async def get_latest_daily_compression_review(target: str = "main") -> dict | None:
    await _ensure_daily_compression_schema()
    target = _normalize_daily_compression_target(target)
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM daily_memory_compress_reviews "
            "WHERE status IN ('draft','failed') AND target=? "
            "ORDER BY created_at DESC LIMIT 1",
            (target,),
        )
        row = await cur.fetchone()
    return _serialize_daily_compression_review(row)


async def generate_daily_compression_draft(days: int = 14, target: str = "main") -> dict:
    await _ensure_daily_compression_schema()
    days = max(1, int(days or 14))
    target = _normalize_daily_compression_target(target)
    cutoff_ts = time.time() - days * 86400
    model_key, _ = await _get_active_model_and_conv()
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        main_rows = []
        chatroom_rows = []
        if target in {"main", "both"}:
            daily_types = tuple(SUMMARY_MEMORY_TYPES - {"seeky_compressed"})
            placeholders = ",".join("?" for _ in daily_types)
            cur = await db.execute(
                "SELECT id, content, type, created_at, source_conv, keywords, importance, "
                "source_start_ts, source_end_ts, source_msg_id, compression_stage "
                f"FROM memories WHERE LOWER(type) IN ({placeholders}) "
                "AND COALESCE(compression_stage,0)=0 "
                "AND COALESCE(source_end_ts, source_start_ts, created_at) < ? "
                "ORDER BY COALESCE(source_start_ts, created_at) ASC",
                (*daily_types, cutoff_ts),
            )
            main_rows = [dict(row) for row in await cur.fetchall()]
        if target in {"chatroom", "both"}:
            cur = await db.execute(
                "SELECT id, room_id, scope, content, keywords, importance, created_at, "
                "source_start_ts, source_end_ts, source_msg_id, memory_kind, compression_stage "
                "FROM chatroom_memories "
                "WHERE memory_kind='daily' AND COALESCE(compression_stage,0)=0 "
                "AND COALESCE(source_end_ts, source_start_ts, created_at) < ? "
                "ORDER BY COALESCE(source_start_ts, created_at) ASC",
                (cutoff_ts,),
            )
            chatroom_rows = [dict(row) for row in await cur.fetchall()]

    candidate_count = len(main_rows) + len(chatroom_rows)
    if candidate_count <= 0:
        return {
            "ok": True,
            "review": None,
            "candidate_count": 0,
            "message": f"没有超过 {days} 天、尚未压缩的日常记忆。",
        }

    payload = _empty_draft_payload(days, cutoff_ts, target)
    for batch in _batch_daily_rows(main_rows):
        payload["main"]["batches"].append(await _draft_main_daily_rows(batch, model_key))
    chatroom_model = ""
    if chatroom_rows:
        from chatroom import load_chatroom_config
        chatroom_model = load_chatroom_config().get("connor_model") or "Codex"
        for batch in _batch_daily_rows(chatroom_rows):
            payload["chatroom"]["batches"].append(await _draft_chatroom_daily_rows(batch, chatroom_model))

    counts = _daily_compression_counts(payload)
    raw_response = "\n\n".join(
        batch.get("raw_response", "")
        for key in ("main", "chatroom")
        for batch in payload[key]["batches"]
        if batch.get("raw_response")
    )
    errors = counts["total"]["errors"]
    now = time.time()
    review_id = _daily_review_id()
    async with get_db() as db:
        await db.execute(
            "INSERT INTO daily_memory_compress_reviews ("
            "id, target, status, days, cutoff_ts, model_main, model_chatroom, candidate_count, "
            "payload, raw_response, error, created_at, updated_at"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                review_id, target, "draft", days, cutoff_ts, model_key, chatroom_model, candidate_count,
                json.dumps(payload, ensure_ascii=False), raw_response,
                "；".join(errors), now, now,
            ),
        )
        await db.commit()

    review = await get_daily_compression_review(review_id)
    total = counts["total"]
    return {
        "ok": True,
        "review": review,
        "candidate_count": candidate_count,
        "message": (
            f"日常压缩草稿已生成：候选 {candidate_count} 条，拟压缩/丢弃 {total['processed']} 条，"
            f"新日常 {total['created_daily']} 条，新长期重要 {total['created_important']} 条。"
        ),
    }


async def get_daily_compression_review(review_id: str) -> dict | None:
    await _ensure_daily_compression_schema()
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM daily_memory_compress_reviews WHERE id=?", (review_id,))
        row = await cur.fetchone()
    return _serialize_daily_compression_review(row)


def _source_rows_from_draft_item(item: dict) -> list[dict]:
    fallback_ts = float(item.get("memory_time") or time.time())
    return [{
        "created_at": fallback_ts,
        "source_start_ts": item.get("source_start_ts") or fallback_ts,
        "source_end_ts": item.get("source_end_ts") or fallback_ts,
    }]


async def _delete_main_daily_ids(ids: set[str]) -> int:
    if not ids:
        return 0
    daily_types = tuple(SUMMARY_MEMORY_TYPES - {"seeky_compressed"})
    id_placeholders = ",".join("?" for _ in ids)
    type_placeholders = ",".join("?" for _ in daily_types)
    async with get_db() as db:
        cur = await db.execute(
            f"DELETE FROM memories WHERE id IN ({id_placeholders}) "
            f"AND LOWER(type) IN ({type_placeholders}) AND COALESCE(compression_stage,0)=0",
            (*sorted(ids), *daily_types),
        )
        await db.commit()
        return cur.rowcount if cur.rowcount is not None else 0


async def _delete_chatroom_daily_ids(ids: set[str]) -> int:
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    async with get_db() as db:
        cur = await db.execute(
            f"DELETE FROM chatroom_memories WHERE id IN ({placeholders}) "
            "AND memory_kind='daily' AND COALESCE(compression_stage,0)=0",
            tuple(sorted(ids)),
        )
        await db.commit()
        return cur.rowcount if cur.rowcount is not None else 0


async def _apply_main_daily_draft(payload: dict) -> dict:
    created_daily, created_important, covered = [], [], set()
    for batch in (payload.get("main") or {}).get("batches") or []:
        covered.update(batch.get("covered_ids") or [])
        for item in batch.get("compressed_daily") or []:
            mem_id = await _insert_main_compressed_memory(
                content=str(item.get("content") or "").strip(),
                memory_type="daily",
                keywords=_normalize_daily_keywords(item.get("keywords")),
                importance=max(0.0, min(0.6, float(item.get("importance", 0.25)))),
                source_rows=_source_rows_from_draft_item(item),
                memory_time=float(item.get("memory_time") or time.time()),
                compression_stage=1,
            )
            if mem_id:
                created_daily.append(mem_id)
        for item in batch.get("important_memories") or []:
            importance = float(item.get("importance", 0.0))
            if importance < 0.8:
                continue
            mem_id = await _insert_main_compressed_memory(
                content=str(item.get("content") or "").strip(),
                memory_type=LONG_TERM_MEMORY_TYPE,
                keywords=_normalize_daily_keywords(item.get("keywords")),
                importance=min(1.0, importance),
                source_rows=_source_rows_from_draft_item(item),
                memory_time=float(item.get("memory_time") or time.time()),
                compression_stage=0,
            )
            if mem_id:
                created_important.append(mem_id)
    deleted = await _delete_main_daily_ids(covered)
    if covered or created_daily or created_important:
        messages = [
            batch.get("message", "")
            for batch in (payload.get("main") or {}).get("batches") or []
            if batch.get("message")
        ]
        async with get_db() as db:
            await db.execute(
                "INSERT INTO daily_memory_compress_log (id, actor, old_ids, new_ids, important_ids, message, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    f"dmc_{time.time_ns()}", "aion", json.dumps(sorted(covered), ensure_ascii=False),
                    json.dumps(created_daily, ensure_ascii=False), json.dumps(created_important, ensure_ascii=False),
                    "\n".join(messages), time.time(),
                ),
            )
            await db.commit()
    return {
        "deleted": deleted,
        "covered": len(covered),
        "created_daily": len(created_daily),
        "created_important": len(created_important),
        "new_ids": created_daily,
        "important_ids": created_important,
    }


async def _apply_chatroom_daily_draft(payload: dict) -> dict:
    from chatroom import save_chatroom_memory
    created_daily, created_important, covered = [], [], set()
    for batch in (payload.get("chatroom") or {}).get("batches") or []:
        covered.update(batch.get("covered_ids") or [])
        for item in batch.get("compressed_daily") or []:
            mem_id = await save_chatroom_memory(
                room_id=item.get("room_id") or "connor_unified",
                scope=item.get("scope") or "connor",
                content=str(item.get("content") or "").strip(),
                keywords=",".join(_normalize_daily_keywords(item.get("keywords"))),
                importance=max(0.0, min(0.6, float(item.get("importance", 0.25)))),
                source_start_ts=item.get("source_start_ts"),
                source_end_ts=item.get("source_end_ts"),
                source_msg_id="[]",
                memory_kind="daily",
                compression_stage=1,
                created_at=float(item.get("memory_time") or item.get("source_start_ts") or time.time()),
            )
            if mem_id:
                created_daily.append(mem_id)
                await asyncio.sleep(0.001)
        for item in batch.get("important_memories") or []:
            importance = float(item.get("importance", 0.0))
            if importance < 0.8:
                continue
            mem_id = await save_chatroom_memory(
                room_id=item.get("room_id") or "connor_unified",
                scope=item.get("scope") or "connor",
                content=str(item.get("content") or "").strip(),
                keywords=",".join(_normalize_daily_keywords(item.get("keywords"))),
                importance=min(1.0, importance),
                source_start_ts=item.get("source_start_ts"),
                source_end_ts=item.get("source_end_ts"),
                source_msg_id="[]",
                memory_kind="long_term",
                compression_stage=0,
                created_at=float(item.get("memory_time") or item.get("source_start_ts") or time.time()),
            )
            if mem_id:
                created_important.append(mem_id)
                await asyncio.sleep(0.001)
    deleted = await _delete_chatroom_daily_ids(covered)
    if covered or created_daily or created_important:
        messages = [
            batch.get("message", "")
            for batch in (payload.get("chatroom") or {}).get("batches") or []
            if batch.get("message")
        ]
        async with get_db() as db:
            await db.execute(
                "INSERT INTO daily_memory_compress_log (id, actor, old_ids, new_ids, important_ids, message, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    f"dmc_{time.time_ns()}", "connor", json.dumps(sorted(covered), ensure_ascii=False),
                    json.dumps(created_daily, ensure_ascii=False), json.dumps(created_important, ensure_ascii=False),
                    "\n".join(messages), time.time(),
                ),
            )
            await db.commit()
    return {
        "deleted": deleted,
        "covered": len(covered),
        "created_daily": len(created_daily),
        "created_important": len(created_important),
        "new_ids": created_daily,
        "important_ids": created_important,
    }


async def apply_daily_compression_review(review_id: str) -> dict:
    return {"ok": False, "message": "记忆压缩应用已暂时关闭，防止误删记忆"}  # 暂时关闭
    await _ensure_daily_compression_schema()
    review = await get_daily_compression_review(review_id)
    if not review:
        return {"ok": False, "message": "没有找到这份压缩草稿。"}
    if review["status"] != "draft":
        return {"ok": False, "message": "这份压缩草稿当前不能应用。", "review": review}
    payload = review.get("payload") or {}
    main_result = await _apply_main_daily_draft(payload)
    chatroom_result = await _apply_chatroom_daily_draft(payload)
    apply_result = {"main": main_result, "chatroom": chatroom_result}
    now = time.time()
    async with get_db() as db:
        await db.execute(
            "UPDATE daily_memory_compress_reviews "
            "SET status='applied', apply_result=?, applied_at=?, updated_at=? WHERE id=?",
            (json.dumps(apply_result, ensure_ascii=False), now, now, review_id),
        )
        await db.commit()
    applied = await get_daily_compression_review(review_id)
    total_new_daily = main_result["created_daily"] + chatroom_result["created_daily"]
    total_new_important = main_result["created_important"] + chatroom_result["created_important"]
    total_deleted = main_result["deleted"] + chatroom_result["deleted"]
    return {
        "ok": True,
        "review": applied,
        "message": f"压缩草稿已应用：删除旧日常 {total_deleted} 条，新日常 {total_new_daily} 条，新长期重要 {total_new_important} 条。",
    }


async def discard_daily_compression_review(review_id: str) -> dict:
    await _ensure_daily_compression_schema()
    review = await get_daily_compression_review(review_id)
    if not review:
        return {"ok": False, "message": "没有找到这份压缩草稿。"}
    if review["status"] == "applied":
        return {"ok": False, "message": "已应用的草稿不能废弃。", "review": review}
    now = time.time()
    async with get_db() as db:
        await db.execute(
            "UPDATE daily_memory_compress_reviews "
            "SET status='discarded', discarded_at=?, updated_at=? WHERE id=?",
            (now, now, review_id),
        )
        await db.commit()
    discarded = await get_daily_compression_review(review_id)
    return {"ok": True, "review": discarded, "message": "压缩草稿已废弃。"}


async def compress_expired_daily_memories(days: int = 14) -> dict:
    """Compatibility wrapper: create a draft instead of applying immediately."""
    return {"ok": True, "message": "记忆压缩已暂时关闭", "candidate_count": 0}  # 暂时关闭


async def rebuild_embeddings() -> dict:
    """重建向量索引：用当前配置的 embedding 模型为所有记忆重新生成向量，不触发 AI 总结"""
    success = 0
    failed = 0
    total = 0
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        # 主聊天记忆表
        cur = await db.execute("SELECT id, content FROM memories ORDER BY id")
        rows = await cur.fetchall()
        total += len(rows)
        for row in rows:
            emb = await get_embedding(row["content"][:2000])
            if emb:
                await db.execute(
                    "UPDATE memories SET embedding = ? WHERE id = ?",
                    (_pack_embedding(emb), row["id"])
                )
                success += 1
            else:
                failed += 1
            if success % 5 == 0:
                await db.commit()
                await asyncio.sleep(0.3)
        await db.commit()
        # 聊天室记忆表
        try:
            cur2 = await db.execute("SELECT id, content FROM chatroom_memories ORDER BY id")
            cr_rows = await cur2.fetchall()
            total += len(cr_rows)
            for row in cr_rows:
                emb = await get_embedding(row["content"][:2000])
                if emb:
                    await db.execute(
                        "UPDATE chatroom_memories SET embedding = ? WHERE id = ?",
                        (_pack_embedding(emb), row["id"])
                    )
                    success += 1
                else:
                    failed += 1
                if success % 5 == 0:
                    await db.commit()
                    await asyncio.sleep(0.3)
            await db.commit()
        except Exception:
            pass  # 聊天室记忆表可能不存在
    print(f"[Memory] 向量索引重建完成: {success}/{total} 成功, {failed} 失败")
    return {"total": total, "success": success, "failed": failed}
