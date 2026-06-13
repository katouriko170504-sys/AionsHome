"""
设置、世界书、模型列表、TTS 路由
"""

import json

from fastapi import APIRouter
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

import httpx

from config import SETTINGS, MODELS, save_settings, get_key, get_sentinel_config, load_worldbook, save_worldbook, load_chat_status, TTS_CACHE_DIR, TTS_CACHE_MAX_BYTES, THEATER_TTS_CACHE_DIR, PROVIDERS, load_providers, get_provider, reload_models, _save_models, MODELS_PATH
from tts import cleanup_tts_cache_dir

router = APIRouter()

# ── 模型列表 ──────────────────────────────────────
@router.get("/api/models")
async def list_models():
    return [{"key": k, "provider": v["provider"], "model": v.get("model", ""), "vision": v.get("vision", False)} for k, v in MODELS.items()]

# ── 可用模型（只返回已配 key 的接入方下的模型）──
@router.get("/api/available-models")
async def list_available_models():
    result = []
    for key, cfg in MODELS.items():
        provider_id = cfg["provider"]
        prov = get_provider(provider_id)
        if not prov:
            continue
        if prov["type"] in ("gemini_cli", "antigravity_cli", "codex_cli"):
            result.append({"key": key, "provider": provider_id, "provider_name": prov["name"]})
            continue
        api_key = get_key(provider_id)
        if api_key:
            result.append({"key": key, "provider": provider_id, "provider_name": prov["name"]})
    return result

# ── 接入方管理 ──────────────────────────────────
@router.get("/api/providers")
async def list_providers():
    result = []
    for p in PROVIDERS:
        has_key = False
        if p.get("key_field"):
            has_key = bool(SETTINGS.get(p["key_field"], ""))
        result.append({
            "id": p["id"],
            "name": p["name"],
            "base_url": p.get("base_url", ""),
            "type": p["type"],
            "key_field": p.get("key_field", ""),
            "has_key": has_key
        })
    return result

class ProviderKeyUpdate(BaseModel):
    key: str

@router.put("/api/providers/{provider_id}/key")
async def update_provider_key(provider_id: str, body: ProviderKeyUpdate):
    prov = get_provider(provider_id)
    if not prov:
        return Response(content=json.dumps({"error": f"未知接入方: {provider_id}"}), status_code=404, media_type="application/json")
    key_field = prov.get("key_field", "")
    if not key_field:
        return Response(content=json.dumps({"error": f"接入方 {provider_id} 不需要 API Key"}), status_code=400, media_type="application/json")
    SETTINGS[key_field] = body.key
    save_settings(SETTINGS)
    return {"ok": True}

# ── 模型管理 ──────────────────────────────────────
class ModelCreate(BaseModel):
    key: str
    provider: str
    model: str
    vision: bool = False

@router.post("/api/models")
async def create_model(body: ModelCreate):
    if body.key in MODELS:
        return Response(content=json.dumps({"error": f"模型名 {body.key} 已存在"}), status_code=400, media_type="application/json")
    MODELS[body.key] = {"provider": body.provider, "model": body.model, "vision": body.vision}
    _save_models(MODELS)
    return {"ok": True}

class ModelUpdate(BaseModel):
    provider: str | None = None
    model: str | None = None
    vision: bool | None = None

@router.put("/api/models/{model_key}")
async def update_model(model_key: str, body: ModelUpdate):
    if model_key not in MODELS:
        return Response(content=json.dumps({"error": f"模型 {model_key} 不存在"}), status_code=404, media_type="application/json")
    if body.provider is not None:
        MODELS[model_key]["provider"] = body.provider
    if body.model is not None:
        MODELS[model_key]["model"] = body.model
    if body.vision is not None:
        MODELS[model_key]["vision"] = body.vision
    _save_models(MODELS)
    return {"ok": True}

@router.delete("/api/models/{model_key}")
async def delete_model(model_key: str):
    if model_key not in MODELS:
        return Response(content=json.dumps({"error": f"模型 {model_key} 不存在"}), status_code=404, media_type="application/json")
    del MODELS[model_key]
    _save_models(MODELS)
    return {"ok": True}

# ── 设置 ──────────────────────────────────────────
class SettingsUpdate(BaseModel):
    gemini_key: Optional[str] = None
    siliconflow_key: Optional[str] = None
    gemini_free_key: Optional[str] = None
    aipro_key: Optional[str] = None
    netease_music_u: Optional[str] = None
    sentinel_base_url: Optional[str] = None
    sentinel_api_key: Optional[str] = None
    sentinel_model: Optional[str] = None
    embedding_base_url: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_model: Optional[str] = None
    openrouter_key: Optional[str] = None
    luckin_mcp_enabled: Optional[bool] = None
    luckin_mcp_token: Optional[str] = None
    luckin_default_longitude: Optional[str] = None
    luckin_default_latitude: Optional[str] = None
    luckin_default_shop_keyword: Optional[str] = None

@router.get("/api/settings")
async def get_settings():
    def mask(k):
        if not k or len(k) < 8:
            return k
        return k[:4] + "*" * (len(k) - 8) + k[-4:]
    return {
        "gemini_key": SETTINGS.get("gemini_key", ""),
        "siliconflow_key": SETTINGS.get("siliconflow_key", ""),
        "gemini_free_key": SETTINGS.get("gemini_free_key", ""),
        "aipro_key": SETTINGS.get("aipro_key", ""),
        "netease_music_u": SETTINGS.get("netease_music_u", ""),
        "sentinel_base_url": SETTINGS.get("sentinel_base_url", ""),
        "sentinel_api_key": SETTINGS.get("sentinel_api_key", ""),
        "sentinel_model": SETTINGS.get("sentinel_model", ""),
        "embedding_base_url": SETTINGS.get("embedding_base_url", ""),
        "embedding_api_key": SETTINGS.get("embedding_api_key", ""),
        "embedding_model": SETTINGS.get("embedding_model", ""),
        "luckin_mcp_enabled": SETTINGS.get("luckin_mcp_enabled", False),
        "luckin_mcp_token": SETTINGS.get("luckin_mcp_token", ""),
        "luckin_default_longitude": SETTINGS.get("luckin_default_longitude", ""),
        "luckin_default_latitude": SETTINGS.get("luckin_default_latitude", ""),
        "luckin_default_shop_keyword": SETTINGS.get("luckin_default_shop_keyword", ""),
        "gemini_key_masked": mask(SETTINGS.get("gemini_key", "")),
        "siliconflow_key_masked": mask(SETTINGS.get("siliconflow_key", "")),
        "gemini_free_key_masked": mask(SETTINGS.get("gemini_free_key", "")),
        "aipro_key_masked": mask(SETTINGS.get("aipro_key", "")),
        "netease_music_u_masked": mask(SETTINGS.get("netease_music_u", "")),
        "sentinel_api_key_masked": mask(SETTINGS.get("sentinel_api_key", "")),
        "embedding_api_key_masked": mask(SETTINGS.get("embedding_api_key", "")),
        "openrouter_key": SETTINGS.get("openrouter_key", ""),
        "openrouter_key_masked": mask(SETTINGS.get("openrouter_key", "")),
    }

@router.put("/api/settings")
async def update_settings(body: SettingsUpdate):
    luckin_changed = False
    if body.gemini_key is not None:
        SETTINGS["gemini_key"] = body.gemini_key
    if body.siliconflow_key is not None:
        SETTINGS["siliconflow_key"] = body.siliconflow_key
    if body.gemini_free_key is not None:
        SETTINGS["gemini_free_key"] = body.gemini_free_key
    if body.aipro_key is not None:
        SETTINGS["aipro_key"] = body.aipro_key
    if body.openrouter_key is not None:
        SETTINGS["openrouter_key"] = body.openrouter_key
    if body.sentinel_base_url is not None:
        SETTINGS["sentinel_base_url"] = body.sentinel_base_url
    if body.sentinel_api_key is not None:
        SETTINGS["sentinel_api_key"] = body.sentinel_api_key
    if body.sentinel_model is not None:
        SETTINGS["sentinel_model"] = body.sentinel_model
    if body.embedding_base_url is not None:
        SETTINGS["embedding_base_url"] = body.embedding_base_url
    if body.embedding_api_key is not None:
        SETTINGS["embedding_api_key"] = body.embedding_api_key
    if body.embedding_model is not None:
        SETTINGS["embedding_model"] = body.embedding_model
    if body.luckin_mcp_enabled is not None:
        luckin_changed = luckin_changed or SETTINGS.get("luckin_mcp_enabled") != body.luckin_mcp_enabled
        SETTINGS["luckin_mcp_enabled"] = body.luckin_mcp_enabled
    if body.luckin_mcp_token is not None:
        luckin_changed = luckin_changed or SETTINGS.get("luckin_mcp_token", "") != body.luckin_mcp_token
        SETTINGS["luckin_mcp_token"] = body.luckin_mcp_token
    if body.luckin_default_longitude is not None:
        SETTINGS["luckin_default_longitude"] = body.luckin_default_longitude
    if body.luckin_default_latitude is not None:
        SETTINGS["luckin_default_latitude"] = body.luckin_default_latitude
    if body.luckin_default_shop_keyword is not None:
        SETTINGS["luckin_default_shop_keyword"] = body.luckin_default_shop_keyword
    if body.netease_music_u is not None:
        old_mu = SETTINGS.get("netease_music_u", "")
        SETTINGS["netease_music_u"] = body.netease_music_u
        if body.netease_music_u != old_mu:
            # MUSIC_U 变更，重新登录 pyncm
            try:
                from music import reload_login
                reload_login()
            except Exception:
                pass
    save_settings(SETTINGS)
    if luckin_changed:
        try:
            from luckin import LUCKIN_SERVER_NAME
            from mcp_client import mcp_manager
            await mcp_manager.disconnect(LUCKIN_SERVER_NAME)
        except Exception:
            pass
    return {"ok": True}

# ── 温度设置 ──────────────────────────────────────
class TempUpdate(BaseModel):
    temperature: float

@router.put("/api/settings/temperature")
async def update_temperature(body: TempUpdate):
    SETTINGS["temperature"] = body.temperature
    save_settings(SETTINGS)
    return {"ok": True}

# ── 视频通话开关 ──────────────────────────────────
@router.get("/api/settings/video-call")
async def get_video_call_setting():
    return {"video_call_enabled": SETTINGS.get("video_call_enabled", True)}

class VideoCallToggle(BaseModel):
    enabled: bool

@router.put("/api/settings/video-call")
async def update_video_call_setting(body: VideoCallToggle):
    SETTINGS["video_call_enabled"] = body.enabled
    save_settings(SETTINGS)
    return {"ok": True, "video_call_enabled": body.enabled}

# ── AI 生图开关 ───────────────────────────────────
@router.get("/api/settings/image-gen")
async def get_image_gen_setting():
    return {"image_gen_enabled": SETTINGS.get("image_gen_enabled", False)}

class ImageGenToggle(BaseModel):
    enabled: bool

@router.put("/api/settings/image-gen")
async def update_image_gen_setting(body: ImageGenToggle):
    SETTINGS["image_gen_enabled"] = body.enabled
    save_settings(SETTINGS)
    return {"ok": True, "image_gen_enabled": body.enabled}

# ── CLI 工具调用开关（Gemini CLI / Antigravity CLI） ─────────────────
@router.get("/api/settings/gemini-cli-tools")
async def get_gemini_cli_tools_setting():
    return {"gemini_cli_tools_enabled": SETTINGS.get("gemini_cli_tools_enabled", False)}

class GeminiCliToolsToggle(BaseModel):
    enabled: bool

@router.put("/api/settings/gemini-cli-tools")
async def update_gemini_cli_tools_setting(body: GeminiCliToolsToggle):
    SETTINGS["gemini_cli_tools_enabled"] = body.enabled
    save_settings(SETTINGS)
    return {"ok": True, "gemini_cli_tools_enabled": body.enabled}

# ── 桌宠开关 ──────────────────────────────────────
@router.get("/api/settings/pet")
async def get_pet_setting():
    return {"pet_enabled": SETTINGS.get("pet_enabled", False)}

class PetToggle(BaseModel):
    enabled: bool

@router.put("/api/settings/pet")
async def update_pet_setting(body: PetToggle):
    SETTINGS["pet_enabled"] = body.enabled
    save_settings(SETTINGS)
    return {"ok": True, "pet_enabled": body.enabled}

# ── 健康数据分享开关 ──────────────────────────────
@router.get("/api/settings/health-share")
async def get_health_share_setting():
    return {"health_share_enabled": SETTINGS.get("health_share_enabled", False)}

class HealthShareToggle(BaseModel):
    enabled: bool

@router.put("/api/settings/health-share")
async def update_health_share_setting(body: HealthShareToggle):
    SETTINGS["health_share_enabled"] = body.enabled
    save_settings(SETTINGS)
    return {"ok": True, "health_share_enabled": body.enabled}

# ── 世界书 ────────────────────────────────────────
class WorldBookUpdate(BaseModel):
    ai_persona: str = ""
    user_persona: str = ""
    system_prompt: str = ""
    system_prompt_enabled: bool = True
    ai_name: str = "AI"
    user_name: str = "你"
    persona_schema_version: int = 1
    ai_persona_sections: Dict[str, str] = Field(default_factory=dict)
    user_persona_sections: Dict[str, str] = Field(default_factory=dict)
    creative_rules: str = ""
    persona_section_locks: Dict[str, Any] = Field(default_factory=dict)
    persona_evolution_enabled: bool = False

@router.get("/api/worldbook")
async def get_worldbook():
    return load_worldbook()

@router.put("/api/worldbook")
async def update_worldbook(body: WorldBookUpdate):
    current = load_worldbook()
    payload = body.model_dump() if hasattr(body, "model_dump") else body.dict()
    current.update(payload)
    save_worldbook(current)
    return {"ok": True}

# ── 聊天状态 ──────────────────────────────────────
@router.get("/api/chat_status")
async def get_chat_status_api():
    return load_chat_status()

# ── TTS 语音合成 ──────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    voice: str = ""
    msg_id: Optional[str] = None

@router.post("/api/tts")
async def tts_synthesize(body: TTSRequest):
    key = get_key("siliconflow")
    if not key:
        return Response(content=json.dumps({"error": "未配置硅基流动 API Key"}), status_code=400, media_type="application/json")
    if not body.text.strip():
        return Response(content=json.dumps({"error": "文本不能为空"}), status_code=400, media_type="application/json")
    if not body.voice:
        return Response(content=json.dumps({"error": "未选择语音"}), status_code=400, media_type="application/json")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.siliconflow.cn/v1/audio/speech",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": "FunAudioLLM/CosyVoice2-0.5B",
                    "input": body.text.strip(),
                    "voice": body.voice,
                    "response_format": "mp3",
                    "speed": 1.0,
                    "gain": 0
                }
            )
        if resp.status_code != 200:
            return Response(content=json.dumps({"error": f"TTS API 错误: {resp.status_code}"}), status_code=502, media_type="application/json")
        audio_data = resp.content
        # 如果提供了 msg_id，将音频缓存到服务器
        if body.msg_id:
            import re
            safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '', body.msg_id)
            if safe_id:
                cache_path = TTS_CACHE_DIR / f"{safe_id}.mp3"
                cache_path.write_bytes(audio_data)
                cleanup_tts_cache_dir(TTS_CACHE_DIR, TTS_CACHE_MAX_BYTES, skip={cache_path})
        return Response(content=audio_data, media_type="audio/mpeg")
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}), status_code=500, media_type="application/json")

@router.head("/api/tts/audio/{msg_id}")
@router.get("/api/tts/audio/{msg_id}")
async def tts_audio(msg_id: str):
    import re
    safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '', msg_id)
    if not safe_id:
        return Response(status_code=404)
    cache_path = TTS_CACHE_DIR / f"{safe_id}.mp3"
    if not cache_path.exists():
        return Response(status_code=404)
    return FileResponse(cache_path, media_type="audio/mpeg", filename=f"{safe_id}.mp3")

@router.head("/api/theater/tts/audio/{msg_id}")
@router.get("/api/theater/tts/audio/{msg_id}")
async def theater_tts_audio(msg_id: str):
    import re
    safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '', msg_id)
    if not safe_id:
        return Response(status_code=404)
    cache_path = THEATER_TTS_CACHE_DIR / f"{safe_id}.mp3"
    if not cache_path.exists():
        return Response(status_code=404)
    return FileResponse(cache_path, media_type="audio/mpeg", filename=f"{safe_id}.mp3")

@router.get("/api/tts/voices")
async def tts_voice_list():
    key = get_key("siliconflow")
    if not key:
        return {"voices": [], "error": "未配置硅基流动 API Key"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.siliconflow.cn/v1/audio/voice/list",
                headers={"Authorization": f"Bearer {key}"}
            )
        if resp.status_code != 200:
            return {"voices": [], "error": "获取音色列表失败"}
        data = resp.json()
        voices = data.get("result") or data.get("voices") or data.get("data") or []
        return {"voices": voices}
    except Exception as e:
        return {"voices": [], "error": str(e)}
