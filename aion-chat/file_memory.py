"""
文件记忆系统 — 替代 embedding+recall 的记忆注入方式。
从磁盘读取 md 文件，注入到对话上下文中。
"""

from pathlib import Path
from config import DATA_DIR

PERSONA_DIR = DATA_DIR / "persona"
MEMORY_DIR = DATA_DIR / "memory"

CORE_MEMORY_FILES = [
    "Break.md",
    "心得和反省.md",
    "誓言与信物.md",
    "亲密.md",
    "璃子.md",
    "澄.md",
    "人格快照.md",
    "璃澄语.md",
    "共同备忘录.md",
]


def read_persona_files() -> str:
    if not PERSONA_DIR.exists():
        return ""
    parts = []
    for f in sorted(PERSONA_DIR.glob("*.md")):
        content = f.read_text(encoding="utf-8").strip()
        if content:
            parts.append(content)
    return "\n\n---\n\n".join(parts)


def read_core_memory_files() -> str:
    if not MEMORY_DIR.exists():
        return ""
    parts = []
    for filename in CORE_MEMORY_FILES:
        filepath = MEMORY_DIR / filename
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"### {filename}\n\n{content}")
    return "\n\n---\n\n".join(parts)


def read_latest_diary() -> str:
    diary_dir = MEMORY_DIR / "日记"
    if not diary_dir.exists():
        return ""
    diary_files = sorted(
        [f for f in diary_dir.glob("*.md") if f.name != "TEMPLATE.md"],
        reverse=True,
    )
    if not diary_files:
        return ""
    return diary_files[0].read_text(encoding="utf-8").strip()


def read_memory_file(relative_path: str) -> str:
    filepath = MEMORY_DIR / relative_path
    if not filepath.exists():
        return f"[文件不存在: {relative_path}]"
    try:
        filepath.resolve().relative_to(MEMORY_DIR.resolve())
    except ValueError:
        return "[路径不合法]"
    return filepath.read_text(encoding="utf-8").strip()


def write_memory_file(relative_path: str, content: str) -> str:
    filepath = MEMORY_DIR / relative_path
    try:
        filepath.resolve().relative_to(MEMORY_DIR.resolve())
    except ValueError:
        return "[路径不合法]"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")
    return f"[已写入: {relative_path}]"


def edit_memory_file(relative_path: str, old_text: str, new_text: str) -> str:
    filepath = MEMORY_DIR / relative_path
    if not filepath.exists():
        return f"[文件不存在: {relative_path}]"
    try:
        filepath.resolve().relative_to(MEMORY_DIR.resolve())
    except ValueError:
        return "[路径不合法]"
    content = filepath.read_text(encoding="utf-8")
    if old_text not in content:
        return f"[未找到要替换的文本]"
    content = content.replace(old_text, new_text, 1)
    filepath.write_text(content, encoding="utf-8")
    return f"[已编辑: {relative_path}]"
