"""Telegram access control (CLAUDE.md §5).

The bot never defaults to open. Admins are always allowed. If the allowlist is
empty, only admins may use the bot; otherwise a chat must be in the allowlist
(or be an admin).
"""


def is_allowed(chat_id: int, allowed: list[int], admins: list[int]) -> bool:
    """Whether ``chat_id`` may use the bot."""
    if chat_id in admins:
        return True
    if not allowed:
        return False  # empty allowlist -> only admins
    return chat_id in allowed
