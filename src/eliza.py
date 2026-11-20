"""Area B - ELIZA-style regex rules for greetings and small talk.

Rules are tried in order; the first matching pattern produces a reply. Captured
groups are reflected ("my X" -> "your X") like Weizenbaum's ELIZA.
"""
import random
import re

REFLECTIONS = {"i": "you", "me": "you", "my": "your", "am": "are", "i'm": "you're",
               "mine": "yours", "myself": "yourself", "you": "I", "your": "my"}

RULES = [
    (r"^\s*(hi|hello|hey|merhaba|selam|good (morning|afternoon|evening)|gunaydin)\b.*",
     ["Hello! Please describe your problem and I will route it to the right team.",
      "Merhaba! How can we help you today?"]),
    (r".*\b(thank you|thanks|thx|tesekkur(ler)?|sagol)\b.*",
     ["You're welcome! Is there anything else I can help with?"]),
    (r"^\s*(bye|goodbye|see you|gorusuruz|hoscakal)\b.*",
     ["Goodbye! Have a nice day."]),
    (r".*\bare you (a )?(bot|robot|human|real)\b.*",
     ["I am an automatic routing assistant. A human agent takes over when I am not sure."]),
    (r".*\bhow are you\b.*|.*\bnasilsin\b.*",
     ["I'm fine, thanks for asking. What can I do for you?"]),
    (r".*\bi feel (.*)",
     ["Why do you feel {0}? Tell me about the problem with your appliance."]),
    (r".*\bi am (angry|upset|frustrated|annoyed)\b.*",
     ["I'm sorry you are {0}. Let me route your message to the right team quickly."]),
    (r".*\b(joke|funny)\b.*",
     ["Why did the fridge go to therapy? It had too many cold feelings."]),
    (r".*\b(your name|who are you)\b.*",
     ["I'm the support router of this appliance brand."]),
]
_COMPILED = [(re.compile(p, re.I), r) for p, r in RULES]
GENERIC = ["Thanks for chatting! Please tell me what problem you have with your product."]


def reflect(fragment: str) -> str:
    return " ".join(REFLECTIONS.get(w, w) for w in fragment.lower().split())


def eliza_reply(text: str, force: bool = False, rng=random.Random(0)):
    """Return a small-talk reply, or None if no rule matches (unless force=True)."""
    t = text.strip().rstrip("!.?")
    for pat, replies in _COMPILED:
        m = pat.match(t)
        if m:
            groups = [reflect(g) for g in m.groups() if g]
            reply = rng.choice(replies)
            return reply.format(*groups) if "{0}" in reply and groups else reply
    return rng.choice(GENERIC) if force else None
