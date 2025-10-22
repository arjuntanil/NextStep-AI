"""Fix emoji encoding issues in backend_api.py"""

with open('backend_api.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace emojis with ASCII-safe versions
replacements = {
    '🚀': '[INIT]',
    '✅': '[OK]',
    '⚠️': '[WARN]',
    '❌': '[ERROR]',
    '📦': '[LOAD]',
    'ℹ️': '[INFO]',
    '💬': '[MSG]',
    '📚': '[DATA]',
    '🔄': '[WAIT]',
    '⚡': '[SPEED]',
}

for emoji, replacement in replacements.items():
    content = content.replace(emoji, replacement)

with open('backend_api.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Replaced all emojis with ASCII-safe versions")
