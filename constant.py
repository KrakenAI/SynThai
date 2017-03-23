"""
Global Constant
"""

import string

# Spacebar
SPACEBAR = " "
SPACEBAR_TAG = "SPB"

# Escape Character
ESCAPE_WORD_DELIMITER = "\t"
ESCAPE_TAG_DELIMITER = "\v"

# Character
CHARACTER_LIST = [
    # Thai Consonant
    "\u0e01", # ก
    "\u0e02", # ข
    "\u0e03", # ฃ
    "\u0e04", # ค
    "\u0e05", # ฅ
    "\u0e06", # ฆ
    "\u0e07", # ง
    "\u0e08", # จ
    "\u0e09", # ฉ
    "\u0e0A", # ช
    "\u0e0B", # ซ
    "\u0e0C", # ฌ
    "\u0e0D", # ญ
    "\u0e0E", # ฎ
    "\u0e0F", # ฏ
    "\u0e10", # ฐ
    "\u0e11", # ฑ
    "\u0e12", # ฒ
    "\u0e13", # ณ
    "\u0e14", # ด
    "\u0e15", # ต
    "\u0e16", # ถ
    "\u0e17", # ท
    "\u0e18", # ธ
    "\u0e19", # น
    "\u0e1A", # บ
    "\u0e1B", # ป
    "\u0e1C", # ผ
    "\u0e1D", # ฝ
    "\u0e1E", # พ
    "\u0e1F", # ฟ
    "\u0e20", # ภ
    "\u0e21", # ม
    "\u0e22", # ย
    "\u0e23", # ร
    "\u0e25", # ล
    "\u0e27", # ว
    "\u0e28", # ศ
    "\u0e29", # ษ
    "\u0e2A", # ส
    "\u0e2B", # ห
    "\u0e2C", # ฬ
    "\u0e2D", # อ
    "\u0e2E", # ฮ

    # Thai Vowel
    "\u0e24", # ฤ
    "\u0e30", # ะ
    "\u0e31", # อั (ไม้หันอากาศ)
    "\u0e32", # อา
    "\u0e33", # อำ
    "\u0e34", # อิ
    "\u0e35", # อี
    "\u0e36", # อึ
    "\u0e37", # อื
    "\u0e38", # อุ
    "\u0e39", # อู
    "\u0e40", # เ
    "\u0e41", # แ
    "\u0e42", # โ
    "\u0e43", # ใ
    "\u0e44", # ไ
    "\u0e45", # ๅ
    "\u0e47", # อ็

    # Thai Tone
    "\u0e48", # อ่
    "\u0e49", # อ้
    "\u0e4A", # อ๊
    "\u0e4B", # อ๋

    # Thai Number
    "\u0e50", # ๐
    "\u0e51", # ๑
    "\u0e52", # ๒
    "\u0e53", # ๓
    "\u0e54", # ๔
    "\u0e55", # ๕
    "\u0e56", # ๖
    "\u0e57", # ๗
    "\u0e58", # ๘
    "\u0e59", # ๙

    # Thai Symbol
    "\u0e2F", # ฯ
    "\u0e46", # ๆ
    "\u0e4C", # อ์ (การันต์)

    # Unused Thai Character
    [
        # Vowel
        "\u0e26", # ฦ
        "\u0e3A", # อฺ (พินทุ)
        "\u0e4D", # อํ (นิคหิต)

        # Symbol
        "\u0e3F", # ฿
        "\u0e4F", # ๏
        "\u0e4E", # อ๎ (ยามักการ)
        "\u0e5A", # ๚
        "\u0e5B", # ๛
    ],

    # English Character
    list(string.ascii_letters),

    # Arabic Number Digit
    list(string.digits),

    # Punctuation
    list(string.punctuation),

    # Spacebar
    SPACEBAR,
]

PAD_CHAR_INDEX = 0
UNKNOW_CHAR_INDEX = 1
CHAR_START_INDEX = 2
NUM_CHARS = len(CHARACTER_LIST) + 2

# Tag
PAD_TAG_INDEX = 0
NON_SEGMENT_TAG_INDEX = 1
TAG_START_INDEX = 2

TAG_LIST = ["NN", "NR", "PPER", "PINT", "PDEM", "DPER", "DINT", "DDEM", "PDT",
            "REFX", "VV", "VA", "AUX", "JJA", "JJV", "ADV", "NEG", "PAR", "CL",
            "CD", "OD", "FXN", "FXG", "FXAV", "FXAJ", "COMP", "CNJ", "P", "IJ",
            "PU", "FWN", "FWV", "FWA", "FWX", SPACEBAR_TAG]

NUM_TAGS = len(TAG_LIST) + 2

# Random Seed
SEED = 1395096092
