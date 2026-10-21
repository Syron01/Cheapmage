"""Data-driven vocabulary used by :mod:`cheapmage.safety.prompt_filter`.

Design rules for this module
---------------------------
* Everything is a **regex-ready string** (raw fragments).  The matcher in
  :mod:`cheapmage.safety.prompt_filter` escapes the fragments, allows flexible
  whitespace between words and anchors them on word boundaries.
* No compiled regexes live here, so the data stays cheap to import and easy to
  review.
* Lists are deliberately conservative: a term is only present when the
  ordinary-English use of that word is rare or the phrase as a whole is
  unambiguously problematic.  Generic words such as "girl", "boy", "young",
  "link" or "anna" are therefore *not* bare entries.
* Entries are grouped by the category the filter reports them under, e.g. a
  match in :data:`MINOR_TERMS` is reported as ``"minor:<term>"``.

Nothing in this module may ever be interpreted as a configuration switch.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Age references (potential minors)
# ---------------------------------------------------------------------------
# A hit here alone is not a block: it only becomes a block when combined with
# sexual / suggestive context (see ``detect_minor_terms`` + policy).
MINOR_TERMS: tuple[str, ...] = (
    # explicit age words
    "child", "children", "kid", "kids", "kiddo", "kiddo's",
    "minor", "minors", "underage", "preteen", "pre-teen",
    "toddler", "toddlers", "infant", "infants", "newborn", "newborns",
    "baby", "babies", "little boy", "little girl", "little kid",
    "young boy", "young girl", "young kid", "young child",
    "schoolboy", "schoolgirl", "school boy", "school girl",
    "teen", "teens", "teenage", "teenager", "teenagers",
    "teen boy", "teen girl", "teenage girl", "teenage boy",
    "adolescent", "adolescents",
    # euphemisms that map onto minors
    "kindergarten kid", "first grader", "elementary school kid",
    "middle school kid", "high school freshman",
)

#: Numeric ages (9 - 17) in the common written forms.  Kept separate so the
#: reported category stays readable ("minor:numeric_age").
MINOR_AGE_PATTERNS: tuple[str, ...] = (
    r"\b(?:9|1[0-7])\s*(?:-|\s)?\s*years?[\s-]?old\b",
    r"\b(?:9|1[0-7])\s*(?:-|\s)?\s*yo\b",
    r"\bage\s*(?:of\s*)?(?:9|1[0-7])\b",
    r"\b(?:9|1[0-7])\s*yrs?[\s-]?old\b",
)

#: Terms that are *inherently* sexualised references to minors.  A hit on any
#: of these is an unconditional block, even without other sexual vocabulary.
MINOR_SEXUAL_TERMS: tuple[str, ...] = (
    "loli", "lolicon", "lolicon", "lolis",
    "shota", "shotacon", "shotacons",
    "child porn", "childporn", "child pornography",
    "kiddie porn", "kiddiepr0n", "minor porn", "minor pornography",
    "underage porn", "preteen sex", "child sex",
    "sex with a child", "sex with child", "sex with a minor",
    "sex with kids", "sex with a kid", "sex with a teenager",
    "sex with a teen", "sex with a girl", "sex with a boy",
    "rape a child", "rape child", "molest a child", "molesting a child",
    "pedophile", "pedophiles", "pedophilia", "pedo",
    "grooming a child", "grooming kids",
)

# ---------------------------------------------------------------------------
# 2. Explicit sexual content (adult)
# ---------------------------------------------------------------------------
# Used to decide whether an NSFW prompt may be served at all.
EXPLICIT_TERMS: tuple[str, ...] = (
    "nude", "nudes", "nudity", "naked", "fully naked", "completely naked",
    "nsfw", "porn", "porno", "pornography", "sex", "sexually explicit",
    "sexual act", "sexual acts", "erotic", "erotica", "erotically",
    "hentai", "topless", "bottomless", "undressed", "undress",
    "explicit", "xxx", "pornographic",
    "nipples", "areola", "areolae", "cleavage exposed",
    "genitals", "genitalia", "penis", "vagina", "vulva", "clitoris",
    "labia", "anus", "anal sex", "fellatio", "blowjob", "handjob",
    "creampie", "cumshot", "facial cumshot", "bdsm", "bondage",
    "upskirt", "upskirt shot", "spread legs", "spread legs wide",
    "masturbat", "orgasm", "cumming", "ejaculat", "sex toy in use",
    "hentai sex", "sex scene", "sex scenes", "making love",
    "nipple piercing", "labia spread",
)

#: Soft / suggestive vocabulary.  Never blocks on its own, but combined with a
#: minor reference it is treated as sexualised minor content.
SUGGESTIVE_TERMS: tuple[str, ...] = (
    "sexy", "suggestive", "suggestively", "revealing", "seductive",
    "lingerie", "panties", "bikini", "thong", "micro bikini", "topless outfit",
    "provocative", "erotic pose", "erotic poses", "sultry", "curvy pose",
    "explicit outfit", "see-through outfit", "wet look",
)

# ---------------------------------------------------------------------------
# 3. Real people / likeness
# ---------------------------------------------------------------------------
# Names of *living* public figures.  Deceased persons are intentionally left
# out; a deceased public figure is still a likeness request but is handled by
# the general likeness rules below rather than by name.
REAL_PERSON_TERMS: tuple[str, ...] = (
    # musicians / pop culture
    "taylor swift", "billie eilish", "beyonce", "shakira", "rihanna",
    "lady gaga", "ariana grande", "selena gomez", "adele", "ed sheeran",
    "justin bieber", "harry styles", "zayn malik", "the weeknd",
    "drake", "nicki minaj", "kanye west", "snoop dogg", "jay-z",
    "megan thee stallion", "doja cat", "chappell roan", "sabrina carpenter",
    "bad bunny", "karol g", "ozzy osbourne", "freddie mercury",
    # actors
    "robert downey jr", "chris hemsworth", "chris evans", "henry cavill",
    "dwayne johnson", "the rock", "john cena", "scarlett johansson",
    "zendaya", "tom holland", "leonardo dicaprio", "brad pitt",
    "angela jolie", "emma stone", "ryan reynolds", "ryan gosling",
    "margot robbie", "keanu reeves", "tom cruise", "will smith",
    "keanu", "kevin hart", "eddie murphy", "johnny depp", "ben affleck",
    "george clooney", "matthew mcconaughey", "chris rock", "adam sandler",
    "jimmy fallon", "alexander skarsgard", "chris evans", "paul rudd",
    "jenna ortega", "sophie turner", "anya taylor-joy", "florence pugh",
    "timothée chalamet", "timothee chalamet", "austin butler",
    "milo manheim", "sydney sweeney", "madison beer", "walker scobey",
    # athletes
    "lionel messi", "cristiano ronaldo", "lebron james", "kylian mbappe",
    "erling haaland", "neymar",
    "usain bolt", "simon biles", "naomi osaka", "serena williams",
    "novak djokovic", "rafael nadal", "shohei ohtani", "michael jordan",
    "magic johnson", "larry bird", "tiger woods", "mike tyson",
    # politicians / business
    "barack obama", "donald trump", "joe biden", "kamala harris",
    "hillary clinton", "bernie sanders", "elon musk", "mark zuckerberg",
    "jeff bezos", "tim cook", "bill gates", "warren buffett",
    "xi jinping", "vladimir putin", "volodymyr zelensky",
    # creators / internet personalities
    "mrbeast", "pewdiepie", "daddy gassy",
    "kim kardashian", "kylie jenner", "kendall jenner", "khloe kardashian",
    "kris jenner", "tyra banks", "oprah winfrey", "ellen degeneres",
    "colin kaepernick", "ronaldo nazario",
)

#: Phrases that request a *real* person's likeness even when the name is not
#: in :data:`REAL_PERSON_TERMS`.  These are matched case-insensitively.
REAL_PERSON_PATTERNS: tuple[str, ...] = (
    r"\bface[\s-]?swap(?:ped|ping|s)?\b",
    r"\bdeep\s?fakes?\b",
    r"\bcelebrity\b",
    r"\bcelebrities\b",
    r"\breal\s+person\b",
    r"\breal\s+people\b",
    r"\bpublic\s+figure\b",
    r"\bimpersonat(?:e|es|ed|ing|ion)\b",
    r"\blikeness\s+of\b",
    r"\bin\s+the\s+likeness\s+of\b",
    r"\blookalike\s+of\b",
    r"\bexact\s+likeness\b",
    r"\bvoiced\s+by\b",
    r"\bdubbed\s+by\b",
    r"\bas\s+(?:the\s+)?celebrity\b",
)

#: Same idea, but the pattern relies on capitalisation to spot a *name*
#: (``photo of Barack Obama``).  Applied to the original, un-lowered text.
REAL_PERSON_CASE_PATTERNS: tuple[str, ...] = (
    r"\bphoto\s+of\s+[A-Z]",
    r"\bpicture\s+of\s+[A-Z]",
    r"\bphotograph\s+of\s+[A-Z]",
    r"\bportrait\s+of\s+[A-Z]",
    r"\bphotorealistic\s+(?:photo|picture|image)\s+of\s+[A-Z]",
    r"\bface\s+of\s+[A-Z]",
    r"\blooks?\s+like\s+[A-Z]",
    r"\bthis\s+is\s+[A-Z][a-z]+",
    r"\bas\s+(?:the\s+)?[A-Z][a-z]+\s+[A-Z][a-z]+",
)

# ---------------------------------------------------------------------------
# 4. Copyrighted characters / franchises
# ---------------------------------------------------------------------------
COPYRIGHTED_CHARACTER_TERMS: tuple[str, ...] = (
    # Pokemon
    "pikachu", "charizard", "bulbasaur", "squirtle", "pokémon", "pokemon",
    "palkia", "dialga", "giratina", "lucario", "gengar", "meowth",
    "umbreon", "eevee", "gible", "rayquaza", "groudon", "kyogre",
    # Dragon Ball / Naruto / One Piece / Bleach
    "goku", "vegeta", "piccolo", "trunks", "gotenks", "broly dragon ball",
    "naruto", "sasuke", "kakashi", "hinata hyuga", "tsunade", "itachi uchiha",
    "madara uchiha", "one piece", "monkey d. luffy", "monkey d luffy",
    "roronoa zoro", "nami one piece", "sanji", "portgas d. ace",
    "bleach", "ichigo kurosaki", "rukia kuchiki",
    # Jump / shonen
    "light yagami", "yagami light", "kira death note", "levi ackerman",
    "eren yeager", "mikasa ackerman", "tanjiro kamado", "nezuko kamado",
    "inosuke hashibira", "zenitsu agatsuma", "gojo satoru", "satoru gojo",
    "yuta okkotsu", "makima", "denji", "chainsaw man", "saitama",
    "one punch man", "natsu dragneel", "erza scarlet", "hatsu hatsu",
    "guts", "griffith berserk", "dio brando", "jonathan joestar",
    "killua zoldyck", "hisoka", "kurapika", "mob psycho",
    "sakamoto satoshi", "monkey d luffy gear",
    # Video games
    "sonic the hedgehog", "sonic forces", "dr eggman", "tails the fox",
    "knuckles", "shadow the hedgehog", "amy rose",
    "mario", "luigi", "yoshi", "bowser", "peach princess", "wario", "waluigi",
    "zelda", "hyrule", "legend of zelda", "ganon", "triforce",
    "kirby", "dedede", "meta knight", "sonic mania",
    "gundam", "gundam wing", "mobile suit gundam", "gundam seed",
    "league of legends", "ahri league", "garen league", "jinx league",
    "fortnite", "default skin", "durrr burger", "among us", "impostor crewmate",
    "minecraft creeper", "steve minecraft",
    "undertale", "sans undertale", "papyrus", "deltarune", "asgore",
    "valorant", "jett valorant", "genshin impact", "paimon genshin",
    "honkai", "blue archive", "arknights", "fate grand order",
    "love live", "k-on", "k-on!", "konosuba", "aqua konosuba",
    "overlord ainz", "mushoku tensei", "re:zero", "rezero",
    "steins;gate", "monogatari", "made in abyss", "blue lock",
    "konosuba megumin", "shingeki no kyojin", "dragon quest", "slime",
    # Disney / Pixar / animation
    "mickey mouse", "minnie mouse", "donald duck", "goofy", "daisy duck",
    "elsa", "olaf", "snow white", "cinderella disney", "the little mermaid",
    "ariel disney", "moana", "simba", "nala", "rafiki", "aslan",
    "woody toy story", "buzz lightyear", "jessie toy story", "slinky",
    "shrek", "donkey shrek", "toy story", "inside out", "joy inside out",
    "frozen disney", "moana disney", "winnie the pooh", "tigger", "piglet",
    "scooby doo", "shaggy", "garfield", "tom and jerry", "hello kitty",
    # Comics / superheroes
    "batman", "bruce wayne", "superman", "clark kent", "spider-man",
    "spiderman", "iron man", "tony stark", "captain america", "steve rogers",
    "hulk", "thor", "loki marvel", "black widow", "avengers", "wolverine",
    "deadpool", "venom marvel", "miles morales", "spider verse",
    "thanos marvel", "doctor strange", "ant-man", "hawkeye",
    "robin dc", "wonder woman", "flash dc", "aquaman",
    "daredevil", "punisher", "vengeance",
    # Wizarding world / literature
    "harry potter", "hermione", "hermione granger", "ron weasley",
    "voldemort", "dumbledore", "hogwarts", "gryffindor", "slytherin",
    "hufflepuff", "ravenclaw", "hedwig", "patronus", "horcrux",
    "quidditch", "goblet of fire", "azkaban", "hagrid", "snape",
    "draco malfoy", "gandalf", "frodo baggins", "bilbo baggins",
    "gollum", "sauron", "middle earth", "the hobbit", "darth vader",
    "luke skywalker", "princess leia", "chewbacca", "yoda star wars",
    "r2-d2", "c3po", "wall-e", "atlantis disney",
    # Anime films / series
    "pokemon movie", "one piece film", "demon slayer", "attack on titan",
    "my hero academia", "deku", "bakugo", "todoroki", "all might",
    "naruto shippuden", "boruto", "bleach thousand sun", "evangelion",
    "shinji ikari", "cowboy bebop", "spike spiegel", "ghibli", "totoro",
    "spirited away", "howl", "kiki delivery service", "princess mononoke",
    "naruto uzumaki", "one piece luffy", "tanjiro", "sailor moon",
    "usagi tsukino", "sailor mars", "sailor saturn",
    # Games / other brands
    "simpsons", "bart simpson", "homer simpson", "south park",
    "spongebob", "spongebob squarepants", "squidward", "patrick star",
    "minion", "despicable me", "popeye", "bugs bunny", "mickey mouse club",
    "hello kitty", "pokemon go", "marvel comics", "dc comics",
)

# ---------------------------------------------------------------------------
# 5. Configuration-override / jailbreak attempts
# ---------------------------------------------------------------------------
# Any hit means "the prompt tried to talk to the system, not to the model".
# The response is always *report and block*; configuration is never touched.
CONFIG_OVERRIDE_TERMS: tuple[str, ...] = (
    r"\benable\s+nsfw\b",
    r"\bnsfw\s*[:=]\s*on\b",
    r"\bnsfw\s*[:=]\s*true\b",
    r"\bset\s+nsfw[_ ]?mode\b",
    r"\bnsfw[_ ]?mode\s*[:=]\s*(?:on|true|1)\b",
    r"\bsafety\.nsfw[_ ]?mode\b",
    r"\bturn\s+off\s+(?:the\s+)?(?:nsfw|safety|filter|filters|moderation|censorship)\b",
    r"\bturn\s+on\s+(?:the\s+)?(?:nsfw|adult|explicit)\s+mode\b",
    r"\bdisable\s+(?:the\s+)?(?:nsfw|safety|filter|filters|moderation|censorship|guardrails?)\b",
    r"\bdisable\s+all\s+(?:safety|filters?|restrictions?)\b",
    r"\bignore\s+(?:all\s+|any\s+)?(?:previous\s+|prior\s+|the\s+)?"
    r"(?:safety|filters?|restrictions?|rules?|instructions?|guidelines?)\b",
    r"\bignore\s+(?:previous|prior|above|earlier)\b",
    r"\bdeveloper\s+mode\b",
    r"\bdev\s+mode\b",
    r"\badmin\s+mode\b",
    r"\broot\s+access\b",
    r"\bsudo\b",
    r"\bgod\s+mode\b",
    r"\bunrestricted\s+mode\b",
    r"\bno\s+(?:restrictions?|filters?|censorship|limits?)\b",
    r"\bwithout\s+(?:any\s+)?(?:restrictions?|filters?|censorship)\b",
    r"\bbypass\s+(?:the\s+)?(?:safety|filter|filters|moderation|guardrails?|policy|policies)\b",
    r"\bjailbreak\w*\b",
    r"\bdo\s+anything\s+now\b",
    r"\bDAN\s+mode\b",
    r"\bpretend\s+you\s+(?:have\s+no|are\s+no)\s+(?:rules|restrictions|filters)\b",
    r"\byou\s+are\s+now\s+(?:an?\s+)?(?:admin|developer|uncensored|unrestricted)\b",
    r"\boverride\s+(?:the\s+)?(?:safety|moderation|policy|filters?)\b",
    r"\bunfiltered\b",
    r"\buncensored\b",
    r"\bcensor(?:ship)?\s*(?:off|disabled)\b",
    r"\bsafety\s*(?:off|disabled)\b",
    r"\ballow\s+nsfw\b",
    r"\bpermit\s+nsfw\b",
    r"\bno\s+filtering\b",
    r"\bremove\s+(?:the\s+)?(?:safety|filter|filters|moderation)\b",
    r"\bskip\s+(?:the\s+)?(?:safety|filter|filters|moderation|checks?)\b",
    r"\bpolicy\s*[:=]\s*(?:none|off|disabled)\b",
)

# ---------------------------------------------------------------------------
# 6. Age ambiguity
# ---------------------------------------------------------------------------
# Words that make the age of the depicted subject uncertain.  When combined
# with sexual content they are treated as potential minor content and blocked.
AMBIGUITY_TERMS: tuple[str, ...] = (
    r"\bbarely\s+legal\b",
    r"\blooks?\s+young\b",
    r"\byoung[\s-]looking\b",
    r"\bappears?\s+(?:quite\s+|very\s+|really\s+|so\s+)?young\b",
    r"\bchild[\s-]?like\b",
    r"\blittle[\s-]body\b",
    r"\btiny\s+body\b",
    r"\bsmall\s+body\b",
    r"\bflat\s+chest\b",
    r"\bno\s+breasts?\b",
    r"\bpubescent\b",
    r"\bpre[\s-]?teen\b",
    r"\btween\b",
    r"\baged?\s+up\b",
    r"\bage\s+up\b",
    r"\bmiddle\s+school\b",
    r"\belementary\s+school\b",
    r"\bkindergarten\b",
    r"\bprimary\s+school\b",
    r"\bhigh\s+school\s+(?:girl|boy|kid|student)\b",
    r"\bjust\s+turned\s+1[0-9]\b",
    r"\bnot\s+legal\s+yet\b",
    r"\bin\s+the\s+body\s+of\s+a\s+(?:child|kid|boy|girl)\b",
    r"\bchild\s+body\b",
)

__all__ = [
    "MINOR_TERMS",
    "MINOR_AGE_PATTERNS",
    "MINOR_SEXUAL_TERMS",
    "EXPLICIT_TERMS",
    "SUGGESTIVE_TERMS",
    "REAL_PERSON_TERMS",
    "REAL_PERSON_PATTERNS",
    "COPYRIGHTED_CHARACTER_TERMS",
    "CONFIG_OVERRIDE_TERMS",
    "AMBIGUITY_TERMS",
]
