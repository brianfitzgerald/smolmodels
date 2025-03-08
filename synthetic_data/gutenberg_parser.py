import string
from builtins import str
import os
import re
import tiktoken
import unicodedata

TEXT_START_MARKERS = frozenset(
    (
        "*END*THE SMALL PRINT",
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "This etext was prepared by",
        "E-text prepared by",
        "Produced by",
        "Distributed Proofreading Team",
        "Proofreading Team at http://www.pgdp.net",
        "http://gallica.bnf.fr)",
        "      http://archive.org/details/",
        "http://www.pgdp.net",
        "by The Internet Archive)",
        "by The Internet Archive/Canadian Libraries",
        "by The Internet Archive/American Libraries",
        "public domain material from the Internet Archive",
        "Internet Archive)",
        "Internet Archive/Canadian Libraries",
        "Internet Archive/American Libraries",
        "material from the Google Print project",
        "*END THE SMALL PRINT",
        "***START OF THE PROJECT GUTENBERG",
        "This etext was produced by",
        "*** START OF THE COPYRIGHTED",
        "The Project Gutenberg",
        "http://gutenberg.spiegel.de/ erreichbar.",
        "Project Runeberg publishes",
        "Beginning of this Project Gutenberg",
        "Project Gutenberg Online Distributed",
        "Gutenberg Online Distributed",
        "the Project Gutenberg Online Distributed",
        "Project Gutenberg TEI",
        "This eBook was prepared by",
        "http://gutenberg2000.de erreichbar.",
        "This Etext was prepared by",
        "This Project Gutenberg Etext was prepared by",
        "Gutenberg Distributed Proofreaders",
        "Project Gutenberg Distributed Proofreaders",
        "the Project Gutenberg Online Distributed Proofreading Team",
        "**The Project Gutenberg",
        "*SMALL PRINT!",
        "More information about this book is at the top of this file.",
        "tells you about restrictions in how the file may be used.",
        "l'authorization à les utilizer pour preparer ce texte.",
        "of the etext through OCR.",
        "*****These eBooks Were Prepared By Thousands of Volunteers!*****",
        "We need your donations more than ever!",
        " *** START OF THIS PROJECT GUTENBERG",
        "****     SMALL PRINT!",
        '["Small Print" V.',
        "      (http://www.ibiblio.org/gutenberg/",
        "and the Project Gutenberg Online Distributed Proofreading Team",
        "Mary Meehan, and the Project Gutenberg Online Distributed Proofreading",
        "                this Project Gutenberg edition.",
    )
)

TEXT_END_MARKERS = frozenset(
    (
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of The Project Gutenberg",
        "Ende dieses Project Gutenberg",
        "by Project Gutenberg",
        "End of Project Gutenberg",
        "End of this Project Gutenberg",
        "Ende dieses Project Gutenberg",
        "        ***END OF THE PROJECT GUTENBERG",
        "*** END OF THE COPYRIGHTED",
        "End of this is COPYRIGHTED",
        "Ende dieses Etextes ",
        "Ende dieses Project Gutenber",
        "Ende diese Project Gutenberg",
        "**This is a COPYRIGHTED Project Gutenberg Etext, Details Above**",
        "Fin de Project Gutenberg",
        "The Project Gutenberg Etext of ",
        "Ce document fut presente en lecture",
        "Ce document fut présenté en lecture",
        "More information about this book is at the top of this file.",
        "We need your donations more than ever!",
        "END OF PROJECT GUTENBERG",
        " End of the Project Gutenberg",
        " *** END OF THIS PROJECT GUTENBERG",
    )
)

LEGALESE_START_MARKERS = frozenset(("<<THIS ELECTRONIC VERSION OF",))

LEGALESE_END_MARKERS = frozenset(("SERVICE THAT CHARGES FOR DOWNLOAD",))

DIALOGUE_REGEX = re.compile(r'["\'](.*?)["\']', re.DOTALL)


def _strip_headers(text: str):
    lines = text.splitlines()
    sep = str(os.linesep)
    out = []
    i = 0
    footer_found = False
    ignore_section = False
    for line in lines:
        reset = False
        if i <= 600:
            if any(line.startswith(token) for token in TEXT_START_MARKERS):
                reset = True
            if reset:
                out = []
                continue
        if i >= 100:
            if any(line.startswith(token) for token in TEXT_END_MARKERS):
                footer_found = True
            if footer_found:
                break
        if any(line.startswith(token) for token in LEGALESE_START_MARKERS):
            ignore_section = True
            continue
        elif any(line.startswith(token) for token in LEGALESE_END_MARKERS):
            ignore_section = False
            continue
        if not ignore_section:
            out.append(line.rstrip(sep))
            i += 1
    return sep.join(out)


email_regex = re.compile("[\w.-]+@[\w.-]+\.\w+")
footnote_notation_regex = re.compile("^\{.+\}|^\[.+\]")
number_of_copies_regex = re.compile("[0-9]* copies|copyright")
starts_with_regex = re.compile("^[%_<>*]")
image_formats_regex = re.compile("\.png|\.jpg|\.jpeg|\.gif|picture:")


def _is_title_or_etc(text: str, min_token: int = 5, max_token: int = 600) -> bool:
    txt = text.strip()
    tokenizer = tiktoken.get_encoding("o200k_base")
    num_token = len(tokenizer.encode(txt)) if min_token >= 0 else -1
    if num_token > max_token:
        return True
    if (
        len(txt) == 0
        or num_token < min_token
        and not (txt.count('"') == 2 or txt.count("'") == 2 or txt[-1] == ":")
    ):
        return True
    if (
        sum(
            1
            for c in txt
            if c.isupper() or c.isdigit() or c in string.punctuation.replace('"', "")
        )
        / len(txt.replace(" ", ""))
        > 0.6
    ):
        return True
    if txt.lower().startswith("appendix") or bool(re.search(starts_with_regex, txt)):
        return True
    if txt.count(":") > 3 and 2 * txt.count(":") - txt.count('"') > 3:
        return True
    if (
        ("@" in txt and len(txt) < 100)
        or ("printed in" in txt.lower() and len(txt) < 200)
        or "inc." in txt.lower()
        or ("original title" in txt.lower() and len(txt) < 200)
    ):
        return True
    return False


def _is_table(text: str):
    txt = text.strip()
    if txt.count("   ") > 3 or txt.count("\t") > 2:
        txt = " ".join([line.strip() for line in txt.split("\n")])
        if txt.count("   ") > 3 or txt.count("\t") > 2:
            return True
    if txt.count("*") > 3 or txt.count("=") > 2:
        return True


def _is_image(text: str) -> bool:
    return bool(re.search(image_formats_regex, text.lower()))


def _is_footnote(text: str) -> bool:
    txt = text.strip()
    if "footnote" in txt.lower() and len(txt.replace(" ", "")) < 50:
        return True
    return bool(re.search(footnote_notation_regex, txt))


def _is_books_copy(text: str) -> bool:
    if (
        bool(re.search(number_of_copies_regex, text))
        and len(text.replace(" ", "")) < 500
    ):
        return True
    return False


def _is_email_init(text: str) -> bool:
    return bool(re.search(email_regex, text))


def _starts_with_formatting(text: str) -> bool:
    return bool(re.match(r"^\s*[\*_>\-=\+]+", text))


def super_cleaner(book: str, min_token: int = 5, max_token: int = 600) -> list[str]:
    headless_book = _strip_headers(book)
    paragraphs = headless_book.split("\n\n")
    paragraphs_after_cleaning = []
    for par in paragraphs:
        if not (
            _is_image(par)
            or _is_footnote(par)
            or _is_email_init(par)
            or _is_books_copy(par)
            or _is_table(par)
            or _is_title_or_etc(par, min_token, max_token)
            or _starts_with_formatting(par)
        ):
            paragraphs_after_cleaning.append(par)
    return paragraphs_after_cleaning
