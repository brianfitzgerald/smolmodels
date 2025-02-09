import string
from builtins import str
import os
import re
import tiktoken

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
        "Ende dieses Projekt Gutenberg",
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
    """Remove lines that are part of the Project Gutenberg header or footer.
    Note: The original version of the code can be found at:
    https://github.com/c-w/gutenberg/blob/master/gutenberg/cleanup/strip_headers.py
    Args:
        text (unicode): The body of the text to clean up.
    Returns:
        unicode: The text with any non-text content removed.
    """
    lines = text.splitlines()
    sep = str(os.linesep)

    out = []
    i = 0
    footer_found = False
    ignore_section = False

    for line in lines:
        reset = False

        if i <= 600:
            # Check if the header ends here
            if any(line.startswith(token) for token in TEXT_START_MARKERS):
                reset = True

            # If it's the end of the header, delete the output produced so far.
            # May be done several times, if multiple lines occur indicating the
            # end of the header
            if reset:
                out = []
                continue

        if i >= 100:
            # Check if the footer begins here
            if any(line.startswith(token) for token in TEXT_END_MARKERS):
                footer_found = True

            # If it's the beginning of the footer, stop output
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


email_regex = re.compile("[\w.-]+@[\w.-]+\.\w+")  # type: ignore # Regex to find Emails.
footnote_notation_regex = re.compile(
    "^\{.+\}|^\[.+\]"  # type: ignore
)  # Regex to find start of footnotes.
number_of_copies_regex = re.compile(
    "[0-9]* copies|copyright"
)  # Regex to find copy mentioning.
starts_with_regex = re.compile(
    "^[%_<>*]"
)  # If the text is started with these, it is not a good one.
image_formats_regex = re.compile(
    "\.png|\.jpg|\.jpeg|\.gif|picture:"  # type: ignore
)  # Regex to find images.


def _is_title_or_etc(text: str, min_token: int = 5, max_token: int = 600) -> bool:
    """
    determining if a paragraph is title or information of the book.
    IMPORTANT: if you don't want the text to be tokenize, just put min_token = -1.
    :rtype: bool
    :param text: Raw paragraph.
    :param min_token: The minimum tokens of a paragraph that is not "dialog" or "quote",
     -1 means don't tokenize the txt (so it will be faster).
    :param max_token: The maximum tokens of a paragraph.
    :return: Boolean, True if it is title or information of the book or a bad paragraph.
    """
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
        return True  # Length is short but not "dialog" or "quote"
    if (
        sum(
            1
            for c in txt
            if c.isupper() or c.isdigit() or c in string.punctuation.replace('"', "")
        )
        / len(txt.replace(" ", ""))
        > 0.6
    ):
        return True  # More than 60% of chars are UPPER or digits or punctuations so it might be title or etc.
    if txt.lower().startswith("appendix") or bool(re.search(starts_with_regex, txt)):
        return True
    if txt.count(":") > 3 and 2 * txt.count(":") - txt.count('"') > 3:
        return True  # mostly information about the book.
    if (
        ("@" in txt and len(txt) < 100)
        or ("printed in" in txt.lower() and len(txt) < 200)
        or "inc." in txt.lower()
        or ("original title" in txt.lower() and len(txt) < 200)
    ):
        return True
    return False


def _is_table(text: str):
    """
    determining if a paragraph is a table or catalog.
    :rtype: bool
    :param text: Raw paragraph.
    :return:  Boolean, True if it is a table or catalog.
    """
    txt = text.strip()
    if txt.count("   ") > 3 or txt.count("\t") > 2:
        txt = " ".join([line.strip() for line in txt.split("\n")])
        if txt.count("   ") > 3 or txt.count("\t") > 2:
            return True  # mostly tables.
    if txt.count("*") > 3 or txt.count("=") > 2:
        return True  # mostly catalogs and etc.


def _is_image(text: str) -> bool:
    """
    determining if a paragraph is for mentioning an image.
    :param text: Raw paragraph.
    :return: Boolean, True if it is for mentioning an image.
    """
    return bool(re.search(image_formats_regex, text.lower()))


def _is_footnote(text: str) -> bool:
    """
    determining if a paragraph is the footnote of the book.
    :rtype: bool
    :param text: Raw paragraph.
    :return: Boolean, True if it is the footnote of the book.
    """
    txt = text.strip()
    if "footnote" in txt.lower() and len(txt.replace(" ", "")) < 50:
        return True
    return bool(
        re.search(footnote_notation_regex, txt)
    )  # if a line starts with {...} it might be a footnote.


def _is_books_copy(text: str) -> bool:
    """
    determining if a paragraph indicates the number of copies of this book.
    :rtype: bool
    :param text: text: Raw paragraph.
    :return: Boolean, True if it is indicating the copy of book or copyrights.
    """
    if (
        bool(re.search(number_of_copies_regex, text))
        and len(text.replace(" ", "")) < 500
    ):
        return True
    return False


def _is_email_init(text: str) -> bool:
    """
    determining if a paragraph includes an Email.
    :rtype: bool
    :param text: Raw paragraph.
    :return: Boolean, True if it includes an Email.
    """
    return bool(re.search(email_regex, text))


def super_cleaner(book: str, min_token: int = 5, max_token: int = 600) -> list[str]:
    """
    Super clean the book (titles, footnotes, images, book information, etc.). may delete some good lines too.
    ^_^ Do you have a comment to make it better? make an issue here: https://github.com/kiasar/gutenberg_cleaner ^_^.
    IMPORTANT: if you don't want the text to be tokenize, just put min_token = -1.
    :rtype: str
    :param book: str of a gutenberg's book.
    :param min_token: The minimum tokens of a paragraph that is not "dialog" or "quote",
     -1 means don't tokenize the txt (so it will be faster).
    :param max_token: The maximum tokens of a paragraph.
    :return: str of the book with paragraphs that have been deleted are shown with "[deleted]" in it.
    you can split the book to paragraphs by "\n\n".
    """
    headless_book = _strip_headers(book)
    paragraphs = headless_book.split("\n\n")  # split the book to paragraphs.

    paragraphs_after_cleaning = []
    for par in paragraphs:
        if (
            _is_image(par)
            or _is_footnote(par)
            or _is_email_init(par)
            or _is_books_copy(par)
            or _is_table(par)
            or _is_title_or_etc(par, min_token, max_token)
        ):
            paragraphs_after_cleaning.append(
                "[deleted]"
            )  # if the paragraph is not good , replace it with [deleted]
        else:
            paragraphs_after_cleaning.append(par)

    return paragraphs_after_cleaning
