import re
import sys
import tomllib
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
import json

import click
import pandas as pd
from loguru import logger

from wa_analyzer.settings import (BaseRegexes, Folders, PreprocessConfig,
                                  androidRegexes, csvRegexes, iosRegexes,
                                  oldRegexes)
from wa_analyzer.humanhasher import humanize

logger.remove()
logger.add("logs/logfile.log", rotation="1 week", level="DEBUG")
logger.add(sys.stderr, level="INFO")

logger.debug(f"Python path: {sys.path}")


class WhatsappPreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.folders = config.folders
        self.regexes = config.regexes
        self.datetime_format = config.datetime_format
        self.drop_authors = config.drop_authors

    def __call__(self):
        records, _ = self.process()
        self.save(records)

    @staticmethod
    def normalize_author(name: str) -> str:
        """Return a canonical representation of an author name.

        This reduces spurious duplicates caused by invisible characters,
        tildes/prefixes, and odd unicode spaces.

        Steps:
        - Unicode normalize to NFKC (compatibility composition)
        - Remove zero-width chars (ZWSP, ZWJ, ZWNJ, BOM)
        - Replace non-breaking space variants with regular spaces
        - Remove leading tilde variants and adjacent spaces (seen in exports)
        - Collapse internal whitespace and strip ends

        We intentionally do NOT change case or strip diacritics to preserve
        readability of display names while still unifying common variants.
        """
        if not isinstance(name, str):
            return name
        s = unicodedata.normalize("NFKC", name)
        # Remove zero-width characters
        s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
        # Normalize common non-breaking spaces to regular spaces
        s = s.replace("\u00A0", " ")  # NBSP
        s = s.replace("\u202F", " ")  # NNBSP (narrow no-break space)
        # Remove leading tilde variants and any following spaces
        s = re.sub(r"^[~\u223C]\s*", "", s)
        # Collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def save(self, records: list[tuple]) -> Path:
        df = pd.DataFrame(records, columns=["timestamp", "author", "message"])
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        outfile = self.folders.processed / f"whatsapp-{now}.csv"
        logger.info(f"Writing to {outfile}")
        df.to_csv(outfile, index=False)
        # Also write anonymized reference mapping built from ORIGINAL authors
        try:
            authors = df["author"].dropna().unique().tolist()
            anon = {k: humanize(k) for k in authors}
            ref = {v: k for k, v in anon.items()}  # anonymized -> original
            reference_file = self.folders.processed / "anon_reference.json"
            with reference_file.open("w", encoding="utf-8") as f:
                json.dump({k: ref[k] for k in sorted(ref.keys())}, f, ensure_ascii=False, indent=2)
            logger.info(f"Wrote anonymization reference to {reference_file}")
        except Exception as e:
            logger.warning(f"Could not write anon_reference.json: {e}")
        logger.success("Done!")

        return outfile

    def process(self) -> tuple:
        records = []
        appended = []
        datafile = self.folders.raw / self.folders.datafile

        tsreg = self.regexes.timestamp
        messagereg = self.regexes.message
        authorreg = self.regexes.author

        with datafile.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f.readlines()):
                ts = re.match(tsreg, line)
                if ts:
                    try:
                        timestamp = datetime.strptime(
                            ts.groups()[0], self.datetime_format
                        ).replace(tzinfo=timezone.utc)
                    except ValueError as e:
                        logger.error(
                            f"Error while processing timestamp of line {line_number}: {e}"
                        )
                        continue
                    msg_ = re.search(messagereg, line)
                    author_ = re.search(authorreg, line)
                    if msg_ is None:
                        logger.error(
                            f"Could not find a message for line {line_number}. Please check the data and / or the message regex"
                        )
                        continue
                    if author_ is None:
                        logger.error(
                            f"Could not find an author for line {line_number}. Please check the data and / or the author regex"
                        )
                        continue
                    raw_author = author_.groups()[0].strip()
                    author = self.normalize_author(raw_author)
                    if any(drop_author in author for drop_author in self.drop_authors):
                        logger.warning(f"Skipping author {author}")
                        continue
                    msg = msg_.groups()[0].strip()
                    records.append((timestamp, author, msg))
                elif len(records) > 0:
                    appended.append(timestamp)
                    msg += " " + line.strip()
                    records[-1] = (timestamp, author, msg)

        logger.info(f"Found {len(records)} valid records")
        logger.info(f"Appended {len(appended)} records")
        return records, appended


@click.command()
@click.option(
    "--device", default="android", help="Device type: iOS, Android, old, or csv"
)
def main(device: str):
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
        raw = Path(config["raw"])
        processed = Path(config["processed"])
        datafile = Path(config["input"])
        datetime_format = config["datetime_format"]
        drop_authors = config["drop_authors"]

    if device.lower() == "ios":
        logger.info("Using iOS regexes")
        regexes: BaseRegexes = iosRegexes
    elif device.lower() == "old":
        logger.info("Using old version regexes")
        regexes: BaseRegexes = oldRegexes  # type: ignore
    elif device.lower() == "csv":
        logger.info("Using CSV regexes")
        regexes: BaseRegexes = csvRegexes  # type: ignore
    else:
        logger.info("Using Android regexes")
        regexes: BaseRegexes = androidRegexes  # type: ignore

    if not (raw / datafile).exists():
        logger.error(f"File {raw / datafile} not found")
    else:
        logger.info(f"Reading from {raw / datafile}")

    folders = Folders(
        raw=raw,
        processed=processed,
        datafile=datafile,
    )
    preprocessconfig = PreprocessConfig(
        folders=folders,
        regexes=regexes,
        datetime_format=datetime_format,
        drop_authors=drop_authors,
    )
    preprocessor = WhatsappPreprocessor(preprocessconfig)
    preprocessor()


if __name__ == "__main__":
    main()
