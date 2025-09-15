import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from pathlib import Path
import tomllib
from loguru import logger
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Save and start
OUTPUT_DIR = Path("/Users/floridomeacci/Documents/HU/MADS-DAV/MADS-DAV/img/florido-images")
TOPK = 15
MSG_THRESHOLD = 700
LEN_THRESHOLD = 50

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_CANDIDATES = [
    SCRIPT_DIR.parent / "config.toml",         # .../MADS-DAV/MADS-DAV/config.toml
    SCRIPT_DIR.parent.parent / "config.toml",  # .../MADS-DAV/config.toml
]
configfile = next((p for p in CONFIG_CANDIDATES if p.exists()), None)
if not configfile:
    tried = "\n".join(str(p) for p in CONFIG_CANDIDATES)
    raise FileNotFoundError(f"config.toml not found. Tried:\n{tried}")

with configfile.open("rb") as f:
    config = tomllib.load(f)

root = configfile.parent
processed = root / Path(config["processed"])
datafile = processed / config["current"]
if not datafile.exists():
    raise FileNotFoundError(f"{datafile} not found. Run preprocessing or fix config.")

df = pd.read_parquet(datafile)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(fig, name: str) -> None:
    out = OUTPUT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    logger.info(f"Saved {out}")

# 1) Most messages
p1 = (
    df[["author", "message"]]
    .groupby("author").count()
    .sort_values("message", ascending=False)
)
topk = p1.head(TOPK)
colors = [0 if x < MSG_THRESHOLD else 1 for x in topk["message"]]
palette = {0: "grey", 1: "blue"}

fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(y=topk.index, x="message", hue=colors, data=topk, palette=palette, legend=False, ax=ax1)
ax1.set_title("Sending the most messages...")
ax1.set_xlabel("message")
save_fig(fig1, "001_most_messages")
plt.close(fig1)

# 2) Longest messages
df["message_length"] = df["message"].str.len()
p_len = (
    df[["author", "message_length"]]
    .groupby("author").mean()
    .sort_values("message_length", ascending=False)
)
topk_len = p_len.head(TOPK)
colors = [0 if x < LEN_THRESHOLD else 1 for x in topk_len["message_length"]]

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(y=topk_len.index, x="message_length", hue=colors, data=topk_len,
            palette=palette, dodge=False, legend=False, ax=ax2)
ax2.set_xlabel("Average message length")
ax2.set_title("Sending the longest messages...")
save_fig(fig2, "002_longest_messages")
plt.close(fig2)

# 3) Most links
df["has_link"] = df["message"].str.contains(r"http")
if df["has_link"].sum() > 0:
    p_link = (
        df[["author", "has_link"]]
        .groupby("author").mean()
        .sort_values("has_link", ascending=False)
    ).head(TOPK)

    imax = int(p_link["has_link"].to_numpy().argmax())
    colors = [1 if i == imax else 0 for i in range(len(p_link))]

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(y=p_link.index, x="has_link", hue=colors, data=p_link,
                palette=palette, dodge=False, legend=False, ax=ax3)
    ax3.set_xlabel("Fraction of messages with a link")
    ax3.set_title("Most links by...")
    save_fig(fig3, "003_most_links")
    plt.close(fig3)
else:
    logger.info("No links found in the messages")

# 4) Most emojis
p_emoji = (
    df[["author", "has_emoji"]]
    .groupby("author").agg(["sum", "mean"])
    .sort_values(("has_emoji", "sum"), ascending=False)
)
p_emoji.columns = p_emoji.columns.droplevel(0)
topk_emoji = p_emoji.sort_values("mean", ascending=False).head(TOPK)

imax = int(topk_emoji["mean"].to_numpy().argmax())
colors = [1 if i == imax else 0 for i in range(len(topk_emoji))]

fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.barplot(y=topk_emoji.index, x="mean", hue=colors, data=topk_emoji,
            palette=palette, dodge=False, legend=False, ax=ax4)
ax4.set_xlabel("Average number of messages with an emoji")
ax4.set_title("Sending the most emoji's")
save_fig(fig4, "004_most_emojis")
plt.close(fig4)