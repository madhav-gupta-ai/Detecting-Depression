# Dataset

989 Reddit text posts (title + body, one post per `.txt` file), collected with
`src/collect.py` via the official Reddit API (PRAW) in 2023:

| Folder | Posts | Contents |
|---|---|---|
| `Depression/` | 291 | Posts from users of r/depression with a detected self-reported diagnosis ("Diagnosis Post"), taken from mental-health subreddits within 365 days after the diagnosis post (at most the 5 most verbose posts per user, minimum 60 characters). |
| `Control/` | 698 | Posts from r/casualconversation users whose entire post history contains no mental-health subreddit activity and no mental-health phrases. |
| `Depression_Clean/`, `Control_Clean/` | 291 + 698 | The same posts after keyword cleaning (`src/clean.py`): all occurrences of the condition, diagnosis and doctor keywords are removed, so classifiers cannot rely on explicit clue words. |
| `support/` | — | Hand-crafted keyword and phrase lists driving collection and cleaning (see below). |

Author anonymity is preserved: usernames are only ever handled as SHA-1 hashes,
subreddit moderators are excluded, and nothing but the post title and body text
is stored.

Note that Reddit content changes over time (posts get deleted, users leave),
so re-running `src/collect.py` reproduces the collection *method*, not this
exact dataset. The committed posts are the dataset used in the published study,
and all committed feature files and results derive from them.

## `support/` files

- `API_credentials_example.txt` — template for the Reddit API credentials.
  Copy it to `API_credentials.txt` (gitignored) and fill in your own client id,
  secret and user agent to run `src/collect.py`.
- `condition_words.txt` — names of the condition (depression and variants)
- `diagnosis_words.txt` — diagnosis-related keywords (also used as search queries)
- `doctor_words.txt` — mental-health professions (psychiatrist, therapist, ...)
- `positive_diagnosis.txt` — phrases indicating a self-reported diagnosis
- `negative_diagnosis.txt` — phrases that would cause false positives
  (e.g. "my mother was diagnosed"), used to reject posts
- `mh_patterns.txt` — general mental-health phrases used to screen control users
- `mh_subreddits.txt` — the mental-health subreddits considered
