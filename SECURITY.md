# Security policy

## Reporting a vulnerability

Do not open a public issue for a vulnerability. Report it privately through
GitHub's private vulnerability reporting, which will be enabled when the
repository is public: the "Report a vulnerability" button under this
repository's Security tab. Include the steps to reproduce it. Use made-up
data only, never data about real people.

There is no release yet. Security fixes go into the next release.

## What to know before hosting it

Group Diversifier has no login and no user accounts. Run it on your own
machine, or put a reverse proxy that handles login and TLS in front of it;
`compose.yaml` and the files in `deploy/` say how. A roster holds personal
data. Anyone who can open the app can upload data and use it; each browser
session sees only its own uploads, but the server keeps uploaded files in its
memory until it restarts, so whoever runs the server can reach them.
