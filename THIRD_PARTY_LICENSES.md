# Third-party licences

Group Diversifier is AGPL-3.0-or-later (see `LICENSE`). The container image
installs the packages below, which keep their own licences. The list is every
package pinned in `requirements.lock.txt`, with the version the lock pins and
the licence each package declares in its installed metadata. The table is
generated from that metadata by `scripts/make_third_party_notices.py`, not
written by hand.

A licence that the metadata does not declare is listed as "not determined".
Check that package's own project before relying on its licence.

| Package | Version | Declared licence | Project |
|---|---|---|---|
| altair | 6.3.0 | BSD License | https://altair-viz.github.io |
| anyio | 4.16.0 | MIT | https://anyio.readthedocs.io/en/latest/ |
| attrs | 26.1.0 | MIT | https://www.attrs.org/ |
| certifi | 2026.11.20 | MPL-2.0 | https://github.com/certifi/python-certifi |
| charset-normalizer | 3.5.3 | MIT | https://github.com/jawah/charset_normalizer |
| click | 8.5.1 | BSD-3-Clause | https://click.palletsprojects.com/page/changes/ |
| et-xmlfile | 2.0.0 | MIT | https://foss.heptapod.net/openpyxl/et_xmlfile |
| h11 | 0.16.0 | MIT | https://github.com/python-hyper/h11 |
| idna | 3.20 | BSD-3-Clause | https://github.com/kjd/idna |
| itsdangerous | 2.2.0 | BSD License | https://itsdangerous.palletsprojects.com/changes/ |
| jinja2 | 3.1.6 | BSD License | https://jinja.palletsprojects.com/changes/ |
| jsonschema | 4.26.0 | MIT | https://github.com/python-jsonschema/jsonschema |
| jsonschema-specifications | 2025.9.1 | MIT | https://jsonschema-specifications.readthedocs.io/ |
| markupsafe | 3.0.4 | BSD-3-Clause | https://palletsprojects.com/donate |
| narwhals | 2.29.0 | MIT | https://github.com/narwhals-dev/narwhals |
| networkx | 3.7 | BSD-3-Clause | https://networkx.org/ |
| numpy | 2.5.5 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org |
| openpyxl | 3.1.5 | MIT | https://openpyxl.readthedocs.io |
| packaging | 26.4 | Apache-2.0 OR BSD-2-Clause | https://packaging.pypa.io/ |
| pandas | 3.1.1 | BSD-3-Clause | https://pandas.pydata.org |
| pillow | 13.0.0 | MIT-CMU | https://pillow.readthedocs.io/en/stable/releasenotes/index.html |
| plotly | 7.2.0 | MIT | https://plotly.com/python/ |
| protobuf | 7.36.2 | 3-Clause BSD License | https://developers.google.com/protocol-buffers/ |
| pyarrow | 26.0.1 | Apache-2.0 | https://arrow.apache.org/ |
| pydeck | 0.9.3 | Apache License 2.0 | https://github.com/visgl/deck.gl |
| python-dateutil | 4.0.0 | BSD-3-Clause | https://github.com/dateutil/dateutil |
| python-multipart | 0.0.34 | Apache-2.0 | https://github.com/Kludex/python-multipart |
| pyyaml | 6.0.4 | MIT | https://pyyaml.org/ |
| referencing | 0.37.0 | MIT | https://referencing.readthedocs.io/ |
| requests | 2.34.2 | Apache-2.0 | https://requests.readthedocs.io |
| rpds-py | 2026.9.1 | MIT | https://rpds.readthedocs.io/ |
| scipy | 1.18.1 | BSD License | https://scipy.org/ |
| starlette | 1.10.0 | BSD-3-Clause | https://github.com/Kludex/starlette |
| streamlit | 1.67.0 | Apache-2.0 | https://streamlit.io |
| typing-extensions | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions |
| tzdata | 2026.6 | Apache-2.0 | https://github.com/python/tzdata |
| urllib3 | 2.8.0 | MIT | https://github.com/urllib3/urllib3 |
| uvicorn | 0.54.0 | BSD-3-Clause | https://uvicorn.dev/release-notes |
| watchdog | 7.1.1 | Apache-2.0 | https://github.com/gorakhargosh/watchdog |
| websockets | 17.4 | BSD-3-Clause | https://github.com/python-websockets/websockets |

The development lock adds pytest and ruff for development. They are not in the
image and are not listed.
