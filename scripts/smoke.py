"""Smoke test against a running stack.

Usage: uv run python scripts/smoke.py --key dev-key [--url http://localhost:8000]

The burst check assumes the default rate limit (burst 10, 5 requests per second).
"""

import argparse
import asyncio
import sys

import httpx

BODY = {"model": "mock-model", "prompt": "hello"}


async def run(url: str, key: str) -> int:
    """Run all checks. Return the number of failures."""
    failures = 0

    def check(name: str, ok: bool) -> None:
        nonlocal failures
        print(("PASS " if ok else "FAIL ") + name)
        failures += 0 if ok else 1

    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient(base_url=url, timeout=10) as client:
        check("health -> 200", (await client.get("/health")).status_code == 200)

        no_key = await client.post("/v1/inference", json=BODY)
        check("no key -> 401", no_key.status_code == 401)

        ok = await client.post("/v1/inference", json=BODY, headers=headers)
        check(
            "valid request -> 200 with mock output",
            ok.status_code == 200
            and ok.json()["output"].startswith("mock response to:"),
        )

        empty = await client.post(
            "/v1/inference", json={**BODY, "prompt": ""}, headers=headers
        )
        check("empty prompt -> 422", empty.status_code == 422)

        unknown = await client.post(
            "/v1/inference", json={**BODY, "model": "nope"}, headers=headers
        )
        check("unknown model -> 503", unknown.status_code == 503)

        burst = await asyncio.gather(
            *(
                client.post("/v1/inference", json=BODY, headers=headers)
                for _ in range(40)
            )
        )
        check("burst of 40 -> some 429", 429 in {r.status_code for r in burst})
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    failures = asyncio.run(run(args.url, args.key))
    print(f"\n{failures} check(s) failed" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
