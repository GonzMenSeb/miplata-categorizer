"""Populate own_accounts.miplata_account_id from miplata's accounts table.

Idempotent. Run after a deploy or whenever miplata adds / edits an
account. Requires both DATABASE_URL (categorizer's own DB, read-write)
and MIPLATA_RO_DATABASE_URL (miplata's DB, read-only) to be set in env.

Matching heuristic (deterministic, conservative — prefers "no match"
over a wrong one):

  1. Load every categorizer own_account row (slug, institution, tail,
     aliases).
  2. Load every miplata account (id, bank_id, account_number, alias)
     + join against banks.name.
  3. For each miplata account, resolve its institution (banks.name,
     lowercased + stripped) to one of our known institutions
     ({bancolombia, nequi, coink, ...}).
  4. Extract the tail: last 4 digits of miplata.account_number, OR a
     4-digit token from miplata.alias if account_number is empty.
  5. Match on (institution, tail). If the bank is single-account in our
     own_accounts (e.g. nequi), match by institution alone. If multiple
     candidates remain, log an ambiguity warning and skip.

Usage (inside the container):
    docker exec categorizer-api python /app/scripts/sync_miplata_accounts.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from collections import defaultdict

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

_BANK_ALIASES = {
    "bancolombia": "bancolombia",
    "banco de colombia": "bancolombia",
    "nequi": "nequi",
    "coink": "coink",
    "davivienda": "davivienda",
    "bbva": "bbva",
}

_TAIL_RE = re.compile(r"(\d{4})(?!.*\d)")


def _institution_from_bank_name(bank_name: str | None) -> str | None:
    if not bank_name:
        return None
    key = bank_name.strip().lower()
    if key in _BANK_ALIASES:
        return _BANK_ALIASES[key]
    for alias, inst in _BANK_ALIASES.items():
        if alias in key:
            return inst
    return None


def _extract_tail(account_number: str | None, alias: str | None) -> str | None:
    for source in (account_number, alias):
        if not source:
            continue
        m = _TAIL_RE.search(source)
        if m:
            return m.group(1)
    return None


async def _fetch_miplata_accounts(url: str) -> list[dict]:
    eng = create_async_engine(url)
    try:
        async with eng.connect() as c:
            rows = await c.execute(
                text(
                    "SELECT a.id, a.account_number, a.alias, b.name AS bank_name "
                    "FROM accounts a LEFT JOIN banks b ON b.id = a.bank_id"
                )
            )
            return [dict(r._mapping) for r in rows]
    finally:
        await eng.dispose()


async def _fetch_own_accounts(url: str) -> list[dict]:
    eng = create_async_engine(url)
    try:
        async with eng.connect() as c:
            rows = await c.execute(
                text(
                    "SELECT slug, institution, account_number_tail, miplata_account_id "
                    "FROM own_accounts"
                )
            )
            return [dict(r._mapping) for r in rows]
    finally:
        await eng.dispose()


async def _update_mapping(url: str, slug: str, miplata_id) -> None:
    eng = create_async_engine(url)
    try:
        async with eng.begin() as c:
            await c.execute(
                text("UPDATE own_accounts SET miplata_account_id = :mid WHERE slug = :slug"),
                {"mid": miplata_id, "slug": slug},
            )
    finally:
        await eng.dispose()


def _match(
    mip: dict, by_institution: dict[str, list[dict]]
) -> tuple[dict | None, str]:
    institution = _institution_from_bank_name(mip.get("bank_name"))
    if institution is None:
        return None, f"unknown institution from bank_name={mip.get('bank_name')!r}"
    candidates = by_institution.get(institution, [])
    if not candidates:
        return None, f"no own_account with institution={institution!r}"
    tail = _extract_tail(mip.get("account_number"), mip.get("alias"))
    if len(candidates) == 1:
        return candidates[0], f"sole-candidate by institution={institution!r}"
    if tail is None:
        return (
            None,
            f"institution={institution!r} has {len(candidates)} candidates "
            f"and no tail could be extracted from account_number / alias",
        )
    exact = [c for c in candidates if (c.get("account_number_tail") or "") == tail]
    if len(exact) == 1:
        return exact[0], f"matched by institution+tail ({institution}, {tail})"
    if len(exact) > 1:
        return (
            None,
            f"ambiguous: institution={institution!r} tail={tail!r} "
            f"matches {len(exact)} own_accounts",
        )
    return None, f"no own_account with institution={institution!r} tail={tail!r}"


async def main() -> int:
    dsn = os.environ.get("DATABASE_URL")
    ro_dsn = os.environ.get("MIPLATA_RO_DATABASE_URL")
    if not dsn:
        print("DATABASE_URL not set", file=sys.stderr)
        return 2
    if not ro_dsn:
        print("MIPLATA_RO_DATABASE_URL not set", file=sys.stderr)
        return 2

    own = await _fetch_own_accounts(dsn)
    miplata = await _fetch_miplata_accounts(ro_dsn)
    print(f"own_accounts: {len(own)} rows")
    print(f"miplata accounts: {len(miplata)} rows")

    by_institution: dict[str, list[dict]] = defaultdict(list)
    for r in own:
        by_institution[r["institution"]].append(r)

    matched = 0
    unmatched: list[tuple[dict, str]] = []
    for mip in miplata:
        target, reason = _match(mip, by_institution)
        if target is None:
            unmatched.append((mip, reason))
            continue
        await _update_mapping(dsn, target["slug"], mip["id"])
        matched += 1
        print(
            f"  matched miplata={mip['id']} bank={mip.get('bank_name')!r} "
            f"tail-source={mip.get('account_number') or mip.get('alias')!r} "
            f"-> slug={target['slug']} ({reason})"
        )

    if unmatched:
        print(f"\nUnmatched miplata accounts ({len(unmatched)}):")
        for mip, reason in unmatched:
            print(
                f"  id={mip['id']} bank={mip.get('bank_name')!r} "
                f"account_number={mip.get('account_number')!r} "
                f"alias={mip.get('alias')!r} -> {reason}"
            )

    print(f"\nSummary: matched={matched}, unmatched={len(unmatched)}, "
          f"own_accounts_total={len(own)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
