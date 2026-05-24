"""Step 13 — L5 Dedup + cross-platform clustering.

Sits between L3 (Discovery) and L6 (Triage). Collapses three classes of
duplicates so triage doesn't waste Gemini calls re-judging the same evidence:

  1. **Tracking-param noise** — `?utm_source=…&fbclid=…&gclid=…` produces
     two URLs that point at the same page. Canonicalise: strip the known
     tracking-param list, sort remaining params, drop fragment, lowercase
     host. Identical canonical URLs collapse to one cluster.

  2. **Cross-platform discovery** — the same YouTube Shorts URL might be
     discovered by both the YT Shorts API AND by Brave (via TikTok's
     `/discover/` index, or via Reddit linking to it). Same canonical URL,
     different channels → ONE cluster whose `representative` carries an
     `also_found_on` list of the other channels.

  3. **Syndicated content** — wire-service news articles are reprinted on
     N publisher sites with different URLs. Pass 2 hashes `title + body[:200]`
     (lowercased, whitespace-collapsed) so syndicated copies collapse even
     across hosts.

The orchestrator then sends only the `representative` link from each cluster
to triage. After triage, the cluster's verdict is propagated to every member
(so the UI can show "this verdict applies to 4 sources" / link out).

Representative selection prefers richness:
  - ShortVideoLink > DiscoveredLink   (carries duration / views / transcript)
  - Higher view_count wins ties
  - Earlier discovery wins absolute ties
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .models import ChannelId, DiscoveredLink, ShortVideoLink

log = logging.getLogger(__name__)


# Tracking parameters to drop from URLs. Lowercased; we match case-insensitively.
# Conservative list — keeps anything that might be content-bearing
# (e.g. YouTube's `v` and `t`, Twitter's `s` for "spreading", etc.).
_TRACKING_PARAMS: frozenset[str] = frozenset({
    # Google Analytics / Ads
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_name", "utm_brand", "utm_social", "utm_social-type",
    "gclid", "dclid", "gclsrc", "gbraid", "wbraid", "_gl",
    # Meta family
    "fbclid", "fb_action_ids", "fb_action_types", "fb_source", "fb_ref",
    "igshid", "igsh",
    # Mailchimp / Constant Contact
    "mc_cid", "mc_eid", "mkt_tok",
    # TikTok / X / Twitter
    "_t", "_d", "_r", "_t_t", "ttclid", "is_from_webapp", "sender_device",
    # Generic referrer / share trackers
    "ref", "ref_src", "ref_url", "referrer", "referer", "share",
    "share_id", "share_token", "shareid", "shared_from",
    "source", "src", "from",
    # YouTube share param (the actual video id is `v`, leave it alone)
    "feature", "app", "ab_channel", "embeds_referring_euri",
    "embeds_referring_origin", "embeds_widget_referrer", "si",
    # Branch / Adobe / Hubspot
    "_branch_referrer", "_branch_match_id", "_hsenc", "_hsmi",
    "__hstc", "__hssc", "hsCtaTracking",
    # Pinterest / LinkedIn / Reddit
    "epik", "trk", "trkInfo", "originalSubdomain",
})


# ─── URL canonicalisation ────────────────────────────────────────────────────


def canonical_url(url: str) -> str:
    """Normalise a URL for dedup keying.

    - Lowercase scheme + host
    - Drop tracking params (case-insensitive)
    - Sort remaining params for stable key
    - Drop fragment
    - Drop trailing slash on path (except root)
    - Strip default ports (`:443` for https, `:80` for http)

    Idempotent: `canonical_url(canonical_url(u)) == canonical_url(u)`.
    """
    if not url or "://" not in url:
        return (url or "").strip()
    try:
        p = urlparse(url.strip())
    except Exception:
        return url.strip()

    # Scheme/host normalisation
    scheme = (p.scheme or "https").lower()
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        # Strip leading "www." so `www.youtube.com` and `youtube.com` cluster
        host = host[4:]
    netloc = host
    if p.port:
        default = {"https": 443, "http": 80}.get(scheme)
        if p.port != default:
            netloc = f"{host}:{p.port}"

    # Query: drop trackers, sort the rest, drop empty values for stability
    pairs = parse_qsl(p.query or "", keep_blank_values=False)
    kept = sorted(
        (k, v) for (k, v) in pairs
        if k.lower() not in _TRACKING_PARAMS
    )
    query = urlencode(kept)

    # Path: normalise trailing slash (except root) + collapse double slashes
    path = re.sub(r"/{2,}", "/", p.path or "/")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    return urlunparse((scheme, netloc, path, "", query, ""))


# ─── Content hashing ─────────────────────────────────────────────────────────


def _normalise_text(s: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation outliers."""
    if not s:
        return ""
    out = s.lower()
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def content_hash(link: DiscoveredLink) -> Optional[str]:
    """Stable hash for syndication detection: title + body[:200].

    Returns None when neither title nor snippet is informative — those
    links can only be deduped via canonical URL, not by content.
    """
    title = _normalise_text(link.title or "")
    if isinstance(link, ShortVideoLink):
        # Short videos: title + caption first 200 chars (the snippet field
        # is often a description trim, but caption carries more signal).
        body = _normalise_text((link.caption or link.snippet or "")[:200])
    else:
        body = _normalise_text((link.snippet or "")[:200])

    # Need at least one of title or body, AND at least 16 chars total, before
    # we'll commit to a content hash. Too-short content yields false merges.
    combined = f"{title}\n{body}".strip()
    if len(combined) < 16:
        return None
    return hashlib.sha1(combined.encode("utf-8")).hexdigest()[:16]


# ─── Cluster ─────────────────────────────────────────────────────────────────


@dataclass
class LinkCluster:
    """N discoveries that resolve to the same underlying piece of evidence."""

    cluster_id: str                       # canonical URL or "ch:<hash>"
    representative: DiscoveredLink        # the richest member; goes to triage
    members: List[DiscoveredLink] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def channels(self) -> List[ChannelId]:
        """Distinct channels across all members, representative first."""
        out: List[ChannelId] = [self.representative.channel]
        seen = {self.representative.channel}
        for m in self.members:
            if m.channel not in seen:
                seen.add(m.channel)
                out.append(m.channel)
        return out

    @property
    def cross_platform(self) -> bool:
        """True if discovered via ≥2 distinct channels."""
        return len(self.channels) > 1


# ─── Representative selection ────────────────────────────────────────────────


def _representative_score(link: DiscoveredLink) -> tuple:
    """Higher tuple = better representative.

    Priority:
      1. ShortVideoLink (has rich metadata) > plain DiscoveredLink
      2. Has view_count > no view_count
      3. Higher view_count wins
      4. Has top_comments or transcript > none
      5. Earlier discovery wins (stable ordering tiebreaker)
    """
    is_sv = isinstance(link, ShortVideoLink)
    views = 0
    enrichment = 0
    if is_sv:
        sv = link  # type: ignore[assignment]
        views = sv.view_count or 0
        if sv.top_comments:
            enrichment += 1
        if sv.transcript:
            enrichment += 1
    # discovered_at as POSIX seconds — earlier (smaller) = better, so negate
    age = -(link.discovered_at.timestamp() if link.discovered_at else 0.0)
    return (int(is_sv), views, enrichment, age)


def _pick_representative(members: Sequence[DiscoveredLink]) -> DiscoveredLink:
    if not members:
        raise ValueError("cannot pick representative from empty members list")
    return max(members, key=_representative_score)


# ─── Main entry ──────────────────────────────────────────────────────────────


def cluster_links(links: Sequence[DiscoveredLink]) -> List[LinkCluster]:
    """Two-pass deduplication of a flat link list.

    Pass 1: bucket by `canonical_url(link.url)`. Same URL = same cluster.
    Pass 2: scan unclustered hosts for matching `content_hash` to catch
            syndicated copies across different domains.

    Each output cluster mutates its `representative` to set
    `also_found_on` to the distinct *other* channels in the cluster (the
    representative's own channel is excluded).
    """
    if not links:
        return []

    # ── Pass 1: canonical URL buckets ─────────────────────────────────────
    by_url: dict[str, List[DiscoveredLink]] = {}
    for link in links:
        key = canonical_url(link.canonical_url or link.url)
        by_url.setdefault(key, []).append(link)

    # ── Pass 2: collapse syndicated content across URL clusters ───────────
    # Each URL-cluster might still be a syndication target of another. We
    # hash on the representative of each URL bucket to keep the math cheap.
    by_content: dict[str, List[str]] = {}  # content_hash → [url_key, ...]
    url_to_content: dict[str, str] = {}
    for url_key, members in by_url.items():
        rep = _pick_representative(members)
        ch = content_hash(rep)
        if ch:
            by_content.setdefault(ch, []).append(url_key)
            url_to_content[url_key] = ch

    # Merge URL clusters that share a content hash
    merged_groups: List[List[str]] = []
    seen_keys: set[str] = set()
    for url_key in by_url:
        if url_key in seen_keys:
            continue
        ch = url_to_content.get(url_key)
        if ch and len(by_content[ch]) > 1:
            group = by_content[ch]
            merged_groups.append(group)
            seen_keys.update(group)
        else:
            merged_groups.append([url_key])
            seen_keys.add(url_key)

    # ── Build LinkCluster objects ─────────────────────────────────────────
    clusters: List[LinkCluster] = []
    for group in merged_groups:
        all_members: List[DiscoveredLink] = []
        for url_key in group:
            all_members.extend(by_url[url_key])
        rep = _pick_representative(all_members)
        cluster_id = canonical_url(rep.canonical_url or rep.url) or "cluster_unknown"

        # Set also_found_on on the representative (and only on the representative;
        # cluster members keep their original channel for reference).
        distinct_others: List[ChannelId] = []
        seen_ch: set[ChannelId] = {rep.channel}
        for m in all_members:
            if m.channel not in seen_ch:
                seen_ch.add(m.channel)
                distinct_others.append(m.channel)
        rep.also_found_on = distinct_others

        clusters.append(LinkCluster(
            cluster_id=cluster_id,
            representative=rep,
            members=list(all_members),
        ))

    # Stable ordering: clusters with more members first, then richer
    # representatives. Helps the orchestrator log + triage prioritise.
    clusters.sort(
        key=lambda c: (c.size, _representative_score(c.representative)),
        reverse=True,
    )
    return clusters


def representatives(clusters: Iterable[LinkCluster]) -> List[DiscoveredLink]:
    """Convenience: extract just the representative links (the triage input)."""
    return [c.representative for c in clusters]


def propagate_verdicts_to_members(clusters: Iterable[LinkCluster]) -> None:
    """After triage mutates each cluster's representative, copy the verdict
    onto the cluster's other members so any per-member export carries it too.

    No-op for clusters of size 1.
    """
    for c in clusters:
        if c.size <= 1:
            continue
        rep = c.representative
        for m in c.members:
            if m is rep:
                continue
            m.supports_or_refutes = rep.supports_or_refutes
            m.confidence = rep.confidence
            # Merge signal_tags from rep into member (member keeps its own
            # discovery-time tags; rep's verdict-time tags are union'd in).
            m.signal_tags = sorted(set(m.signal_tags) | set(rep.signal_tags))


# ─── Stats helper (used by orchestrator + smoke test) ────────────────────────


def cluster_stats(clusters: Sequence[LinkCluster]) -> dict:
    """Quick summary for the L5 stage_done event."""
    n_input = sum(c.size for c in clusters)
    n_clusters = len(clusters)
    n_cross_platform = sum(1 for c in clusters if c.cross_platform)
    n_collapsed = n_input - n_clusters
    return {
        "input_links": n_input,
        "clusters": n_clusters,
        "collapsed": n_collapsed,
        "cross_platform_clusters": n_cross_platform,
        "compression_ratio": (round(n_input / n_clusters, 2)
                              if n_clusters else 0.0),
    }
