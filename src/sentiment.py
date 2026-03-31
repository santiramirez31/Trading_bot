"""
sentiment.py -- Industry-Aware LLM Sentiment Agent

Two-layer scoring:
  Layer 1 (LLM):      DistilRoBERTa-financial classifies each headline ->
                       base conviction 0-10 (POSITIVE: 5-10, NEGATIVE: 0-5, NEUTRAL: 5.0)
  Layer 2 (Industry): industry_weights.py maps headline to sector-relevant category.
                       Relevance weight amplifies/dampens the base conviction.

Relevance amplification:
  direction           = base_conviction - 5      (signed distance from neutral)
  adjusted_conviction = clamp(5 + direction * weight, 0, 10)

Final combined score (from Strategy PDF Option B):
  tech_signal_norm  = normalized technical trend strength (0-1), passed in from caller
  sentiment_norm    = adjusted_conviction / 10.0  (0-1)
  final_score       = 0.7 * tech_signal_norm + 0.3 * sentiment_norm

Sentiment momentum:
  Splits headlines into recent half vs older half, compares avg conviction.
  Improving = recent avg > older avg + 0.5 -> positive momentum.

Guardrails:
  min_headlines=5, consensus_threshold=70%, recency_decay=0.9x
"""

from transformers import pipeline
from industry_weights import get_sector_weight, TICKER_SECTOR_MAP, normalize_sector, TRUSTED_SOURCES


def _deduplicate_news(news_items: list) -> list:
    """
    Remove near-duplicate headlines before scoring.

    When a major event breaks, 10-15 outlets publish the same story within minutes.
    Without dedup, one real event appears as 15 independent confirmations, inflating
    consensus to 100% from a single source. This normalises on the first 60 chars of
    each headline (lowercase, alphanumeric only) and keeps only the first occurrence.
    """
    seen = set()
    unique = []
    for item in news_items:
        raw = item.get('headline', '')
        key = ''.join(c for c in raw[:60].lower() if c.isalnum() or c == ' ').strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


class SentimentAgent:

    MIN_HEADLINES        = 5
    CONSENSUS_THRESHOLD  = 0.70
    RECENCY_DECAY        = 0.90
    FINAL_SCORE_THRESHOLD = 0.65
    TECH_WEIGHT          = 0.70
    SENTIMENT_WEIGHT     = 0.30

    def __init__(self, model_name: str = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"):
        print(f"Loading sentiment model: {model_name}...")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
        print("Model loaded successfully.")

    def analyze_news(self, news_items: list, sector: str = None, ticker: str = None,
                     tech_signal_norm: float = 0.5, verbose: bool = True) -> dict:
        """
        Score a list of news items for a stock's sector.

        Parameters
        ----------
        news_items       : list of dicts with 'headline' and/or 'summary'
        sector           : GICS sector string (preferred, e.g. 'Technology')
        ticker           : fallback if sector not provided (looked up in TICKER_SECTOR_MAP)
        tech_signal_norm : normalized technical trend strength 0-1 (from data.py)
        verbose          : print per-headline breakdown

        Returns
        -------
        dict:
            score           : float 0-10 (recency-weighted conviction)
            sentiment_norm  : float 0-1 (score / 10)
            consensus_pct   : float 0-1
            momentum        : int +1 (improving) / 0 (stable) / -1 (worsening)
            article_count   : int
            positive_count  : int
            negative_count  : int
            neutral_count   : int
            final_score     : float 0-1 (0.7*tech + 0.3*sentiment)
            tradeable       : bool
        """
        # Resolve and normalize sector (handles Wikipedia long names like 'Information Technology')
        if sector:
            sector = normalize_sector(sector)
        elif ticker:
            sector = TICKER_SECTOR_MAP.get(ticker.upper())

        neutral_result = {
            'score': 5.0, 'sentiment_norm': 0.5, 'consensus_pct': 0.0,
            'momentum': 0, 'article_count': 0, 'positive_count': 0,
            'negative_count': 0, 'neutral_count': 0,
            'final_score': self.TECH_WEIGHT * tech_signal_norm + self.SENTIMENT_WEIGHT * 0.5,
            'tradeable': False,
        }

        label_str = f" for {ticker or sector or 'unknown'}"
        if not news_items:
            if verbose:
                print(f"  [SENTIMENT] No news items provided{label_str}. Defaulting to neutral.")
            return neutral_result

        # --- Guardrail: Deduplication ---
        raw_count = len(news_items)
        news_items = _deduplicate_news(news_items)
        dedup_count = len(news_items)

        # --- Guardrail: Source Whitelist ---
        # Only score headlines from trusted financial news outlets.
        # Unknown blogs/aggregators are filtered out to prevent low-quality
        # sources from tipping consensus. If source field is empty, pass through.
        trusted_items = [
            item for item in news_items
            if not item.get('source') or item['source'] in TRUSTED_SOURCES
        ]
        source_filtered = dedup_count - len(trusted_items)
        news_items = trusted_items

        if verbose:
            print(f"\n  {raw_count} headlines fetched | {dedup_count} after dedup | {len(news_items)} from trusted sources{label_str}")
            print(f"  Sector: {sector or 'unknown'}")
            if source_filtered > 0:
                print(f"  [GUARDRAIL] {source_filtered} headline(s) removed (untrusted source)")

        results = []
        for item in news_items:
            headline = item.get('headline', '')
            summary  = item.get('summary', '')
            text     = (headline + '. ' + summary).strip('. ')
            if not text:
                continue
            try:
                raw        = self.sentiment_pipeline(text[:512])[0]
                label      = raw['label'].lower()
                confidence = raw['score']

                # Layer 1: base conviction (0-10)
                if label == 'positive':
                    base_conviction = 5.0 + confidence * 5.0
                elif label == 'negative':
                    base_conviction = 5.0 - confidence * 5.0
                else:
                    base_conviction = 5.0

                # Layer 2: industry-aware relevance amplification
                weight, category, cat_desc = get_sector_weight(sector or '', headline + ' ' + summary)
                direction           = base_conviction - 5.0
                adjusted_conviction = max(0.0, min(10.0, 5.0 + direction * weight))

                results.append({
                    'headline':      headline[:80],
                    'label':         label,
                    'confidence':    confidence,
                    'base':          base_conviction,
                    'weight':        weight,
                    'conviction':    adjusted_conviction,
                    'category':      category,
                    'cat_desc':      cat_desc,
                    # Time-decay weight injected by sentiment_cache (None → fall back to position decay)
                    'decay_weight':  item.get('_decay_weight', None),
                })
            except Exception as e:
                if verbose:
                    print(f"  [SENTIMENT] Error scoring headline: {e}")

        if not results:
            return neutral_result

        # Guardrail: headline minimum
        if len(results) < self.MIN_HEADLINES:
            if verbose:
                print(f"  [GUARDRAIL] Only {len(results)} headlines (min: {self.MIN_HEADLINES}). Defaulting to neutral.")
            neutral_result['article_count'] = len(results)
            return neutral_result

        if verbose:
            print(f"\n  {'SYM':<4} {'HEADLINE':<52} {'CATEGORY':<22} {'W':>4} {'BASE':>5} {'ADJ':>5}")
            print(f"  {'-'*4} {'-'*52} {'-'*22} {'-'*4} {'-'*5} {'-'*5}")
            for r in results:
                sym = '[+]' if r['label'] == 'positive' else ('[-]' if r['label'] == 'negative' else '[~]')
                # Sanitize headline for terminals that don't support full Unicode (e.g. Windows cp1252)
                safe_hl = r['headline'].encode('ascii', errors='replace').decode('ascii')
                print(f"  {sym:<4} {safe_hl:<52} {r['category']:<22} {r['weight']:>4.1f} {r['base']:>5.1f} {r['conviction']:>5.1f}")

        positive_count = sum(1 for r in results if r['label'] == 'positive')
        negative_count = sum(1 for r in results if r['label'] == 'negative')
        neutral_count  = sum(1 for r in results if r['label'] == 'neutral')
        total          = len(results)

        # Consensus = directional agreement: positive/(positive+negative).
        # Neutral articles are abstentions -- they don't dilute signal agreement.
        # Require at least 3 directional articles for a meaningful consensus;
        # if fewer than 3 are directional, consensus defaults to 0 (block trade).
        directional_count = positive_count + negative_count
        if directional_count >= 3:
            consensus_pct = max(positive_count, negative_count) / directional_count
        else:
            consensus_pct = 0.0

        # Time-decay weighted aggregate score.
        # Uses time-decay weight embedded on each article by sentiment_cache.py
        # (exp(-0.08 * age_hours)), falling back to position-based decay when
        # running outside the cache pipeline (e.g. live scoring).
        weighted_score = 0.0
        total_weight   = 0.0
        for i, r in enumerate(results):
            w = r['decay_weight'] if r['decay_weight'] is not None else self.RECENCY_DECAY ** i
            weighted_score += r['conviction'] * w
            total_weight   += w
        avg_score = weighted_score / total_weight if total_weight > 0 else 5.0

        # Sentiment momentum: compare recent half vs older half
        half = max(1, total // 2)
        recent_avg = sum(r['conviction'] for r in results[:half]) / half
        older_avg  = sum(r['conviction'] for r in results[half:]) / max(1, total - half)
        if recent_avg > older_avg + 0.5:
            momentum = 1
            momentum_label = 'improving'
        elif recent_avg < older_avg - 0.5:
            momentum = -1
            momentum_label = 'worsening'
        else:
            momentum = 0
            momentum_label = 'stable'

        # Final combined score
        sentiment_norm = avg_score / 10.0
        final_score    = self.TECH_WEIGHT * tech_signal_norm + self.SENTIMENT_WEIGHT * sentiment_norm

        tradeable = (
            final_score >= self.FINAL_SCORE_THRESHOLD
            and consensus_pct >= self.CONSENSUS_THRESHOLD
            and len(results) >= self.MIN_HEADLINES
            and momentum >= 0  # do not trade if sentiment is deteriorating
        )

        result = {
            'score':          round(avg_score, 2),
            'sentiment_norm': round(sentiment_norm, 3),
            'consensus_pct':  round(consensus_pct, 2),
            'momentum':       momentum,
            'article_count':  total,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count':  neutral_count,
            'final_score':    round(final_score, 3),
            'tradeable':      tradeable,
        }

        if verbose:
            direction_label = 'BULLISH' if positive_count >= negative_count else 'BEARISH'
            print(f"\n  Headlines : {total}  |  [+] {positive_count}  [-] {negative_count}  [~] {neutral_count}  ({direction_label})")
            print(f"  Consensus : {consensus_pct:.0%}  ({'PASS' if consensus_pct >= self.CONSENSUS_THRESHOLD else 'FAIL'}, threshold 70%)")
            print(f"  LLM Score : {avg_score:.2f}/10  (sentiment_norm={sentiment_norm:.2f})")
            print(f"  Momentum  : {momentum_label} ({'+' if momentum > 0 else ''}{momentum})")
            print(f"  Tech norm : {tech_signal_norm:.2f}  |  Final score: {self.TECH_WEIGHT}*{tech_signal_norm:.2f} + {self.SENTIMENT_WEIGHT}*{sentiment_norm:.2f} = {final_score:.3f}  ({'PASS' if final_score >= self.FINAL_SCORE_THRESHOLD else 'FAIL'}, threshold {self.FINAL_SCORE_THRESHOLD})")
            top_cats = sorted({r['category']: r['weight'] for r in results}.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top categories: {', '.join(f'{c}({w:.1f}x)' for c, w in top_cats)}")

        return result
