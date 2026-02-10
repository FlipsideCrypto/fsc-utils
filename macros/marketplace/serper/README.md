# Serper Google Search API Integration

Web search capabilities powered by Serper - the fastest and cheapest Google Search API. Search Google directly from Snowflake for real-time information about blockchains, protocols, and cryptocurrency projects.

## Why Serper?

- **Free tier**: 2,500 Google searches (no credit card required)
- **Simple signup**: Just email/password at [serper.dev](https://serper.dev/)
- **Google results**: Most comprehensive and accurate search results
- **Fast API**: Low-latency responses optimized for AI applications
- **Cost-effective**: After free tier, only $50/month for 50,000 searches

## Setup

### 1. Get Your Serper API Key

1. Go to [serper.dev](https://serper.dev/)
2. Sign up with email/password (no credit card required)
3. Navigate to your dashboard
4. Copy your API key

### 2. Store API Key in Snowflake

Store your Serper API key in Snowflake secrets at `vault/prod/serper`.

### 3. Deploy

```bash
dbt run --models serper__
```

## Functions

### `serper.search(query, options)`

Search Google via Serper API and get raw response.

**Parameters:**
- `query` (STRING): The search query
- `options` (OBJECT): Optional parameters
  - `num`: Number of results (default: 10, max: 100)
  - `gl`: Country code for results (e.g., 'us', 'uk')
  - `hl`: Language code (e.g., 'en', 'es')

**Returns:** VARIANT (Lambda-wrapped response with `data` field containing Serper API response)

### `serper.extract_results(search_response)`

Extract and format search results from raw API response.

**Parameters:**
- `search_response` (VARIANT): Response from `serper.search()`

**Returns:** VARIANT (array of formatted results with title, snippet, url, position)

### `serper.search_and_extract(query, max_results)`

Convenience function that searches and extracts results in one call.

**Parameters:**
- `query` (STRING): The search query
- `max_results` (NUMBER): Maximum results to return (default: 10)

**Returns:** VARIANT (array of formatted Google search results)

## Examples

### Basic Search

```sql
-- Search for blockchain information
SELECT serper.search_and_extract('What is HyperEVM blockchain?', 5) as results;
```

**Output:**
```json
[
  {
    "title": "HyperEVM - Hyperliquid Documentation",
    "snippet": "HyperEVM is an EVM-compatible execution layer built on Hyperliquid...",
    "url": "https://hyperliquid.xyz/hyperevm",
    "position": 1
  },
  ...
]
```

### Research a Protocol

```sql
-- Find information about a DeFi protocol
SELECT serper.search_and_extract('Aave V4 protocol features', 10) as search_results;
```

### Blockchain Classification for Streamline

```sql
-- Use Google search to classify a blockchain project
WITH search_results AS (
  SELECT serper.search_and_extract(
    CONCAT('What type of blockchain is ', 'Monad', '? EVM compatible or alternative L1?'),
    5
  ) as results
),
extracted_info AS (
  SELECT
    result.value:title::STRING as title,
    result.value:snippet::STRING as snippet
  FROM search_results,
  LATERAL FLATTEN(input => results) result
)
SELECT
  LISTAGG(snippet, ' ') as combined_snippets
FROM extracted_info;
```

### Integration with Claude AI

```sql
-- Complete workflow: Google Search → AI Classification
WITH search_results AS (
  SELECT serper.search_and_extract('What is Base blockchain?', 5) as results
),
combined_context AS (
  SELECT
    LISTAGG(
      CONCAT('Title: ', result.value:title::STRING, '\n',
             'Snippet: ', result.value:snippet::STRING),
      '\n\n'
    ) as context
  FROM search_results,
  LATERAL FLATTEN(input => results) result
)
SELECT
  claude.extract_response_text(
    claude.chat_completions([
      OBJECT_CONSTRUCT(
        'role', 'user',
        'content', CONCAT(
          'Based on these Google search results, classify this blockchain into: ',
          'EVM, IBC, ALT_L1, or HIGH_THROUGHPUT. ',
          'Respond with only the classification.\n\n',
          context
        )
      )
    ])
  ) as pod_tag_classification
FROM combined_context;
```

### Batch Research Multiple Blockchains

```sql
-- Research multiple projects at once
WITH projects AS (
  SELECT * FROM VALUES
    ('HyperEVM'),
    ('Monad'),
    ('Berachain')
  AS t(project_name)
),
search_results AS (
  SELECT
    project_name,
    serper.search_and_extract(
      CONCAT('What is ', project_name, ' blockchain? Type and features'),
      3
    ) as results
  FROM projects
)
SELECT
  project_name,
  result.value:title::STRING as source_title,
  result.value:snippet::STRING as description,
  result.value:url::STRING as url,
  result.value:position::NUMBER as rank
FROM search_results,
LATERAL FLATTEN(input => results) result
ORDER BY project_name, rank;
```

### Advanced Search with Location

```sql
-- Search with country and language filters
SELECT serper.search(
  'Ethereum Layer 2 scaling solutions',
  OBJECT_CONSTRUCT(
    'num', 15,
    'gl', 'us',
    'hl', 'en'
  )
) as raw_response;
```

## For Streamline Intelligence Bot

Perfect for researching blockchain projects when users request creation:

```sql
-- Step 1: Search Google for blockchain information
WITH web_research AS (
  SELECT serper.search_and_extract(
    'HyperEVM blockchain type EVM compatible features',
    5
  ) as search_results
),
-- Step 2: Extract and combine information
context AS (
  SELECT
    LISTAGG(result.value:snippet::STRING, ' ') as descriptions
  FROM web_research,
  LATERAL FLATTEN(input => search_results) result
),
-- Step 3: Use Claude AI to classify
classification AS (
  SELECT
    claude.extract_response_text(
      claude.chat_completions([
        OBJECT_CONSTRUCT(
          'role', 'user',
          'content', CONCAT(
            'Based on this information: ', descriptions,
            '\n\nClassify into: EVM, IBC, ALT_L1, or HIGH_THROUGHPUT. ',
            'Also determine if decoder should be included (yes for EVM). ',
            'Respond in JSON: {"pod_tag": "...", "include_decoder": true/false, "reasoning": "..."}'
          )
        )
      ])
    ) as classification_result
  FROM context
)
SELECT
  classification_result:pod_tag::STRING as recommended_pod_tag,
  classification_result:include_decoder::BOOLEAN as recommended_decoder,
  classification_result:reasoning::STRING as reasoning
FROM classification;
```

## Rate Limits & Pricing

- **Free tier**: 2,500 searches (no credit card)
- **After free tier**: $50/month for 50,000 searches ($0.001 per search)
- **Rate limits**: Reasonable limits for typical usage

## Best Practices

1. **Cache results**: Store search results to avoid redundant queries
2. **Combine with AI**: Use search results as context for Claude/LLM analysis
3. **Specific queries**: More specific queries = better Google results
4. **Monitor usage**: Track API calls to stay within free tier
5. **Batch queries**: Process multiple searches efficiently with CTEs

## Error Handling

```sql
-- Check for errors in search response
WITH search AS (
  SELECT serper.search('blockchain query', {}) as response
)
SELECT
  CASE
    WHEN response:status_code::NUMBER != 200 THEN 'API Error'
    WHEN response:data:error IS NOT NULL THEN response:data:error::STRING
    WHEN response:data:organic IS NULL THEN 'No results found'
    WHEN ARRAY_SIZE(response:data:organic) = 0 THEN 'No organic results'
    ELSE 'Success'
  END as status,
  response
FROM search;
```

## Troubleshooting

### "Authentication failed" or "Invalid API key"
- Verify API key is stored correctly in `_FSC_SYS/SERPER`
- Check API key is active in Serper dashboard

### "Rate limit exceeded"
- You've exceeded 2,500 searches on free tier
- Consider upgrading or caching results
- Monitor usage in Serper dashboard

### "No results found"
- Try broader or different search terms
- Check query syntax

## Security

- API keys are securely stored in Snowflake secrets
- All communication uses HTTPS encryption
- No sensitive data is logged

## Comparison with Alternatives

| Provider | Free Tier | API Key Required | Credit Card | Results Quality |
|----------|-----------|-----------------|-------------|-----------------|
| **Serper** | 2,500 searches | ✅ Yes | ❌ No | Google (Best) |
| Brave | 2,000/month | ✅ Yes | ✅ Required | Good |
| DuckDuckGo | Unlimited | ❌ No | ❌ No | Limited (Instant Answers only) |
| Google Custom | 100/day | ✅ Yes | ✅ Yes | Google (Best) |

**Winner**: Serper offers the best balance of free tier, no credit card, and Google-quality results!

## Resources

- [Serper API Documentation](https://serper.dev/docs)
- [Serper Dashboard](https://serper.dev/dashboard)
- [Playground](https://serper.dev/playground) - Test queries without code

## API Response Structure

The Lambda function wraps the Serper API response in a structured format:

```json
{
  "bytes": 4182,
  "data": {
    "organic": [...],      // Array of organic Google search results
    "answerBox": {...},    // Featured snippet or answer box (if available)
    "peopleAlsoAsk": [...], // Related questions
    "relatedSearches": [...], // Related search suggestions
    "searchParameters": {...}
  },
  "headers": {...},
  "status_code": 200
}
```

To access search results, use the `data:organic` path:

- `serper.search()` returns the full wrapped response
- `serper.extract_results()` extracts from `SEARCH_RESPONSE:data:organic`
- `serper.search_and_extract()` handles extraction automatically

## Why Serper for Streamline Intelligence?

Perfect fit because:
1. ✅ **2,500 free searches** - months of usage for the bot
2. ✅ **No credit card** - simple email signup
3. ✅ **Google results** - most comprehensive and accurate
4. ✅ **Fast API** - low latency for real-time bot responses
5. ✅ **AI-optimized** - clean JSON perfect for LLM context
