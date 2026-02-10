{% macro config_serper_udfs(schema_name = "serper", utils_schema_name="serper_utils") -%}
{#
    This macro is used to generate Serper Search API endpoints
    API Documentation: https://serper.dev/
    Free tier: 2,500 searches (no credit card required)
#}

- name: {{ schema_name }}.search
  signature:
    - [QUERY, STRING, The search query string]
    - [OPTIONS, OBJECT, Optional search parameters (num, location, gl, hl)]
  return_type:
    - "VARIANT"
  options: |
    COMMENT = $$Search Google via Serper API. Returns web results with titles, descriptions, and URLs. Free tier: 2,500 searches. [Serper API docs](https://serper.dev/).$$
  sql: |
    SELECT
      live.udf_api(
        'POST',
        'https://google.serper.dev/search',
        {
          'X-API-KEY': '{API_KEY}',
          'Content-Type': 'application/json'
        },
        OBJECT_CONSTRUCT(
          'q', QUERY,
          'num', COALESCE(OPTIONS:num, 10),
          'gl', COALESCE(OPTIONS:gl::STRING, 'us'),
          'hl', COALESCE(OPTIONS:hl::STRING, 'en')
        ),
        'vault/prod/serper'
      ) as response

- name: {{ schema_name }}.extract_results
  signature:
    - [SEARCH_RESPONSE, VARIANT, The response from serper.search()]
  return_type:
    - "VARIANT"
  options: |
    COMMENT = $$Extract and format web results from Serper API response. Returns array of {title, snippet, url, position}.$$
  sql: |
    SELECT
      TO_VARIANT(ARRAY_AGG(
        OBJECT_CONSTRUCT(
          'title', result.value:title::STRING,
          'snippet', result.value:snippet::STRING,
          'url', result.value:link::STRING,
          'position', result.value:position::NUMBER
        )
      )) as results
    FROM TABLE(FLATTEN(SEARCH_RESPONSE:data:organic)) result

- name: {{ schema_name }}.search_and_extract
  signature:
    - [QUERY, STRING, The search query string]
    - [MAX_RESULTS, NUMBER, Maximum number of results to return (default 10)]
  return_type:
    - "VARIANT"
  options: |
    COMMENT = $$Convenience function that searches Google via Serper and extracts results in one call. Returns formatted array of search results.$$
  sql: |
    WITH search_result AS (
      SELECT {{ schema_name }}.search(
        QUERY,
        OBJECT_CONSTRUCT('num', COALESCE(MAX_RESULTS, 10))
      ) as raw_response
    )
    SELECT TO_VARIANT(ARRAY_AGG(
      OBJECT_CONSTRUCT(
        'title', result.value:title::STRING,
        'snippet', result.value:snippet::STRING,
        'url', result.value:link::STRING,
        'position', result.value:position::NUMBER
      )
    )) as results
    FROM search_result,
    LATERAL FLATTEN(input => raw_response:data:organic) result

{% endmacro %}
