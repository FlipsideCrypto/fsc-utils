{% macro decode_logs_history(
        start,
        stop
    ) %}
    WITH look_back AS (
        SELECT
            block_number
        FROM
            {{ ref("_max_block_by_date") }}
            qualify ROW_NUMBER() over (
                ORDER BY
                    block_number DESC
            ) = 1
    )
SELECT
    l.block_number,
    l._log_id,
    A.abi AS abi,
    OBJECT_CONSTRUCT(
        'topics',
        l.topics,
        'data',
        l.data,
        'address',
        l.contract_address
    ) AS DATA
FROM
    {{ ref("silver__logs") }}
    l
    INNER JOIN {{ ref("silver__complete_event_abis") }} A
    ON A.parent_contract_address = l.contract_address
    AND A.event_signature = l.topics[0]:: STRING
    AND l.block_number BETWEEN A.start_block
    AND A.end_block
WHERE
    (
        l.block_number BETWEEN {{ start }}
        AND {{ stop }}
    )
    AND l.block_number <= (
        SELECT
            block_number
        FROM
            look_back
    )
    AND _log_id NOT IN (
        SELECT
            _log_id
        FROM
            {{ ref("streamline__complete_decode_logs") }}
        WHERE
            (
                block_number BETWEEN {{ start }}
                AND {{ stop }}
            )
            AND block_number <= (
                SELECT
                    block_number
                FROM
                    look_back
            )
    )
{% endmacro %}

{% macro block_reorg(reorg_model_list, hours) %}
  {% set models = reorg_model_list.split(",") %}
  {% for model in models %}
  {% set sql %}
    DELETE FROM
        {{ ref(model) }} t
    WHERE
        t._inserted_timestamp > DATEADD(
            'hour',
            -{{ hours }},
            SYSDATE()
        )
        AND NOT EXISTS (
            SELECT
                1
            FROM
                {{ ref('silver__transactions') }}
                s
            WHERE s.block_number = t.block_number
                AND s.tx_hash = t.tx_hash
        );
    {% endset %}
    {% do run_query(sql) %}
  {% endfor %}
{% endmacro %}

{% macro streamline_external_table_query_v2(
        model,
        partition_function,
        evm_balances=False
    ) %}
    WITH meta AS (
        SELECT
            job_created_time AS _inserted_timestamp,
            file_name,
            {{ partition_function }} AS partition_key
        FROM
            TABLE(
                information_schema.external_table_file_registration_history(
                    start_time => DATEADD('day', -3, CURRENT_TIMESTAMP()),
                    table_name => '{{ source( "bronze_streamline", model) }}')
                ) A
            )
        SELECT
            s.*,
            b.file_name,
            b._inserted_timestamp
            {% if evm_balances %}
            , r.block_timestamp :: TIMESTAMP AS block_timestamp
            {% endif %}
        FROM
            {{ source(
                "bronze_streamline",
                model
            ) }}
            s
            JOIN meta b
            ON b.file_name = metadata$filename
            AND b.partition_key = s.partition_key
            {% if evm_balances %}
            JOIN {{ ref('_block_ranges') }} r
            ON r.block_number = COALESCE(s.VALUE :"BLOCK_NUMBER" :: INT,s.VALUE :"block_number" :: INT)
            {% endif %}
        WHERE
            b.partition_key = s.partition_key
            AND DATA :error IS NULL
            AND DATA IS NOT NULL
{% endmacro %}

{% macro streamline_external_table_FR_query_v2(
        model,
        partition_function,
        partition_column="partition_key",
        evm_balances=False
    ) %}
    WITH meta AS (
        SELECT
            registered_on AS _inserted_timestamp,
            file_name,
            {{ partition_function }} AS partition_key
        FROM
            TABLE(
                information_schema.external_table_files(
                    table_name => '{{ source( "bronze_streamline", model) }}'
                )
            ) A
    )
SELECT
    s.*,
    b.file_name,
    b._inserted_timestamp
    {% if evm_balances %}
    , r.block_timestamp :: TIMESTAMP AS block_timestamp
    {% endif %}
FROM
    {{ source(
        "bronze_streamline",
        model
    ) }}
    s
    JOIN meta b
    ON b.file_name = metadata$filename
    AND b.partition_key = s.{{ partition_column }}
    {% if evm_balances %}
    JOIN {{ ref('_block_ranges') }} r
    ON r.block_number = COALESCE(s.VALUE :"BLOCK_NUMBER" :: INT,s.VALUE :"block_number" :: INT)
    {% endif %}
WHERE
    b.partition_key = s.{{ partition_column }}
    AND DATA :error IS NULL
    AND DATA IS NOT NULL
{% endmacro %}

{% macro streamline_external_table_query_decoder(
        model
    ) %}

    WITH meta AS (

        SELECT
            job_created_time AS _inserted_timestamp,
            file_name,
            CAST(SPLIT_PART(SPLIT_PART(file_name, '/', 6), '_', 1) AS INTEGER) AS _partition_by_block_number,
            TO_DATE(
                concat_ws('-', SPLIT_PART(file_name, '/', 3), SPLIT_PART(file_name, '/', 4), SPLIT_PART(file_name, '/', 5))
            ) AS _partition_by_created_date
        FROM
            TABLE(
                information_schema.external_table_file_registration_history(
                    start_time => DATEADD('day', -3, CURRENT_TIMESTAMP()),
                    table_name => '{{ source( "bronze_streamline", model) }}')
                ) A
            )
        SELECT
            block_number,
            id :: STRING AS id,
            DATA,
            _inserted_timestamp,
            s._partition_by_block_number AS _partition_by_block_number,
            s._partition_by_created_date AS _partition_by_created_date
        FROM
            {{ source(
                "bronze_streamline",
                model
            ) }}
            s
            JOIN meta b
            ON b.file_name = metadata$filename
            AND b._partition_by_block_number = s._partition_by_block_number
            AND b._partition_by_created_date = s._partition_by_created_date
        WHERE
            b._partition_by_block_number = s._partition_by_block_number
            AND b._partition_by_created_date = s._partition_by_created_date
            AND s._partition_by_created_date >= DATEADD('day', -2, CURRENT_TIMESTAMP())
{% endmacro %}

{% macro streamline_external_table_FR_query_decoder(
        model
    ) %}

    WITH meta AS (

        SELECT
            registered_on AS _inserted_timestamp,
            file_name,
            CAST(SPLIT_PART(SPLIT_PART(file_name, '/', 6), '_', 1) AS INTEGER) AS _partition_by_block_number,
            TO_DATE(
                concat_ws('-', SPLIT_PART(file_name, '/', 3), SPLIT_PART(file_name, '/', 4), SPLIT_PART(file_name, '/', 5))
            ) AS _partition_by_created_date
        FROM
            TABLE(
                information_schema.external_table_files(
                    table_name => '{{ source( "bronze_streamline", model) }}'
                )
            ) A
    )
    SELECT
        block_number,
        id :: STRING AS id,
        DATA,
        _inserted_timestamp,
        s._partition_by_block_number AS _partition_by_block_number,
        s._partition_by_created_date AS _partition_by_created_date
    FROM
        {{ source(
            "bronze_streamline",
            model
        ) }}
        s
        JOIN meta b
        ON b.file_name = metadata$filename
        AND b._partition_by_block_number = s._partition_by_block_number
        AND b._partition_by_created_date = s._partition_by_created_date
    WHERE
        b._partition_by_block_number = s._partition_by_block_number
        AND b._partition_by_created_date = s._partition_by_created_date
        AND DATA NOT ILIKE '%"error":%'
{% endmacro %}