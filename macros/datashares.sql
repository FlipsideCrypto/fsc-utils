{%- macro get_ancestors(node, include_depth=false, exclude_source=false) -%}
{#
    Return a list of ancestors for a node in a DAG.
 #}
    {%- for dep in node.depends_on.nodes | unique | list  recursive %}
        {% if dep.startswith("model.") and "bronze__" not in dep %}
            "{{- loop.depth0 ~ '-'if include_depth else '' }}{{node.config.materialized }}-{{ dep -}}",
            {{- loop(graph.nodes[dep].depends_on.nodes) -}}
        {% elif not exclude_source %}
            "{{- loop.depth0 ~ '-'if include_depth else '' }}{{node.config.materialized }}-{{ dep -}}",
        {%- endif -%}
    {%- endfor %}
{%- endmacro -%}

{% macro get_view_ddl() %}
{#
    Return a dictionary of view names and their DDL statements.
    The DDL statements are escaped to be used in a Snowflake query.
    The dictionary is converted to JSON to be used in a dbt macro..
 #}
    {% if execute %}
        {% set query %}
            SELECT
            CONCAT_WS('.', TABLE_SCHEMA, TABLE_NAME) as VIEW_NAME,
            VIEW_DEFINITION
            FROM {{target.database}}.INFORMATION_SCHEMA.VIEWS
            WHERE TABLE_SCHEMA NOT IN ('INFORMATION_SCHEMA', 'STREAMLINE')
            AND TABLE_SCHEMA NOT LIKE 'TEST_%'
        {%- endset -%}
        {%- set results = run_query(query) -%}
        {% set ddl = {} %}
        {% for key, value in results.rows %}
          {%- do ddl.update({key: value|replace("$$", "\$\$")}) -%}
        {%- endfor -%}
        {{- tojson(ddl) -}}
    {%- endif -%}
{%- endmacro -%}

{% macro replace_database_references(references_to_replace, ddl) %}
{#
    Return the DDL statement for a view with the references replaced.
    All items in references_to_replace are replaced with __NEW__,
    while all remaining references to the database are replaced with __SOURCE__.

    references_to_replace: a list of references to replace with __NEW__
    ddl: the DDL statement to be replaced

#}
    {% set re = modules.re %}
    {% set outer = namespace(replaced=ddl) %}
    {% for key in references_to_replace %}
        {%- set original = re.compile("\\b" ~ target.database ~ "." ~ key ~ "\\b", re.IGNORECASE) -%}
        {%- set replacement  =  "__NEW__" ~ "." ~ key -%}
        {% set outer.replaced = original.sub(replacement, outer.replaced) %}
    {%- endfor -%}
    {%- set original = re.compile("\\b" ~ target.database ~ "." ~ "\\b", re.IGNORECASE) -%}
    {% set outer.replaced = original.sub("__SOURCE__.", outer.replaced) %}
    {{- outer.replaced -}}
{%- endmacro -%}

{% macro generate_view_ddl(dag, schema) %}
{#
    Return a list of DDL statements for views in a DAG.

    dag: a DAG of views
    schema: schemas to create schema DDL for
 #}
    {%- set ddl =  fromjson(fsc_utils.get_view_ddl())  -%}
    {%- set created = {} -%}
    {%- set final_text = [] -%}
    {%- for view, deps in dag.items() -%}
        {%- for d in deps -%}
            {%- set table_name = d.split(".")[-1].replace("__", ".").upper() -%}
            {%- if ddl.get(table_name) and table_name not in created -%}
                {%- set replaced = fsc_utils.replace_database_references(ddl.keys(), ddl[table_name]) -%}
                {%- do final_text.append(replaced) -%}
                {%- do created.update({table_name:true}) -%}
            {%- endif -%}
        {%- endfor -%}
    {%- endfor -%}
    {%- set schema_ddl = [] -%}
    {%- for s in schema -%}
        {%- do schema_ddl.append("CREATE SCHEMA IF NOT EXISTS __NEW__." ~ s ~ ";") -%}
    {%- endfor -%}
    {{- toyaml(schema_ddl + final_text) -}}
{%- endmacro -%}

{% macro generate_dag_and_schemas(node_paths, materializations) %}
{#
    Return a DAG of views and a list of schemas to create.

    node_paths: a list of node paths to include in the DAG
    materializations: a list of materializations to include in the DAG
 #}
    {%- set dag = {} -%}
    {%- set schema = [] -%}
    {%- for key, value in graph.nodes.items() -%}
        {%
        if value.refs
        and set(value.fqn).intersection(node_paths)
        and value.config.materialized in materializations
        and value.config.enabled
        and not value.sources
        and not key.endswith("_create_gold")
        -%}
        {%- set name = value.schema + "." + value.alias -%}
        {%- set _result = fromyaml("[" ~ fsc_utils.get_ancestors(value, exclude_source=true)[:-1] ~ "]") -%}
            {% if _result -%}
                {%- do _result.insert(0, key) -%}
                {%- do dag.update({name.upper() : _result | reverse|list})  -%}
                {% for d in _result -%}
                    {%- if d.split(".")[-1].split("__")[0] not in schema -%}
                        {%- do schema.append(d.split(".")[-1].split("__")[0]) -%}
                    {%- endif -%}
                {%- endfor -%}
            {%- else -%}
                {%- do dag.update({name.upper() : [key] }) -%}
                {%- if value.schema not in schema -%}
                    {%- do schema.append(value.schema) -%}
                {%- endif -%}
            {%- endif -%}
        {%- endif -%}
    {%- endfor -%}
    {%- set final = {"dag": dag, "schema": schema} -%}
    {{- tojson(final) -}}
{%- endmacro -%}

{% macro generate_table_views_ddl(tables, schema) %}
{#
    Return a list of DDL statements for views of tables from a list.

    tables: a list of tables to create views for
    schema: schemas to create schema DDL for
 #}
    {%- set schema_ddl = [] -%}
    {%- set view_ddl = [] -%}
    {% for s in schema %}
        {%- do schema_ddl.append("CREATE SCHEMA IF NOT EXISTS __NEW__." ~ s ~ ";") -%}
    {%- endfor -%}
    {% for table in tables %}
        {%- do view_ddl.append("CREATE OR REPLACE VIEW __NEW__." ~ table ~ " copy grants AS SELECT * FROM " ~ "__SOURCE__." ~ table ~";") -%}
    {%- endfor -%}
    {{- toyaml(schema_ddl + view_ddl) -}}
{%- endmacro -%}

{% macro generate_datashare_ddl() %}
{#
    generate DDL for datashare

    Return: DDL for datashare
 #}
    {%- set gold_views = fromjson(fsc_utils.generate_dag_and_schemas(["gold"], ["view"])) -%}
    {%- set gold_views_ddl = fromyaml(fsc_utils.generate_view_ddl(gold_views["dag"], gold_views["schema"])) -%}
    {%- set gold_tables = fromjson(fsc_utils.generate_dag_and_schemas(["gold"], ["incremental", "table"])) -%}
    {%- set gold_tables_ddl = fromyaml(fsc_utils.generate_table_views_ddl(gold_tables["dag"].keys(), gold_tables["schema"])) -%}
    {%- set combined_ddl = gold_views_ddl + gold_tables_ddl -%}
    {%- do combined_ddl.insert(0, "CREATE DATABASE IF NOT EXISTS __NEW__;") -%}
    {{- "BEGIN\n" ~ (combined_ddl | join("\n")) ~ "\nEND" -}}
{%- endmacro -%}

{% macro generate_datashare_udf_ddl() %}
{#
    generate UDF DDL for datashare

    Return: UDF DDL for datashare
 #}
    {%- set schema = "UTILS" -%}
    {%- set udfs = fromyaml(fsc_utils.udf_configs(schema)) -%}
    {%- set combined_ddl = [] -%}
    {%- for udf in udfs -%}
        {% set name_ = udf ["name"] %}
        {% set signature = udf ["signature"] %}
        {% set return_type = udf ["return_type"] %}
        {% set sql_ = udf ["sql"] %}
        {% set options = udf ["options"] %}
        {% set api_integration = udf ["api_integration"] %}
        {% set func_type = udf ["func_type"] %}
        {% set exclude_from_datashare = udf.get("exclude_from_datashare",False) %}
        {% if not exclude_from_datashare %}
            {%- set udf_ddl = fsc_utils.create_sql_function(
                name_ = name_,
                signature = signature,
                return_type = return_type,
                sql_ = sql_,
                options = options,
                api_integration = api_integration,
                func_type = func_type
            ).replace("\\","\\\\").replace("'","\\'") -%}
            {%- do combined_ddl.append(udf_ddl) -%}       
        {% endif %} 
    {% endfor %}
    {%- do combined_ddl.insert(0, "CREATE DATABASE IF NOT EXISTS __NEW__;") -%}
    {%- do combined_ddl.insert(1, "USE DATABASE __NEW__;") -%}
    {%- do combined_ddl.insert(2, "CREATE SCHEMA IF NOT EXISTS "~schema~";") -%}
    {{- "'BEGIN','" ~ (combined_ddl | join("','")) ~ "','END'" -}}
{%- endmacro -%}